# -*- coding: utf-8 -*-
"""
app.py (serving)
作者：Accilia
创建时间：2026-02-26
用途说明：
- /search   : dense-only（保留原接口）
- /retrieve : hybrid retrieval（BM25 + FAISS + fusion + boost + rerank）
- /compute_similarity : 计算相似度
- /compute_bertscore : 计算BERTScore
"""

from __future__ import annotations

import os
import json
import sqlite3
import time
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 导入BERTScore
from bert_score import BERTScorer

# transformers offline flags (keep your original style)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/openbayes/home/.cache/huggingface")

from deepsea_qa.configs import paths
from deepsea_qa.configs.retrieval_config import RetrievalConfig
from deepsea_qa.retrieval.sparse import SparseRetriever
from deepsea_qa.retrieval.dense import DenseRetriever
from deepsea_qa.retrieval.fusion import FusionModule
from deepsea_qa.retrieval.boost import ClassificationBooster
from deepsea_qa.retrieval.rerank import Reranker
from deepsea_qa.retrieval.pipeline import RetrievalPipeline

app = FastAPI(title="DeepSea QA Retrieval Service", version="0.2.0")

# -----------------------------
# 环境变量（服务侧）
# -----------------------------
KB_DIR = os.environ.get("KB_DIR", "/openbayes/home/deepsea_qa/artifacts/kb")
EMB_MODEL_DIR = os.environ.get("EMB_MODEL_DIR", "/openbayes/home/bge-m3")
RERANK_MODEL_DIR  = os.environ.get("RERANK_MODEL_DIR", "/openbayes/home/bge-reranker-v2-m3")
BERTSCORE_MODEL_DIR = os.environ.get("BERTSCORE_MODEL_DIR","/openbayes/home/bert-base-chinese")

# 将 paths 的相对路径“映射到 KB_DIR”
BM25_PKL_PATH = os.path.join(KB_DIR, "bm25", "bm25.pkl")
FAISS_INDEX_PATH = os.path.join(KB_DIR, "faiss", "index_bge-m3_ip.faiss")
FAISS_IDMAP_PATH = os.path.join(KB_DIR, "faiss", "index_bge-m3_ip.id_map.jsonl")
SQLITE_PATH = os.path.join(KB_DIR, "store", "chunks.sqlite")

def _assert_exists(p: str, is_dir: bool = False):
    if is_dir:
        assert os.path.isdir(p), f"[ERROR] Directory not found: {p}"
    else:
        assert os.path.isfile(p), f"[ERROR] File not found: {p}"

_assert_exists(KB_DIR, is_dir=True)
_assert_exists(EMB_MODEL_DIR, is_dir=True)
_assert_exists(BM25_PKL_PATH)
_assert_exists(FAISS_INDEX_PATH)
_assert_exists(FAISS_IDMAP_PATH)
_assert_exists(SQLITE_PATH)

# 响应封装函数
def _resp_template(stage: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "stage": stage,           # "search" | "retrieve"
        "query": {
            "original": "",
            "dense": [],
            "sparse": [],
        },
        "retrieval": {
            "chunk_ids": [],
            "scores": [],
            "score_type": "",
            "scores_raw": [],
            "candidates": [],     # debug 用（可为空）
        },
        "evidence": [],           # 可为空
        "meta": {
            "fusion": {},
            "boost": {},
            "rerank": {},
            "timing": {"seconds": 0.0},
        },
        "error": None,
    }

# -----------------------------
# 启动时加载：Embedding + FAISS + Pipeline
# -----------------------------
embedder = SentenceTransformer(EMB_MODEL_DIR, device="cuda", trust_remote_code=True)

# DenseRetriever 内部会 load faiss + id_map
dense = DenseRetriever(embedder=embedder, faiss_index_path=FAISS_INDEX_PATH, faiss_idmap_path=FAISS_IDMAP_PATH)

# SparseRetriever load bm25.pkl
sparse = SparseRetriever(bm25_pkl_path=BM25_PKL_PATH)

# 默认配置（你后续要调参就改 configs/retrieval_config.py）
cfg = RetrievalConfig()

fusion = FusionModule(strategy=cfg.fusion.strategy, alpha=cfg.fusion.alpha, normalize=cfg.fusion.normalize)

booster = ClassificationBooster(
    enable=cfg.boost.enable,
    mode=cfg.boost.mode,
    use_top_k=cfg.boost.use_top_k,
    boost_weight=cfg.boost.boost_weight,
)

reranker = None
if cfg.rerank.enable:
    # 使用环境变量指定的模型目录加载重排模型
    reranker = Reranker(
        model_name=RERANK_MODEL_DIR,
        device=cfg.rerank.device,
        fp16=cfg.rerank.fp16,
        max_length=cfg.rerank.max_length,
    )

# -----------------------------
# SQLite helpers
# -----------------------------
def _fetch_text_map(chunk_ids: List[str]) -> Dict[str, str]:
    """
    为 rerank 提供 passage 文本（只取 text，避免 IO 过重）
    返回：{chunk_id: text}
    """
    if not chunk_ids:
        return {}
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(chunk_ids))
    sql = f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})"
    rows = cur.execute(sql, chunk_ids).fetchall()
    conn.close()
    mp = {str(r["chunk_id"]): str(r["text"]) for r in rows}
    return mp

def _fetch_evidence_rows(chunk_ids: List[str]) -> List[Dict[str, Any]]:
    """
    给前端/本地提供证据（你现在 app.py 的 evidence 返回格式沿用）
    """
    if not chunk_ids:
        return []
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(chunk_ids))
    sql = f"""
    SELECT chunk_id, text, paper_id, domain, year, source_xlsx
    FROM chunks
    WHERE chunk_id IN ({placeholders})
    """
    rows = cur.execute(sql, chunk_ids).fetchall()
    conn.close()
    row_map = {str(r["chunk_id"]): dict(r) for r in rows}
    return [row_map[cid] for cid in chunk_ids if cid in row_map]

def _load_chunk_domain_map() -> Dict[str, Dict[str, str]]:
    """
    给 ClassificationBooster 用的 chunk_meta_map：
      {chunk_id: {"domain_id": "..."}}
    说明：
      - 你的 sqlite 目前有 domain 字段（中文）
      - booster 默认 mode=domain，需要 domain_id
      - 这里做一个“domain中文->domain_id”的映射
    """
    domain_zh_to_id = {
        "深海感知与通信装备": "SENSOR_COMM",
        "深海可再生能源": "RENEWABLE",
        "深海矿产": "MINERALS",
        "深水油气": "OIL_GAS",
    }

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT chunk_id, domain FROM chunks").fetchall()
    conn.close()

    mp: Dict[str, Dict[str, str]] = {}
    for r in rows:
        cid = str(r["chunk_id"])
        dz = str(r["domain"] or "").strip()
        mp[cid] = {"domain_id": domain_zh_to_id.get(dz, "UNKNOWN")}
    return mp

chunk_meta_map = _load_chunk_domain_map()

pipeline = RetrievalPipeline(
    embedder=embedder,
    cfg=cfg,
    sparse=sparse,
    dense=dense,
    fusion=fusion,
    booster=booster,
    reranker=reranker,
    chunk_meta_map=chunk_meta_map,
    fetch_texts_fn=_fetch_text_map,
)

# -----------------------------
# API models
# -----------------------------
class SearchReq(BaseModel):
    query: str
    top_k: int = 10
    return_evidence: bool = True

class RetrieveReq(BaseModel):
    query_bundle: Dict[str, Any]
    final_top_n: Optional[int] = None
    delta: Optional[float] = None  # 动态截断阈值：保留得分在top1-delta区间内的结果（默认使用配置值）
    return_evidence: bool = True
    return_debug: bool = False

@app.get("/health")
def health():
    return {
        "status": "ok",
        "kb_dir": KB_DIR,
        "bm25": os.path.isfile(BM25_PKL_PATH),
        "faiss": os.path.isfile(FAISS_INDEX_PATH),
        "sqlite": os.path.isfile(SQLITE_PATH),
        "rerank_model": cfg.rerank.model_name if cfg.rerank.enable else None,
    }

# -----------------------------
# Dense-only endpoint
# -----------------------------
@app.post("/search")
def search(req: SearchReq):
    t0 = time.time()
    resp = _resp_template(stage="search")

    try:
        q = (req.query or "").strip()
        top_k = int(req.top_k)

        resp["query"]["original"] = q
        resp["query"]["dense"] = [q]
        resp["query"]["sparse"] = [q]

        res = dense.retrieve([q], top_k=top_k, batch_size=cfg.dense.batch_size)
        chunk_ids = [x.chunk_id for x in res]
        scores = [x.score for x in res]

        resp["retrieval"]["chunk_ids"] = chunk_ids
        resp["retrieval"]["scores"] = scores
        resp["retrieval"]["score_type"] = "dense_only"
        resp["retrieval"]["scores_raw"] = []  # dense-only 不提供原始分数
        resp["retrieval"]["candidates"] = []  # dense-only 不提供候选分解

        if req.return_evidence:
            resp["evidence"] = _fetch_evidence_rows(chunk_ids)

        resp["meta"]["fusion"] = {"strategy": "dense_only"}
        resp["meta"]["boost"] = {"enable": False}
        resp["meta"]["rerank"] = {"enable": False}
        resp["meta"]["timing"]["seconds"] = round(time.time() - t0, 4)
        return resp

    except Exception as e:
        resp["ok"] = False
        resp["error"] = f"{type(e).__name__}: {e}"
        resp["meta"]["timing"]["seconds"] = round(time.time() - t0, 4)
        return resp

# -----------------------------
# Hybrid retrieval endpoint
# -----------------------------
@app.post("/retrieve")
def retrieve(req: RetrieveReq):
    t0 = time.time()
    resp = _resp_template(stage="retrieve")

    try:
        bundle = req.query_bundle

        # 填 query 信息（固定字段）
        resp["query"]["original"] = (bundle.get("variants", {}) or {}).get("original", "") or ""
        resp["query"]["dense"] = bundle.get("queries_dense", []) or []
        resp["query"]["sparse"] = bundle.get("queries_sparse", []) or []

        # 跑 pipeline（你现有 pipeline.retrieve 输出：chunk_ids/scores/(debug可选)）
        out = pipeline.retrieve(bundle, return_debug=bool(req.return_debug))

        # 避免分数为负
        def _sigmoid(x: float) -> float:
            # 数值稳定版 sigmoid
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                return z / (1.0 + z)

        chunk_ids = out.get("chunk_ids", []) or []
        raw_scores = out.get("scores", []) or []   # 这里目前是 rerank logits 或 boosted_fused
        debug_rows = out.get("debug", []) or []

        # 判断当前 scores 的来源：只要 rerank 开启，就认为 out.scores 是 rerank logits
        rerank_on = bool(cfg.rerank.enable)

        if rerank_on:
            # 展示分数用 sigmoid(logit)，避免负数且更易解释
            disp_scores = [_sigmoid(float(s)) for s in raw_scores]
            score_type = "rerank_sigmoid"
        else:
            disp_scores = [float(s) for s in raw_scores]
            score_type = "boosted_fused"  # 或 fused / dense_only，按你的 pipeline 实际

        # 应用动态截断和最终top_n过滤
        filtered_chunk_ids = chunk_ids
        filtered_disp_scores = disp_scores
        filtered_raw_scores = raw_scores
        filtered_debug_rows = debug_rows

        # 1. 应用delta动态截断
        delta = req.delta if req.delta is not None else cfg.rerank.delta
        if delta is not None and len(disp_scores) > 0:
            top1_score = max(disp_scores)
            threshold = top1_score - delta
            
            # 过滤出得分在阈值以上的结果
            filtered_items = [(cid, ds, rs, dr) for cid, ds, rs, dr in zip(chunk_ids, disp_scores, raw_scores, debug_rows) if ds >= threshold]
            if filtered_items:
                filtered_chunk_ids, filtered_disp_scores, filtered_raw_scores, filtered_debug_rows = zip(*filtered_items)
                filtered_chunk_ids = list(filtered_chunk_ids)
                filtered_disp_scores = list(filtered_disp_scores)
                filtered_raw_scores = list(filtered_raw_scores)
                filtered_debug_rows = list(filtered_debug_rows)
            else:
                # 如果所有结果都被过滤，至少保留top1
                top1_idx = disp_scores.index(top1_score)
                filtered_chunk_ids = [chunk_ids[top1_idx]]
                filtered_disp_scores = [disp_scores[top1_idx]]
                filtered_raw_scores = [raw_scores[top1_idx]]
                filtered_debug_rows = [debug_rows[top1_idx]] if debug_rows else []

        # 2. 应用final_top_n（最多10条）
        if req.final_top_n is not None and req.final_top_n > 0:
            filtered_chunk_ids = filtered_chunk_ids[:req.final_top_n]
            filtered_disp_scores = filtered_disp_scores[:req.final_top_n]
            filtered_raw_scores = filtered_raw_scores[:req.final_top_n]
            filtered_debug_rows = filtered_debug_rows[:req.final_top_n]
        
        # 3. 确保最少返回min_final_top_n条结果
        min_final_top_n = cfg.rerank.min_final_top_n
        if len(filtered_chunk_ids) < min_final_top_n and len(chunk_ids) >= min_final_top_n:
            # 如果过滤后不足min_final_top_n条，从原始结果中取前min_final_top_n条
            filtered_chunk_ids = chunk_ids[:min_final_top_n]
            filtered_disp_scores = disp_scores[:min_final_top_n]
            filtered_raw_scores = raw_scores[:min_final_top_n]
            filtered_debug_rows = debug_rows[:min_final_top_n] if debug_rows else []

        resp["retrieval"]["chunk_ids"] = filtered_chunk_ids
        resp["retrieval"]["scores"] = filtered_disp_scores
        resp["retrieval"]["score_type"] = score_type

        # 可选：debug 时再把 raw logits 留下来（便于误差分析/复现）
        if req.return_debug:
            resp["retrieval"]["scores_raw"] = [float(s) for s in filtered_raw_scores]
        else:
            resp["retrieval"]["scores_raw"] = []
        
        resp["retrieval"]["candidates"] = filtered_debug_rows if req.return_debug else []

        # evidence（固定字段）
        if req.return_evidence:
            resp["evidence"] = _fetch_evidence_rows(filtered_chunk_ids)
        else:
            resp["evidence"] = []

        # meta（固定字段）
        resp["meta"]["fusion"] = {
            "strategy": cfg.fusion.strategy,
            "alpha": cfg.fusion.alpha,
            "normalize": cfg.fusion.normalize,
        }
        resp["meta"]["boost"] = {
            "enable": cfg.boost.enable,
            "mode": cfg.boost.mode,
            "use_top_k": cfg.boost.use_top_k,
            "boost_weight": cfg.boost.boost_weight,
        }
        resp["meta"]["rerank"] = {
            "enable": cfg.rerank.enable,
            "model": cfg.rerank.model_name if cfg.rerank.enable else None,
            "top_m": cfg.rerank.top_m,
            "batch_size": cfg.rerank.batch_size,
            "final_top_n": cfg.rerank.final_top_n,
            "min_final_top_n": cfg.rerank.min_final_top_n,
        }
        
        # 添加动态截断信息
        resp["meta"]["truncation"] = {
            "delta": delta,
            "applied": delta is not None,
            "final_count": len(filtered_chunk_ids),
        }

        resp["meta"]["timing"]["seconds"] = round(time.time() - t0, 4)
        return resp

    except Exception as e:
        resp["ok"] = False
        resp["error"] = f"{type(e).__name__}: {e}"
        resp["meta"]["timing"]["seconds"] = round(time.time() - t0, 4)
        # 失败时也保证字段存在
        return resp


# -----------------------------  
# 新增：计算嵌入和相似度接口
# -----------------------------  
from pydantic import BaseModel


class ComputeEmbeddingReq(BaseModel):
    texts: List[str]


class ComputeSimilarityReq(BaseModel):
    text1: str
    text2: str


class ComputeBERTScoreReq(BaseModel):
    predictions: List[str]
    references: List[str]


@app.post("/compute_embedding")
def compute_embedding(req: ComputeEmbeddingReq):
    """计算文本嵌入"""
    try:
        texts = req.texts or []
        if not texts:
            return {
                "ok": False,
                "error": "Texts are required",
                "embeddings": [],
                "count": 0
            }
        
        # 计算嵌入
        embeddings = embedder.encode(texts).tolist()
        
        return {
            "ok": True,
            "embeddings": embeddings,
            "count": len(embeddings)
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "embeddings": [],
            "count": 0
        }


@app.post("/compute_similarity")
def compute_similarity(req: ComputeSimilarityReq):
    """计算文本相似度"""
    try:
        text1 = (req.text1 or "").strip()
        text2 = (req.text2 or "").strip()
        
        if not text1 or not text2:
            return {
                "ok": False,
                "error": "text1 and text2 are required",
                "similarity": 0.0
            }
        
        # 计算嵌入
        emb1 = embedder.encode(text1)
        emb2 = embedder.encode(text2)
        
        # 计算余弦相似度
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        return {
            "ok": True,
            "similarity": similarity,
            "text1": text1,
            "text2": text2
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "similarity": 0.0
        }


class ComputeBERTScoreReq(BaseModel):
    predictions: List[str]
    references: List[str]

@app.post("/compute_bertscore")
def compute_bertscore(req: ComputeBERTScoreReq):
    """计算BERTScore"""
    try:
        predictions = req.predictions
        references = req.references

        # 1. 基础校验
        if not predictions or not references:
            return {
                "ok": False,
                "error": "predictions and references lists cannot be empty",
                "scores": []
            }

        if len(predictions) != len(references):
            return {
                "ok": False,
                "error": f"Length mismatch: predictions ({len(predictions)}) vs references ({len(references)}). They must be equal.",
                "scores": []
            }

        # 2. 使用 bert-base-chinese 计算
        try:
            # 本地加载模型
            scorer = BERTScorer(
                model_type=BERTSCORE_MODEL_DIR,
                num_layers=9,
                lang="zh",
                rescale_with_baseline=False,
                device="cuda"
            )

            P, R, F1 = scorer.score(predictions, references)
        
        except Exception as model_err:
            raise RuntimeError(f"BERTScore calculation failed: {str(model_err)}")

        # 3. 转 numpy
        device = "cpu"
        if isinstance(P, torch.Tensor):
            P = P.to(device).detach().numpy()
            R = R.to(device).detach().numpy()
            F1 = F1.to(device).detach().numpy()

        P = P.flatten()
        R = R.flatten()
        F1 = F1.flatten()

        scores = []
        for i in range(len(predictions)):
            scores.append({
                "prediction": predictions[i],
                "reference": references[i],
                "precision": float(P[i]),
                "recall": float(R[i]),
                "f1": float(F1[i])
            })

        avg_precision = float(np.mean(P))
        avg_recall = float(np.mean(R))
        avg_f1 = float(np.mean(F1))

        return {
            "ok": True,
            "scores": scores,
            "average": {
                "precision": round(avg_precision, 4),
                "recall": round(avg_recall, 4),
                "f1": round(avg_f1, 4)
            },
            "count": len(scores)
        }

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(error_detail)
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "scores": [],
            "debug_info": error_detail
        }

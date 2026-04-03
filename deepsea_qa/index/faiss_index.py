# -*- coding: utf-8 -*-
"""
faiss_index.py

作者：Accilia
创建时间：2026-02-22
用途说明：
1. 使用 BAAI/bge-m3 对知识库 chunk 进行向量化
2. 构建 FAISS IndexFlatIP（内积索引）
3. 使用 id_map 绑定 chunk_id
4. 持久化 index 文件
5. 提供加载与查询接口

设计原则：
- FAISS 只存向量 + chunk_id
- 原始文本与元数据仍存于 SQLite
- 保证严格溯源能力
- 方便后续 embedding 替换与对比实验


"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss 导入失败，请安装 faiss-cpu 或 faiss-gpu") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("未安装 sentence-transformers，请先安装") from e

from deepsea_qa.configs.index_config import FAISSConfig

# =====================================================
# 1) 构建配置
# =====================================================

# 使用 FAISSConfig 替代 FaissBuildConfig
FaissBuildConfig = FAISSConfig


# =====================================================
# 2) Embedding 封装：bge-m3
# =====================================================

class BgeM3Embedder:
    """对 SentenceTransformer 的薄封装，后续换 embedding 只改这里。"""

    def __init__(self, cfg: FaissBuildConfig):
        self.cfg = cfg
        os.environ["TRANSFORMERS_OFFLINE"] = "1" # 离线模式，不下载模型
        self.model = SentenceTransformer(
            cfg.model_name,
            device=cfg.device,
            # trust_remote_code=True,
        )
        if hasattr(self.model, "max_seq_length") and cfg.max_length:
            self.model.max_seq_length = cfg.max_length

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb


# =====================================================
# 3) 从 SQLite 流式读取 chunks（与 A1/A2 一致）
# =====================================================

def iter_chunks_from_sqlite(
    sqlite_path: str,
    table_name: str = "chunks",
    chunk_id_col: str = "chunk_id",   # ✅ A1/A2: chunk_id
    text_col: str = "text",           # ✅ A1/A2: text
) -> Iterable[Tuple[str, str]]:
    """
    逐条读取 (chunk_id, text)

    注意：chunk_id 是字符串；后续会分配一个 faiss_id(int64)
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = f"""
    SELECT {chunk_id_col} AS cid, {text_col} AS txt
    FROM {table_name}
    ORDER BY {chunk_id_col} ASC
    """

    try:
        for row in cur.execute(sql):
            cid = str(row["cid"])
            txt = str(row["txt"]) if row["txt"] is not None else ""
            yield cid, txt
    finally:
        conn.close()


# =====================================================
# 4) 建库：FAISS + id_map 持久化
# =====================================================

def build_faiss_index(
    sqlite_path: str,
    out_faiss_path: str,
    out_meta_path: str,
    out_idmap_path: str,
    cfg: Optional[FaissBuildConfig] = None,
    table_name: str = "chunks",
    chunk_id_col: str = "chunk_id",
    text_col: str = "text",
) -> None:
    """
    从 SQLite 构建 FAISS 索引（一次建好，后续只加载查询）

    输出文件：
    - out_faiss_path : FAISS 索引本体
    - out_idmap_path : JSONL，每行 {"faiss_id": 1, "chunk_id": "..."}
    - out_meta_path  : JSON，记录 embedding/维度/归一化/数量/耗时 等
    """
    cfg = cfg or FaissBuildConfig()
    os.makedirs(os.path.dirname(out_faiss_path), exist_ok=True)

    embedder = BgeM3Embedder(cfg)

    # 动态探测维度，避免硬编码
    dim = int(embedder.encode(["dim probe"]).shape[1])

    # ✅ 归一化后，用内积索引即可近似 cosine
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)

    # 写 id_map（JSONL）—保证可流式写，避免巨大 dict 占内存
    os.makedirs(os.path.dirname(out_idmap_path), exist_ok=True)
    f_idmap = open(out_idmap_path, "w", encoding="utf-8")

    batch_texts: List[str] = []
    batch_faiss_ids: List[int] = []

    total = 0
    faiss_id = 0  # 从 1 开始更直观（0 也行）
    t0 = time.time()

    try:
        for chunk_id, txt in iter_chunks_from_sqlite(
            sqlite_path=sqlite_path,
            table_name=table_name,
            chunk_id_col=chunk_id_col,
            text_col=text_col,
        ):
            if not txt.strip():
                continue

            faiss_id += 1
            # 记录映射：faiss_id -> chunk_id
            f_idmap.write(json.dumps({"faiss_id": faiss_id, "chunk_id": chunk_id}, ensure_ascii=False) + "\n")

            batch_texts.append(txt)
            batch_faiss_ids.append(faiss_id)

            if len(batch_texts) >= cfg.batch_size:
                vecs = embedder.encode(batch_texts)
                ids_np = np.array(batch_faiss_ids, dtype=np.int64)
                index.add_with_ids(vecs, ids_np)

                total += len(batch_texts)
                batch_texts.clear()
                batch_faiss_ids.clear()

        # last batch
        if batch_texts:
            vecs = embedder.encode(batch_texts)
            ids_np = np.array(batch_faiss_ids, dtype=np.int64)
            index.add_with_ids(vecs, ids_np)
            total += len(batch_texts)

    finally:
        f_idmap.close()

    # 持久化 FAISS
    faiss.write_index(index, out_faiss_path)

    meta = {
        "embedding_model": cfg.model_name,
        "device": cfg.device,
        "dim": dim,
        "normalize": cfg.normalize,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "sqlite_source": {
            "sqlite_path": sqlite_path,
            "table_name": table_name,
            "chunk_id_col": chunk_id_col,
            "text_col": text_col,
        },
        "id_map_path": out_idmap_path,
        "total_indexed": total,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 3),
    }

    os.makedirs(os.path.dirname(out_meta_path), exist_ok=True)
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# =====================================================
# 5) 加载 + 查询（返回 chunk_id，保证溯源链路）
# =====================================================

def load_faiss_index(faiss_path: str):
    return faiss.read_index(faiss_path)


def load_id_map(idmap_jsonl_path: str) -> dict:
    """
    加载 faiss_id -> chunk_id 的映射。
    """
    mp = {}
    with open(idmap_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mp[int(obj["faiss_id"])] = obj["chunk_id"]
    return mp


def search(
    index,
    embedder: BgeM3Embedder,
    id_map: dict,
    query: str,
    top_k: int = 10,
) -> Tuple[List[str], List[float]]:
    """
    输入中文 query，输出：
    - chunk_ids（字符串，能直接回 SQLite 查全文与元数据）
    - scores（cosine 相似度分数）
    """
    qv = embedder.encode([query])  # (1, dim)
    scores, ids = index.search(qv, top_k)

    faiss_ids = [int(x) for x in ids[0].tolist() if int(x) != -1]
    chunk_ids = [id_map.get(fid) for fid in faiss_ids]
    chunk_ids = [c for c in chunk_ids if c is not None]
    score_list = [float(x) for x in scores[0].tolist()[: len(chunk_ids)]]

    return chunk_ids, score_list
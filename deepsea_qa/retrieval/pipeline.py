# -*- coding: utf-8 -*-
"""
retrieval/pipeline.py

作者：Accilia
创建时间：2026-02-24
用途说明：服务器端 RetrievalPipeline：
- 稀疏（BM25）
- 稠密（FAISS）
- 融合（RRF/Linear）
- 分类 soft boost
- rerank（batch 推理）

输出：
- 最终 chunk_id 列表（以及可选 debug 字段）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from deepsea_qa.configs.retrieval_config import RetrievalConfig
from deepsea_qa.retrieval.types import FusedChunk
from deepsea_qa.retrieval.sparse import SparseRetriever
from deepsea_qa.retrieval.dense import DenseRetriever
from deepsea_qa.retrieval.fusion import FusionModule
from deepsea_qa.retrieval.boost import ClassificationBooster
from deepsea_qa.retrieval.rerank import Reranker


class RetrievalPipeline:
    def __init__(
        self,
        embedder: SentenceTransformer,
        cfg: RetrievalConfig,
        sparse: SparseRetriever,
        dense: DenseRetriever,
        fusion: FusionModule,
        booster: ClassificationBooster,
        reranker: Optional[Reranker],
        # chunk meta：用于 boost（domain_id / label_id）
        chunk_meta_map: Dict[str, Dict[str, str]],
        # sqlite fetch 函数：{chunk_id:text}
        fetch_texts_fn,
    ):
        self.embedder = embedder
        self.cfg = cfg

        self.sparse = sparse
        self.dense = dense
        self.fusion = fusion
        self.booster = booster
        self.reranker = reranker

        self.chunk_meta_map = chunk_meta_map
        self.fetch_texts_fn = fetch_texts_fn

    def retrieve(self, query_bundle: dict, return_debug: bool = False) -> Dict[str, Any]:
        """
        return_debug:
          True 时返回中间候选与分数（便于你写论文、做消融与误差分析）
        """
        # --------- 读取输入 ---------
        q_sparse = query_bundle.get("queries_sparse", []) or []
        q_dense = query_bundle.get("queries_dense", []) or []
        q_original = (query_bundle.get("variants", {}) or {}).get("original", "") or ""

        # --------- 1) BM25 ---------
        sparse_res = self.sparse.retrieve(
            queries=q_sparse,
            top_k=self.cfg.sparse.top_k,
            merge=self.cfg.sparse.merge,
        )

        # --------- 2) Dense ---------
        dense_res = self.dense.retrieve(
            queries=q_dense,
            top_k=self.cfg.dense.top_k,
            batch_size=self.cfg.dense.batch_size,
        )

        # --------- 3) Fusion ---------
        fused: List[FusedChunk] = self.fusion.fuse(sparse_res, dense_res)

        # --------- 4) 分类 soft boost（默认在 fusion 后做） ---------
        fused = self.booster.apply(
            fused=fused,
            query_bundle=query_bundle,
            chunk_meta_map=self.chunk_meta_map,
        )

        # --------- 5) Rerank（batch） ---------
        final_list: List[FusedChunk]
        if self.reranker and self.cfg.rerank.enable:
            # 回 sqlite 拿 TopM 文本
            top_m_ids = [x.chunk_id for x in fused[: self.cfg.rerank.top_m]]
            passages = self.fetch_texts_fn(top_m_ids)  # {chunk_id:text}

            final_list = self.reranker.rerank(
                query=q_original,
                candidates=fused,
                passages=passages,
                top_m=self.cfg.rerank.top_m,
                final_top_n=self.cfg.rerank.final_top_n,
                batch_size=self.cfg.rerank.batch_size,
            )
        else:
            final_list = fused[: self.cfg.rerank.final_top_n]

        # --------- 输出 ---------
        out = {
            "chunk_ids": [x.chunk_id for x in final_list],
            "scores": [x.score_rerank if (self.reranker and self.cfg.rerank.enable) else x.score_boosted for x in final_list],
        }

        if return_debug and self.cfg.eval.record_debug:
            out["debug"] = [
                {
                    "chunk_id": x.chunk_id,
                    "bm25": x.score_bm25,
                    "dense": x.score_dense,
                    "fused": x.score_fused,
                    "boosted": x.score_boosted,
                    "rerank": x.score_rerank,
                    "why": x.debug,
                }
                for x in final_list
            ]

        return out
# -*- coding: utf-8 -*-
"""
sparse.py

作者：Accilia
创建时间：2026-02-24
用途说明：稀疏检索（BM25）。
- 输入：queries_sparse（多个 query，来自 rewrite + expand）
- 输出：ScoredChunk 列表

注意：
- BM25 索引对象由 bm25_index.py 产出，通常提供 score_query(query)->[(chunk_id, score), ...]
- bm25.pkl 保存的是 BM25Artifacts（数据），不是 BM25Index（方法）
- BM25Index.score_query 需要 query_tokens（list[str]），不能直接传 query string
"""

from __future__ import annotations

from typing import Dict, List

from deepsea_qa.configs import paths
from deepsea_qa.retrieval.types import ScoredChunk
from deepsea_qa.retrieval.utils import build_stopwords, mixed_tokenize
from deepsea_qa.index.bm25_index import BM25Index


class SparseRetriever:
    def __init__(self, bm25_pkl_path: str | None = None):
        """
        bm25_pkl_path:
          默认使用 paths.BM25_PKL_PATH（与你的 paths.py 一致）
        """
        p = bm25_pkl_path or str(paths.BM25_PKL_PATH)

        # 用 BM25Index.load 把 artifacts 包装成 index
        self.index = BM25Index.load(p)

        # 查询分词同建库一致：混合分词 + 停用词
        self.stopwords = build_stopwords(
            cn_stopwords_dir=str(paths.STOPWORDS_DIR),
            use_sklearn_en=True,
            use_nltk_en=False,            # 服务器离线建议先关掉
            nltk_download_if_missing=False
        )

    def retrieve(self, queries: List[str], top_k: int, merge: str = "sum") -> List[ScoredChunk]:
        """
        queries: 来自 query_bundle["queries_sparse"]
        merge: "sum"（默认） or "rrf"
        """
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return []

        if merge == "rrf":
            return self._retrieve_rrf(queries, top_k=top_k)

        # 默认 sum 合并：同一 chunk 的 bm25 分数累加
        acc: Dict[str, float] = {}
        for q in queries:
            q_tokens = mixed_tokenize(q, stopwords=self.stopwords)
            pairs = self.index.score_query(q_tokens, topk=top_k)  # [(chunk_id, score)]

            for cid, s in pairs:
                acc[cid] = acc.get(cid, 0.0) + float(s)

        out = [ScoredChunk(chunk_id=cid, score=sc, source="bm25") for cid, sc in acc.items()]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]

    def _retrieve_rrf(self, queries: List[str], top_k: int, k: int = 60) -> List[ScoredChunk]:
        """
        RRF 合并多个 query 的 bm25 排名（稳但稍慢）
        """
        acc: Dict[str, float] = {}
        for q in queries:
            q_tokens = mixed_tokenize(q, stopwords=self.stopwords)
            pairs = self.index.score_query(q_tokens, topk=top_k)

            for rank, (cid, _s) in enumerate(pairs):
                acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank)

        out = [ScoredChunk(chunk_id=cid, score=sc, source="bm25") for cid, sc in acc.items()]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]
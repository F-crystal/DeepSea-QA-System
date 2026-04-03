# -*- coding: utf-8 -*-
"""
dense.py

作者：Accilia
创建时间：2026-02-24
用途说明：稠密检索（FAISS）。
- 输入：queries_dense（一般只有 original）
- 输出：ScoredChunk 列表
"""

from __future__ import annotations

import json
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from deepsea_qa.configs import paths
from deepsea_qa.retrieval.types import ScoredChunk


class DenseRetriever:
    def __init__(
        self,
        embedder: SentenceTransformer,
        faiss_index_path: str | None = None,
        faiss_idmap_path: str | None = None,
    ):
        self.embedder = embedder

        idx_path = faiss_index_path or str(paths.FAISS_INDEX_PATH)
        idmap_path = faiss_idmap_path or str(paths.FAISS_IDMAP_PATH)

        self.index = faiss.read_index(idx_path)

        # faiss_id -> chunk_id
        self.id_map: Dict[int, str] = {}
        with open(idmap_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.id_map[int(obj["faiss_id"])] = str(obj["chunk_id"])

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        注意：索引是 IP（内积），但 encode 时 normalize_embeddings=True
        ⇒ IP 等价 cosine，相似度可直接用作 dense score。
        """
        vec = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

    def retrieve(self, queries: List[str], top_k: int, batch_size: int = 32) -> List[ScoredChunk]:
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return []

        # 多 query：这里采用“分数累加”合并（和 sparse 的 sum 类似）
        acc: Dict[str, float] = {}
        vec = self._encode(queries, batch_size=batch_size)

        D, I = self.index.search(vec, top_k)  # D: cosine score, I: faiss_id
        for qi in range(len(queries)):
            for score, fid in zip(D[qi].tolist(), I[qi].tolist()):
                if int(fid) == -1:
                    continue
                cid = self.id_map.get(int(fid))
                if not cid:
                    continue
                acc[cid] = acc.get(cid, 0.0) + float(score)

        out = [ScoredChunk(chunk_id=cid, score=sc, source="dense") for cid, sc in acc.items()]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]
# -*- coding: utf-8 -*-
"""
index/bm25_index.py

作者：Accilia
创建时间：2026-02-23
用途说明： BM25 建库（稀疏检索索引）

持久化：
- bm25.pkl：BM25所需统计量 + doc_ids映射
- tokenizer_meta.json：记录分词、停用词来源、参数等（便于论文复现）
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional

from deepsea_qa.configs.index_config import BM25Config


@dataclass
class BM25Artifacts:
    # doc_ids: 文档序号 -> chunk_id（用于溯源）
    doc_ids: List[str]
    # doc_len: 每篇文档 token 数
    doc_len: List[int]
    # avgdl: 平均文档长度
    avgdl: float
    # df: term -> 文档频次
    df: Dict[str, int]
    # idf: term -> idf
    idf: Dict[str, float]
    # postings: term -> [(doc_index, tf), ...] 倒排表（节省空间）
    postings: Dict[str, List[Tuple[int, int]]]
    # BM25 参数
    k1: float
    b: float


class BM25Index:
    def __init__(self, artifacts: BM25Artifacts):
        self.art = artifacts

    @staticmethod
    def _calc_idf(N: int, df_t: int) -> float:
        # 常见BM25平滑IDF：ln((N - df + 0.5)/(df + 0.5) + 1)
        return math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

    @classmethod
    def build(
        cls,
        tokenized_docs: Iterable[Tuple[str, List[str]]],
        cfg: Optional[BM25Config] = None,
    ) -> "BM25Index":
        cfg = cfg or BM25Config()
        k1 = cfg.k1
        b = cfg.b
        min_doc_length = cfg.min_doc_length
        doc_ids: List[str] = []
        doc_len: List[int] = []
        df: Dict[str, int] = {}
        postings: Dict[str, List[Tuple[int, int]]] = {}

        for doc_index, (chunk_id, tokens) in enumerate(tokenized_docs):
            L = len(tokens)
            # 过滤短文档
            if L < min_doc_length:
                continue
                
            doc_ids.append(chunk_id)
            doc_len.append(L)

            # 统计文档内 tf
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            # 更新 df 与 postings
            for t, c in tf.items():
                df[t] = df.get(t, 0) + 1
                postings.setdefault(t, []).append((doc_index, c))

        N = len(doc_ids)
        avgdl = (sum(doc_len) / N) if N > 0 else 0.0
        idf = {t: cls._calc_idf(N, dft) for t, dft in df.items()}

        return cls(BM25Artifacts(
            doc_ids=doc_ids,
            doc_len=doc_len,
            avgdl=avgdl,
            df=df,
            idf=idf,
            postings=postings,
            k1=k1,
            b=b,
        ))

    def save(self, pkl_path: str) -> None:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(self.art, f)

    @classmethod
    def load(cls, pkl_path: str) -> "BM25Index":
        with open(pkl_path, "rb") as f:
            art: BM25Artifacts = pickle.load(f)
        return cls(art)

    def score_query(self, query_tokens: List[str], topk: int = 50) -> List[Tuple[str, float]]:
        """
        BM25 查询：输入 token 列表，输出 [(chunk_id, score)]
        """
        if not query_tokens:
            return []
        N = len(self.art.doc_ids)
        if N == 0:
            return []

        scores = [0.0] * N
        q_tf: Dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        k1 = self.art.k1
        b = self.art.b
        avgdl = self.art.avgdl if self.art.avgdl > 0 else 1.0

        for term, qcount in q_tf.items():
            plist = self.art.postings.get(term)
            if not plist:
                continue
            idf = self.art.idf.get(term, 0.0)

            for doc_index, tf in plist:
                dl = self.art.doc_len[doc_index]
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                score = idf * (tf * (k1 + 1.0)) / (denom + 1e-12)
                scores[doc_index] += score * qcount

        pairs = [(self.art.doc_ids[i], s) for i, s in enumerate(scores) if s > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:topk]


def iter_chunks_from_sqlite(
    sqlite_path: str,
    table_name: str = "chunks",
    chunk_id_col: str = "chunk_id",
    text_col: str = "text",
) -> Iterable[Tuple[str, str]]:
    """
    与 A1/A2/A3 保持一致：从 SQLite 的 chunks 表读取 chunk_id + text
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


def save_tokenizer_meta(meta: Dict[str, Any], out_json_path: str) -> None:
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
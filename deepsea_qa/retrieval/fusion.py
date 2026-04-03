# -*- coding: utf-8 -*-
"""
fusion.py

作者：Accilia
创建时间：2026-02-24
用途说明：融合模块（BM25 + Dense）。
- 支持 RRF：稳、默认推荐
- 支持 Linear：论文友好（含归一化以解决分数尺度不可比）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from deepsea_qa.retrieval.types import FusedChunk, ScoredChunk


def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-12:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _zscore(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    std = var ** 0.5
    if std < 1e-12:
        return [0.0 for _ in scores]
    return [(s - mean) / std for s in scores]


class FusionModule:
    def __init__(self, strategy: str = "rrf", alpha: float = 0.5, normalize: str = "minmax"):
        self.strategy = strategy
        self.alpha = alpha
        self.normalize = normalize

    def fuse(self, sparse: List[ScoredChunk], dense: List[ScoredChunk]) -> List[FusedChunk]:
        if self.strategy == "linear":
            return self._linear(sparse, dense)

        # 默认 RRF
        return self._rrf(sparse, dense)

    def _rrf(self, sparse: List[ScoredChunk], dense: List[ScoredChunk], k: int = 60) -> List[FusedChunk]:
        """
        只依赖 rank 的融合，适合 score 尺度不可比的场景。
        """
        acc: Dict[str, FusedChunk] = {}

        # 稀疏 list 按 score 已经降序；rank 越小越好
        for rank, it in enumerate(sparse):
            fc = acc.get(it.chunk_id) or FusedChunk(chunk_id=it.chunk_id, score_fused=0.0)
            fc.score_bm25 = max(fc.score_bm25, it.score)  # 保留原始路分数（解释用）
            fc.score_fused += 1.0 / (k + rank)
            acc[it.chunk_id] = fc

        for rank, it in enumerate(dense):
            fc = acc.get(it.chunk_id) or FusedChunk(chunk_id=it.chunk_id, score_fused=0.0)
            fc.score_dense = max(fc.score_dense, it.score)
            fc.score_fused += 1.0 / (k + rank)
            acc[it.chunk_id] = fc

        out = list(acc.values())
        out.sort(key=lambda x: x.score_fused, reverse=True)
        return out

    def _linear(self, sparse: List[ScoredChunk], dense: List[ScoredChunk]) -> List[FusedChunk]:
        """
        Score = alpha * Score_BM25 + (1-alpha) * Score_cosine
        但必须做归一化，否则 BM25 与 cosine 尺度不一致会导致某一方碾压。
        """
        s_dict = {x.chunk_id: x.score for x in sparse}
        d_dict = {x.chunk_id: x.score for x in dense}
        all_ids = sorted(set(s_dict) | set(d_dict))

        s_list = [s_dict.get(cid, 0.0) for cid in all_ids]
        d_list = [d_dict.get(cid, 0.0) for cid in all_ids]

        if self.normalize == "zscore":
            s_norm = _zscore(s_list)
            d_norm = _zscore(d_list)
        elif self.normalize == "none":
            s_norm, d_norm = s_list, d_list
        else:
            s_norm = _minmax(s_list)
            d_norm = _minmax(d_list)

        out: List[FusedChunk] = []
        for cid, ss, dd, ss0, dd0 in zip(all_ids, s_norm, d_norm, s_list, d_list):
            fused = self.alpha * ss + (1.0 - self.alpha) * dd
            out.append(FusedChunk(
                chunk_id=cid,
                score_fused=float(fused),
                score_bm25=float(ss0),
                score_dense=float(dd0),
            ))

        out.sort(key=lambda x: x.score_fused, reverse=True)
        return out
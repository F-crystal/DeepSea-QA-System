# -*- coding: utf-8 -*-
"""
types.py

作者：Accilia
创建时间：2026-02-23
用途说明：统一检索阶段的数据结构，便于模块解耦与调试/消融。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


SourceType = Literal["bm25", "dense", "fusion", "rerank"]


@dataclass
class ScoredChunk:
    """单路检索输出（BM25 或 Dense）。"""
    chunk_id: str
    score: float
    source: SourceType


@dataclass
class FusedChunk:
    """融合后的候选：保留各路分数，便于解释与论文写作。"""
    chunk_id: str
    score_fused: float

    score_bm25: float = 0.0
    score_dense: float = 0.0

    # boost 后的分数（如果启用）
    score_boosted: float = 0.0

    # rerank 分数（如果启用）
    score_rerank: float = 0.0

    # 用于调试：这个 chunk 为什么被 boost
    debug: Dict[str, str] = field(default_factory=dict)
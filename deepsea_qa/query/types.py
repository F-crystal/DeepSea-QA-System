# -*- coding: utf-8 -*-
"""
types.py

作者: 冯冉
创建时间: 2026-02-23
用途说明: Query 分类 + rewrite/expand 的标准化数据结构，便于消融实验与下游检索接入
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any


DomainId = Literal[
    "SENSOR_COMM",   # 深海感知与通信装备
    "RENEWABLE",     # 深海可再生能源
    "MINERALS",      # 深海矿产
    "OIL_GAS",       # 深水油气
    "UNKNOWN",
]


@dataclass
class LabelCandidate:
    domain_id: DomainId
    domain_name_zh: str
    label_id: str
    score: float
    evidence_spans: List[str] = field(default_factory=list)
    rationale: str = ""

@dataclass
class QueryClassification:
    domain_id: DomainId
    domain_name_zh: str
    primary_label_id: str
    secondary_label_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_spans: List[str] = field(default_factory=list)
    method: Literal["llm", "rules", "hybrid"] = "rules"
    raw: Optional[Dict[str, Any]] = None  # 可选：保存原始LLM输出便于debug

    # ★新增：多候选（跨方向 topK）
    candidates: List[LabelCandidate] = field(default_factory=list)


@dataclass
class QueryVariants:
    original: str
    rewrites: List[str] = field(default_factory=list)
    expands: List[str] = field(default_factory=list)

    def unique_sparse_queries(self, max_n: int = 8) -> List[str]:
        """BM25 用：original + rewrites + expands 去重截断"""
        seen = set()
        out: List[str] = []
        for q in [self.original, *self.rewrites, *self.expands]:
            q2 = (q or "").strip()
            if not q2 or q2 in seen:
                continue
            seen.add(q2)
            out.append(q2)
            if len(out) >= max_n:
                break
        return out

    def dense_queries(self) -> List[str]:
        """FAISS 用：默认只用 original"""
        return [self.original.strip()] if self.original.strip() else []


@dataclass
class QueryBundle:
    variants: QueryVariants
    classification: QueryClassification
    queries_sparse: List[str]
    queries_dense: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
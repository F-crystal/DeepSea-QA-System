# -*- coding: utf-8 -*-
"""
types.py

作者：Accilia
创建时间：2026-02-25
用途说明： 生成阶段的数据结构
- EvidenceItem：检索证据（来自 retrieval_result.evidence）
- PaperMeta：从原始 corpus.xlsx 补全的元数据
- GenerationResult：最终答案 + 引用 + 验证报告

注意：
- verifier 只用 chunk_id/quote 做一致性验证，不触碰“思维链”
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    chunk_id: str
    paper_id: str
    text: str
    domain: str = ""
    source_xlsx: str = ""


@dataclass
class PaperMeta:
    paper_id: str
    title: str = ""
    authors: str = ""
    venue: str = ""      # journal / conf
    year: Optional[int] = None
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    source_xlsx: str = ""


@dataclass
class CitationItem:
    paper_id: str
    chunk_id: str
    quote: str
    ref_gbt: str = ""
    ok: Optional[bool] = None
    reason: str = ""


@dataclass
class GenerationResult:
    query: str
    answer: str
    citations: List[CitationItem]
    refs_gbt: List[str]
    verified: bool
    verify_report: Dict[str, Any]
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
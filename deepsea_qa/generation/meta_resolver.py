# -*- coding: utf-8 -*-
"""
meta_resolver.py

作者：Accilia
创建时间：2026-02-25
用途说明：
从原始来源 corpus.xlsx 补全 paper 元数据。
输入：EvidenceItem（含 paper_id + source_xlsx）
输出：paper_id -> PaperMeta

工程约束：
- 离线可用（不依赖外部 API）
- 同一个 source_xlsx 做缓存，避免重复读表
"""

from __future__ import annotations

from typing import Dict, List, Optional
import os

import pandas as pd

from deepsea_qa.configs import paths
from deepsea_qa.generation.types import EvidenceItem, PaperMeta


# 兼容列名别名（按你 clean 表 *_std 设计，必要时可继续扩展）
COL_ALIASES = {
    "paper_id": ["paper_id", "ut", "UT", "UT=WOS", "ut_std"],
    "title": ["title_std", "title", "TI"],
    "authors": ["authors_std", "authors", "AU"],
    "venue": ["journal_std", "journal", "SO", "source"],
    "year": ["year_std", "year", "PY", "publication_year", "发表年", "年份"],
    "volume": ["volume_std", "VL", "volume"],
    "issue": ["issue_std", "IS", "issue"],
    "pages": ["pages_std_final", "pages", "BP", "EP"],
    "doi": ["doi_std", "doi", "DI"],
}


def _pick_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    norm = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        k = str(a).strip().lower()
        if k in norm:
            return norm[k]
    return None


def _safe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


class ExcelMetaResolver:
    def __init__(self, corpus_dir: str | None = None):
        # ✅ 关键：默认必须用 DIR_CORPUS
        self.corpus_dir = corpus_dir or str(paths.DIR_CORPUS)

        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._col_cache: Dict[str, Dict[str, Optional[str]]] = {}

    def _load(self, source_xlsx: str) -> pd.DataFrame:
        fp = os.path.join(self.corpus_dir, source_xlsx)
        if fp not in self._df_cache:
            df = pd.read_excel(fp)
            self._df_cache[fp] = df

            cols: Dict[str, Optional[str]] = {}
            for field, aliases in COL_ALIASES.items():
                cols[field] = _pick_col(df, aliases)
            self._col_cache[fp] = cols

        return self._df_cache[fp]

    def resolve(self, evidences: List[EvidenceItem]) -> Dict[str, PaperMeta]:
        """
        返回：paper_id -> PaperMeta
        """
        out: Dict[str, PaperMeta] = {}

        # 按 source_xlsx 分组，批量查
        groups: Dict[str, List[EvidenceItem]] = {}
        for e in evidences:
            sx = (e.source_xlsx or "").strip()
            if not sx:
                continue
            groups.setdefault(sx, []).append(e)

        for sx, items in groups.items():
            df = self._load(sx)
            fp = os.path.join(self.corpus_dir, sx)
            cols = self._col_cache[fp]

            col_pid = cols.get("paper_id")
            if not col_pid:
                # 没有 paper_id 列无法对齐
                continue

            # 为了稳健：统一转 str.strip 后比对
            pid_series = df[col_pid].astype(str).str.strip()

            for e in items:
                pid = (e.paper_id or "").strip()
                if not pid or pid in out:
                    continue

                idx = pid_series[pid_series == pid].index
                if len(idx) == 0:
                    out[pid] = PaperMeta(paper_id=pid, source_xlsx=sx)
                    continue

                row = df.loc[idx[0]]

                def _get(field: str) -> str:
                    c = cols.get(field)
                    if not c:
                        return ""
                    v = row.get(c)
                    if v is None:
                        return ""
                    try:
                        if pd.isna(v):
                            return ""
                    except Exception:
                        pass
                    return str(v).strip()

                out[pid] = PaperMeta(
                    paper_id=pid,
                    title=_get("title"),
                    authors=_get("authors"),
                    venue=_get("venue"),
                    year=_safe_int(row.get(cols.get("year")) if cols.get("year") else None),
                    volume=_get("volume"),
                    issue=_get("issue"),
                    pages=_get("pages"),
                    doi=_get("doi"),
                    source_xlsx=sx,
                )

        return out
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_wos_corpus.py

作者: 冯冉
创建时间: 2026-01-30
目标: 批量读取 WoS 导出的题录 Excel（savedrecs*.xls），完成：
    1) 目录级批量聚合（按 domain 子目录）
    2) 基础清洗（去空、规范字符串）
    3) 增加领域字段 domain
    4) 生成建模语料 text = title + [SEP] + abstract
    5) 全局去重（paper_id）但保留跨方向多标签映射（paper_domain_map）
    6) 导出中间结果（供后续方向内主题建模与知识库建设）

说明 :
- 为支持问答系统按 GB/T 7714-2015 输出参考文献，本脚本会尽量标准化并保留以下关键元数据字段：
  authors, title, journal, year, volume, issue, pages / article_number, doi, ut
- 同时保留原始 WoS 导出中的所有列（合并后列取并集），避免丢失潜在所需字段（如基金、机构、地址、关键词等）。
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# =========================
# 配置区
# =========================
ROOT_DIR = Path("题录信息")  
DOMAINS = ["深水油气", "深海矿产", "深海可再生能源", "深海感知与通信装备"]

OUT_DIR = Path("题录信息_中间结果")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 语料拼接分隔符：对 BERT/SPECTER/SciBERT/Embedding 友好，且可解释
SEP_TOKEN = " [SEP] "

# 是否要求必须同时具备 title 与 abstract 才纳入建模语料
# 若你希望尽量保留，可改为 False（仅要求 title）
REQUIRE_ABSTRACT = True


# =========================
# 1) WoS 列名候选映射（兼容不同导出版本）
#    说明：WoS 的“显示字段名”和“tag 名”会不同，下面尽量覆盖常见情况
# =========================
CAND_COLS: Dict[str, List[str]] = {
    # 参考文献核心字段（GB/T 7714-2015 常用到）
    "title": ["Article Title", "Title", "TI"],
    "abstract": ["Abstract", "AB"],
    "authors": ["Authors", "Author Full Names", "AU", "AF"],
    "journal": ["Source Title", "Publication Name", "SO", "Journal"],
    "year": ["Publication Year", "PY", "Year"],
    "volume": ["Volume", "VL"],
    "issue": ["Issue", "IS"],
    "pages": ["Pages", "BP", "Page Count"],
    "begin_page": ["Begin Page", "BP"],
    "end_page": ["End Page", "EP"],
    "article_number": ["Article Number", "AR"],
    "doi": ["DOI", "DI"],
    "ut": ["UT (Unique WOS ID)", "UT"],
    # 额外常用（建议保留，利于后续分析与溯源）
    "keywords": ["Author Keywords", "DE", "Keywords", "KeyWords Plus", "ID"],
    "doc_type": ["Document Type", "DT"],
    "addresses": ["Addresses", "C1"],
    "affiliations": ["Affiliations", "Organizations-Enhanced", "OO"],
    "funding": ["Funding Agency and Grant Number", "FU", "Funding Text", "FX"],
    "times_cited": ["Times Cited, WoS Core", "TC"],
    "publisher": ["Publisher", "PU"],   
}


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """从候选列名中找到第一个存在于 df.columns 的列名。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_str(x) -> str:
    """统一字符串清洗：去首尾空格、压缩多空格、清理 nan。"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    if s.lower() == "nan":
        return ""
    return s


def _safe_read_excel(path: Path) -> pd.DataFrame:
    """读取 Excel（savedrecs*.xls），失败时给出清晰报错。"""
    try:
        return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"读取失败：{path}；原因：{e}") from e


def _build_paper_id(row: pd.Series) -> str:
    """
    构建稳定 paper_id：优先 UT，其次 DOI，最后 fallback（title+year+journal）
    - 注意：跨方向去重必须依赖这个 paper_id
    """
    ut = row.get("ut_std", "")
    doi = row.get("doi_std", "")
    title = row.get("title_std", "")
    year = row.get("year_std", "")
    journal = row.get("journal_std", "")

    if ut:
        return f"UT={ut}"
    if doi:
        return f"DOI={doi}".lower()

    # 兜底：title + year + journal（尽量稳，但可能存在同名文章风险）
    fb = f"TITLE={title}|YEAR={year}|J={journal}".lower()
    return fb


def _std_pages(row: pd.Series) -> str:
    """
    标准化页码字段：
    - 优先使用 begin_page-end_page
    - 否则使用 pages
    - 否则使用 article_number
    """
    bp = row.get("begin_page_std", "")
    ep = row.get("end_page_std", "")
    pages = row.get("pages_std", "")
    ar = row.get("article_number_std", "")

    if bp and ep and bp != ep:
        return f"{bp}-{ep}"
    if bp and (not ep or bp == ep):
        return bp
    if pages:
        return pages
    if ar:
        return ar
    return ""


def main() -> None:
    # =========================
    # 批量读取：按 domain 子目录聚合
    # =========================
    all_frames: List[pd.DataFrame] = []

    for domain in DOMAINS:
        ddir = ROOT_DIR / domain
        if not ddir.exists():
            print(f"[WARN] 不存在目录：{ddir}", file=sys.stderr)
            continue

        files = sorted(ddir.glob("savedrecs*.xls"))
        if not files:
            print(f"[WARN] {domain} 下未找到 savedrecs*.xls", file=sys.stderr)
            continue

        for f in files:
            df = _safe_read_excel(f)
            df["domain"] = domain                 # 增加领域字段
            df["__source_file"] = f.name          # 保留来源文件（便于追溯）
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError("未读取到任何数据，请检查 ROOT_DIR 与子目录/文件名。")

    # 合并：列取并集（保留所有 WoS 原始元数据列）
    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"[INFO] 读入聚合记录数：{len(df_all)}")

    # =========================
    # 标准化关键字段（用于 GB/T 7714-2015 与后续建模）
    # =========================
    # 1. 找到各关键字段在当前数据中的列名
    col_map: Dict[str, Optional[str]] = {}
    for std_key, cand_list in CAND_COLS.items():
        col_map[std_key] = _pick_col(df_all, cand_list)

    # 2. 生成标准化列（*_std），并做字符串清洗
    def _make_std_col(std_key: str, out_name: str) -> None:
        src = col_map.get(std_key)
        if src is None:
            df_all[out_name] = ""
        else:
            df_all[out_name] = df_all[src].apply(_norm_str)

    _make_std_col("title", "title_std")
    _make_std_col("abstract", "abstract_std")
    _make_std_col("authors", "authors_std")
    _make_std_col("journal", "journal_std")
    _make_std_col("year", "year_std")
    _make_std_col("volume", "volume_std")
    _make_std_col("issue", "issue_std")
    _make_std_col("pages", "pages_std")
    _make_std_col("begin_page", "begin_page_std")
    _make_std_col("end_page", "end_page_std")
    _make_std_col("article_number", "article_number_std")
    _make_std_col("doi", "doi_std")
    _make_std_col("ut", "ut_std")

    # 可选元数据（对后续分析有帮助）
    _make_std_col("keywords", "keywords_std")
    _make_std_col("doc_type", "doc_type_std")
    _make_std_col("addresses", "addresses_std")
    _make_std_col("affiliations", "affiliations_std")
    _make_std_col("funding", "funding_std")
    _make_std_col("times_cited", "times_cited_std")
    _make_std_col("publisher", "publisher_std")

    # =========================
    # 去空：标题必须有；摘要按 REQUIRE_ABSTRACT 控制
    # =========================
    df_all["has_title"] = df_all["title_std"].ne("")
    df_all["has_abstract"] = df_all["abstract_std"].ne("")

    if REQUIRE_ABSTRACT:
        df_clean = df_all[df_all["has_title"] & df_all["has_abstract"]].copy()
    else:
        df_clean = df_all[df_all["has_title"]].copy()

    print(f"[INFO] 去空后记录数：{len(df_clean)}（REQUIRE_ABSTRACT={REQUIRE_ABSTRACT}）")

    # =========================
    # 生成语料 text（供主题建模/向量库）
    # =========================
    # 规范做法：Title. [SEP] Abstract
    df_clean["text"] = df_clean["title_std"] + "." + SEP_TOKEN + df_clean["abstract_std"]

    # =========================
    # 构建 paper_id（用于全局去重）+ 保留多标签映射
    # =========================
    df_clean["paper_id"] = df_clean.apply(_build_paper_id, axis=1)

    # 论文-方向映射表（多对多）：解决“一篇论文属于多个方向”的情况
    paper_domain_map = (
        df_clean[["paper_id", "domain"]]
        .drop_duplicates()
        .sort_values(["paper_id", "domain"])
        .reset_index(drop=True)
    )

    # 论文主表（全局去重）：每篇论文仅保留 1 条记录（但不丢掉方向关系）
    # 这里按 paper_id 去重；保留最关键字段 + 仍可保留原始列（合并后列很多）
    # 为了简洁与可用，主表建议保留“标准化字段 + 必要原始列”
    df_clean["pages_std_final"] = df_clean.apply(_std_pages, axis=1)

    # 主表字段：GB/T 7714-2015 输出常用 + 建模用 + 溯源用
    master_keep_cols = [
        "paper_id",
        "ut_std",
        "doi_std",
        "authors_std",
        "title_std",
        "journal_std",
        "year_std",
        "volume_std",
        "issue_std",
        "pages_std_final",
        "article_number_std",
        "doc_type_std",
        "keywords_std",
        "funding_std",
        "affiliations_std",
        "addresses_std",
        "times_cited_std",
        "publisher_std",
        "abstract_std",
        "text",
    ]

    # 有些列可能不存在（极少数情况），这里做保护
    master_keep_cols = [c for c in master_keep_cols if c in df_clean.columns]

    papers_master = (
        df_clean.sort_values(["paper_id", "domain", "__source_file"])
        .drop_duplicates(subset=["paper_id"])
        .loc[:, master_keep_cols]
        .reset_index(drop=True)
    )

    print(f"[INFO] 论文主表（全局去重后）：{len(papers_master)}")
    print(f"[INFO] 论文-方向映射（多标签边表）：{len(paper_domain_map)}")

    # =========================
    # 导出中间结果
    # =========================
    # 1. 全局：主表 + 映射表
    out_master = OUT_DIR / "papers_master_clean.xlsx"
    out_map = OUT_DIR / "paper_domain_map.xlsx"
    papers_master.to_excel(out_master, index=False)
    paper_domain_map.to_excel(out_map, index=False)

    # 2. 各方向：保留方向内全部清洗记录（不做全局去重，便于方向内建模）
    # 注意：方向内建模时可以直接读取该文件的 text 列
    domain_keep_cols = [
        "paper_id",
        "domain",
        "__source_file",
        "ut_std",
        "doi_std",
        "authors_std",
        "title_std",
        "journal_std",
        "year_std",
        "volume_std",
        "issue_std",
        "pages_std_final",
        "article_number_std",
        "doc_type_std",
        "keywords_std",
        "abstract_std",
        "text",
    ]
    domain_keep_cols = [c for c in domain_keep_cols if c in df_clean.columns]

    for domain in DOMAINS:
        sub = df_clean[df_clean["domain"] == domain].copy()
        # 重新算一次 pages_std_final（以防 sub 是 view）
        if "pages_std_final" not in sub.columns:
            sub["pages_std_final"] = sub.apply(_std_pages, axis=1)

        out_domain = OUT_DIR / f"corpus_{domain}_clean.xlsx"
        sub.loc[:, domain_keep_cols].to_excel(out_domain, index=False)

    # 可选）导出“全量清洗表”（保留所有原始列 + 标准化列）
    # 若你担心遗漏参考文献字段，可启用此导出（文件可能较大）
    # out_full = OUT_DIR / "records_full_clean_union.xlsx"
    # df_clean.to_excel(out_full, index=False)

    print(f"[DONE] 中间结果已导出到：{OUT_DIR.resolve()}")
    print(f"       - {out_master.name}（全局去重主表）")
    print(f"       - {out_map.name}（论文-方向多标签映射）")
    print(f"       - corpus_*.xlsx（四个方向建模语料）")
    # print(f"       - {out_full.name}（全量清洗并集表）")


if __name__ == "__main__":
    main()

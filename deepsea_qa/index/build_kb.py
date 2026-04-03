# -*- coding: utf-8 -*-
"""
build_kb.py
作者：
用途说明:
1) 语义单元切分，对prepare_data/题录信息_中间结果 下：
    corpus_*_clean.xlsx + paper_domain_label_map_*.xlsx
    -> 切分语义单元 chunks（带 offsets/标签/溯源信息）
    -> 写 deepsea_qa/artifacts/kb/chunks/*.jsonl

2) 写 SQLite：deepsea_qa/artifacts/kb/store/chunks.sqlite
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from deepsea_qa.configs.paths import (
    CHUNKS_ALL_JSONL,
    CHUNKS_DIR,
    DIR_CORPUS,
    SQLITE_PATH,
    STORE_DIR,
)
from deepsea_qa.configs.index_config import KBConfig
from deepsea_qa.data.chunking import TokenChunkingConfig, split_into_chunks_token_based


# -------------------------
# 1) Excel 读取与 domain 推断
# -------------------------

def infer_domain_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("corpus_") and name.endswith("_clean"):
        return name[len("corpus_"):-len("_clean")]
    if name.startswith("paper_domain_label_map_"):
        return name[len("paper_domain_label_map_"):]
    if name.startswith("llm_labeled_"):
        return name[len("llm_labeled_"):]
    return name


def load_corpus_excels(dir_corpus: Path) -> Dict[str, Tuple[Path, pd.DataFrame]]:
    corpora: Dict[str, Tuple[Path, pd.DataFrame]] = {}
    for p in sorted(dir_corpus.glob("corpus_*_clean.xlsx")):
        domain = infer_domain_from_filename(p)
        df = pd.read_excel(p)

        # 统一 paper_id / text
        if "paper_id" not in df.columns:
            for alt in ["UT", "PaperID", "id"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "paper_id"})
                    break
        if "text" not in df.columns:
            for alt in ["TI_AB", "content", "abstract_text", "文本", "摘要"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "text"})
                    break

        corpora[domain] = (p, df)
    return corpora


def load_label_maps(dir_corpus: Path) -> Dict[str, pd.DataFrame]:
    maps: Dict[str, pd.DataFrame] = {}
    for p in sorted(dir_corpus.glob("paper_domain_label_map_*.xlsx")):
        domain = infer_domain_from_filename(p)
        df = pd.read_excel(p)

        if "paper_id" not in df.columns:
            for alt in ["UT", "PaperID", "id"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "paper_id"})
                    break

        if "paper_id" not in df.columns:
            continue

        df["paper_id"] = df["paper_id"].astype(str).str.strip()
        maps[domain] = df
    return maps


def merge_corpus_with_label_map(df_corpus: pd.DataFrame, df_map: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df_corpus.copy()
    if "paper_id" not in df.columns or "text" not in df.columns:
        raise ValueError("corpus 缺少必要列：paper_id / text")

    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[(df["paper_id"] != "") & (df["text"] != "")].copy()

    if df_map is None:
        return df

    dfm = df_map.copy()
    dfm["paper_id"] = dfm["paper_id"].astype(str).str.strip()

    keep_cols = [
        "paper_id",
        "domain",
        "primary_label_id",
        "secondary_label_ids",
        "confidence",
        "labels_version",
        "label_source_json",
    ]
    cols = [c for c in keep_cols if c in dfm.columns]
    dfm = dfm[cols].drop_duplicates(subset=["paper_id"])

    return df.merge(dfm, on="paper_id", how="left")


def _get_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["PY", "Year", "year", "年份", "year_std"]:
        if c in df.columns:
            return c
    return None


def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


# -------------------------
# 2) SQLite store（内嵌）
# -------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    year INTEGER,
    source_xlsx TEXT,

    primary_label_id TEXT,
    secondary_label_ids TEXT,
    confidence REAL,
    labels_version TEXT,
    label_source_json TEXT,

    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,

    meta_json TEXT
);
"""

CREATE_INDEX_STMTS = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(domain);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);",
]

def init_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(CREATE_TABLE_SQL)
        for stmt in CREATE_INDEX_STMTS:
            conn.execute(stmt)


def upsert_many(db_path: Path, records: List[dict]) -> int:
    if not records:
        return 0

    sql = """
    INSERT OR REPLACE INTO chunks (
        chunk_id, text, paper_id, domain, year, source_xlsx,
        primary_label_id, secondary_label_ids, confidence, labels_version, label_source_json,
        chunk_index, start_char, end_char,
        meta_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    rows = []
    for r in records:
        meta = {"schema": "ChunkRecord/v1"}
        rows.append((
            r.get("chunk_id"),
            r.get("text"),
            r.get("paper_id"),
            r.get("domain"),
            r.get("year"),
            r.get("source_xlsx"),

            r.get("primary_label_id"),
            r.get("secondary_label_ids"),
            r.get("confidence"),
            r.get("labels_version"),
            r.get("label_source_json"),

            r.get("chunk_index"),
            r.get("start_char"),
            r.get("end_char"),

            json.dumps(meta, ensure_ascii=False),
        ))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executemany(sql, rows)

    return len(rows)


def sqlite_count(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT COUNT(1) FROM chunks;")
        return int(cur.fetchone()[0])


# -------------------------
# 3) Chunk 产出
# -------------------------

def iter_chunk_dicts(
    domain: str,
    corpus_path: Path,
    df: pd.DataFrame,
    chunk_cfg: TokenChunkingConfig,
) -> Iterator[dict]:
    year_col = _get_year_col(df)

    for _, row in df.iterrows():
        paper_id = str(row.get("paper_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not paper_id or not text:
            continue

        year = _safe_int(row.get(year_col)) if year_col else None

        primary_label_id = row.get("primary_label_id", None)
        secondary_label_ids = row.get("secondary_label_ids", None)

        chunks = split_into_chunks_token_based(text, chunk_cfg)
        for i, (chunk_text, s, e) in enumerate(chunks):
            yield {
                "chunk_id": f"{paper_id}__chunk_{i}",
                "text": chunk_text,
                "paper_id": paper_id,
                "domain": domain,
                "year": year,
                "source_xlsx": corpus_path.name,

                "primary_label_id": str(primary_label_id).strip() if pd.notna(primary_label_id) else None,
                "secondary_label_ids": str(secondary_label_ids).strip() if pd.notna(secondary_label_ids) else None,
                "confidence": _safe_float(row.get("confidence", None)),
                "labels_version": str(row.get("labels_version")).strip() if "labels_version" in row and pd.notna(row.get("labels_version")) else None,
                "label_source_json": str(row.get("label_source_json")).strip() if "label_source_json" in row and pd.notna(row.get("label_source_json")) else None,

                "chunk_index": i,
                "start_char": int(s) if s is not None else None,
                "end_char": int(e) if e is not None else None,

                # 写入 meta_json（可审计）
                "_tokenizer_name": chunk_cfg.tokenizer_name,
                "_min_tokens": chunk_cfg.min_tokens,
                "_max_tokens": chunk_cfg.max_tokens,
            }


def build_chunks_jsonl_and_sqlite() -> Dict[str, int]:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    corpora = load_corpus_excels(DIR_CORPUS)
    if not corpora:
        raise FileNotFoundError(f"未找到 corpus_*_clean.xlsx，请检查：{DIR_CORPUS}")

    label_maps = load_label_maps(DIR_CORPUS)
    init_sqlite(SQLITE_PATH)

    # 从配置中读取切分参数
    kb_cfg = KBConfig()
    
    # SciBERT token 切分参数
    chunk_cfg = TokenChunkingConfig(
        min_tokens=kb_cfg.chunk_size // 2,  # 最小块大小设为目标块大小的一半
        max_tokens=kb_cfg.chunk_size,        # 最大块大小设为目标块大小
        tokenizer_name="scibert_scivocab_uncased",
        # 如果你服务器没有缓存 SciBERT tokenizer：把这里改成 False（允许下载一次）
        local_files_only=True,
    )

    stats: Dict[str, int] = {}

    with CHUNKS_ALL_JSONL.open("w", encoding="utf-8") as f_all:
        for domain, (corpus_path, df_corpus) in corpora.items():
            df_map = label_maps.get(domain)
            df = merge_corpus_with_label_map(df_corpus, df_map)

            domain_jsonl = CHUNKS_DIR / f"chunks_{domain}.jsonl"
            n_domain = 0
            batch: List[dict] = []

            with domain_jsonl.open("w", encoding="utf-8") as f_dom:
                for rec in iter_chunk_dicts(domain, corpus_path, df, chunk_cfg):
                    # jsonl 里别保留下划线开头的内部字段
                    rec_out = {k: v for k, v in rec.items() if not k.startswith("_")}
                    line = json.dumps(rec_out, ensure_ascii=False)
                    f_dom.write(line + "\n")
                    f_all.write(line + "\n")

                    batch.append(rec)
                    n_domain += 1

                    if len(batch) >= 500:
                        upsert_many(SQLITE_PATH, batch)
                        batch = []

                if batch:
                    upsert_many(SQLITE_PATH, batch)

            stats[domain] = n_domain

    return stats
# -*- coding: utf-8 -*-
"""
token_stats_chunks.py
作者: 冯冉
创建时间：2026-02-22
用途说明:
- 扫描 prepare_data/题录信息_中间结果/corpus_*_clean.xlsx
- 将每条 paper 的 text 切分为 chunks（正则 + 长度约束）
- 统计 chunk token 分布（总体 + 分domain）
- 输出：
  1) Excel（多sheet）：token_stats/token_stats_chunks.xlsx
  2) 图片：token_stats/fig_chunk_tokens_hist_cdf.png
          token_stats/fig_chunk_tokens_boxplot_by_domain.png

说明（关于 tokenizer）：
- 优先尝试加载本地 transformers tokenizer（scibert / bert-base-uncased）
- 若本机无缓存且无法下载，则自动退化为“近似 token”（简单英文词/符号切分）
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) 路径设置（你只需保证 prepare_data 目录结构符合你的规划）
# =========================

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "题录信息_中间结果"
OUT_DIR = HERE / "token_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = OUT_DIR / "token_stats_chunks.xlsx"
FIG_HIST_CDF = OUT_DIR / "fig_chunk_tokens_hist_cdf.png"
FIG_BOXPLOT = OUT_DIR / "fig_chunk_tokens_boxplot_by_domain.png"


# =========================
# 1) Chunking（与你现在 deepsea_qa 里一致的逻辑：正则切分 + 长度约束）
# =========================

@dataclass
class ChunkingConfig:
    min_len: int = 80
    max_len: int = 600
    hard_max_len: int = 900
    split_pattern: str = r"(?:\n+|(?<=[。！？!?；;])\s+|(?<=\.)\s+|(?<=:)\s+|(?<=：)\s+)"


def _initial_split(text: str, cfg: ChunkingConfig) -> List[Tuple[str, int, int]]:
    text = str(text).strip()
    if not text:
        return []

    parts: List[Tuple[str, int, int]] = []
    last = 0
    for m in re.finditer(cfg.split_pattern, text):
        cut = m.start()
        seg = text[last:cut].strip()
        if seg:
            start = text.find(seg, last, cut)
            end = start + len(seg)
            parts.append((seg, start, end))
        last = m.end()

    tail = text[last:].strip()
    if tail:
        start = text.find(tail, last)
        end = start + len(tail)
        parts.append((tail, start, end))

    return parts


def _merge_short(parts: List[Tuple[str, int, int]], cfg: ChunkingConfig) -> List[Tuple[str, int, int]]:
    if not parts:
        return parts

    merged: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(parts):
        seg, s, e = parts[i]
        if len(seg) >= cfg.min_len:
            merged.append((seg, s, e))
            i += 1
            continue

        if i + 1 < len(parts):
            seg2, s2, e2 = parts[i + 1]
            new_seg = (seg + " " + seg2).strip()
            merged.append((new_seg, s, e2))
            i += 2
        else:
            if merged:
                pseg, ps, pe = merged[-1]
                merged[-1] = ((pseg + " " + seg).strip(), ps, e)
            else:
                merged.append((seg, s, e))
            i += 1

    return merged


def _split_long(seg: str, start: int, cfg: ChunkingConfig) -> List[Tuple[str, int, int]]:
    if len(seg) <= cfg.max_len:
        return [(seg, start, start + len(seg))]

    soft_delims = r"(?<=[，,、；;。！？!?])"
    pieces = [p.strip() for p in re.split(soft_delims, seg) if p.strip()]

    packed: List[str] = []
    cur = ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= cfg.max_len:
            cur = (cur + " " + p).strip()
        else:
            packed.append(cur)
            cur = p
    if cur:
        packed.append(cur)

    final: List[str] = []
    for p in packed:
        if len(p) <= cfg.hard_max_len:
            final.append(p)
        else:
            for k in range(0, len(p), cfg.max_len):
                final.append(p[k:k + cfg.max_len].strip())

    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for sub in final:
        idx = seg.find(sub, cursor)
        if idx < 0:
            idx = cursor
        sub_start = start + idx
        sub_end = sub_start + len(sub)
        out.append((sub, sub_start, sub_end))
        cursor = idx + len(sub)

    return out


def split_into_chunks(text: str, cfg: ChunkingConfig) -> List[Tuple[str, int, int]]:
    parts = _initial_split(text, cfg)
    parts = _merge_short(parts, cfg)

    out: List[Tuple[str, int, int]] = []
    for seg, s, _e in parts:
        out.extend(_split_long(seg, s, cfg))

    out2 = [(t, s, e) for (t, s, e) in out if len(t) >= max(20, cfg.min_len // 2)]
    if not out2 and str(text).strip():
        t = str(text).strip()
        out2 = [(t, 0, len(t))]
    return out2


# =========================
# 2) Tokenizer（优先 transformers，本地无缓存则退化为近似 token）
# =========================

class TokenCounter:
    def __init__(self):
        self.mode = "approx"  # transformers / approx
        self.name = "approx_regex_tokenizer"
        self._tok = None

        # 尝试 transformers tokenizer（不保证有网，所以只用本地缓存能成功的）
        try:
            from transformers import AutoTokenizer  # type: ignore

            candidates = [
                "~/scibert_scivocab_uncased",
                # "bert-base-uncased",
            ]
            for model_id in candidates:
                try:
                    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                    self._tok = tok
                    self.mode = "transformers"
                    self.name = model_id
                    break
                except Exception:
                    continue
        except Exception:
            pass

        # 如果 transformers 都失败，就保持 approx 模式

    def count(self, text: str) -> int:
        text = str(text)
        if self.mode == "transformers" and self._tok is not None:
            # 不加特殊符号，避免 [CLS]/[SEP] 等影响统计
            ids = self._tok.encode(text, add_special_tokens=False)
            return int(len(ids))

        # approx：英文常用分词近似（词 + 数字 + 标点独立）
        # 这不是严格 token，但足够用于“分布形态”判断与参数选择比较
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]", text)
        return int(len(tokens))


# =========================
# 3) 读取 corpus & 统计
# =========================

def infer_domain_from_filename(path: Path) -> str:
    name = path.stem
    if name.startswith("corpus_") and name.endswith("_clean"):
        return name[len("corpus_"):-len("_clean")]
    return name


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return df


def safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def find_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["PY", "Year", "year", "年份"]:
        if c in df.columns:
            return c
    return None


def summarize_series(s: pd.Series) -> pd.Series:
    # 基础统计 + 分位数
    qs = s.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
    return pd.Series({
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        "min": float(s.min()),
        "p05": float(qs.get(0.05, np.nan)),
        "p10": float(qs.get(0.10, np.nan)),
        "p25": float(qs.get(0.25, np.nan)),
        "p50": float(qs.get(0.50, np.nan)),
        "p75": float(qs.get(0.75, np.nan)),
        "p90": float(qs.get(0.90, np.nan)),
        "p95": float(qs.get(0.95, np.nan)),
        "max": float(s.max()),
    })


def main():
    # 你想对比不同切分参数时，只需要改这里
    chunk_cfg = ChunkingConfig(
        min_len=80,
        max_len=600,
        hard_max_len=900,
    )

    token_counter = TokenCounter()
    print("=== TokenCounter ===")
    print(f"mode={token_counter.mode} | name={token_counter.name}")
    print("====================")

    corpus_files = sorted(CORPUS_DIR.glob("corpus_*_clean.xlsx"))
    if not corpus_files:
        raise FileNotFoundError(f"未找到 corpus_*_clean.xlsx：{CORPUS_DIR}")

    rows = []  # chunk-level
    paper_rows = []  # paper-level

    for fp in corpus_files:
        domain = infer_domain_from_filename(fp)
        df = pd.read_excel(fp)
        df = ensure_cols(df)

        if "paper_id" not in df.columns or "text" not in df.columns:
            print(f"[SKIP] {fp.name} 缺少 paper_id/text，实际列：{list(df.columns)[:10]} ...")
            continue

        year_col = find_year_col(df)

        df["paper_id"] = df["paper_id"].astype(str).str.strip()
        df["text"] = df["text"].astype(str).fillna("").str.strip()
        df = df[(df["paper_id"] != "") & (df["text"] != "")].copy()

        for _, r in df.iterrows():
            paper_id = r["paper_id"]
            text = r["text"]
            year = safe_int(r.get(year_col)) if year_col else None

            # paper-level token
            paper_tok = token_counter.count(text)
            paper_rows.append({
                "domain": domain,
                "paper_id": paper_id,
                "year": year,
                "source_xlsx": fp.name,
                "paper_chars": len(text),
                "paper_tokens": paper_tok,
            })

            chunks = split_into_chunks(text, chunk_cfg)
            for i, (ct, s, e) in enumerate(chunks):
                rows.append({
                    "domain": domain,
                    "paper_id": paper_id,
                    "year": year,
                    "source_xlsx": fp.name,

                    "chunk_index": i,
                    "start_char": int(s),
                    "end_char": int(e),
                    "chunk_chars": len(ct),
                    "chunk_tokens": token_counter.count(ct),
                })

    df_chunks = pd.DataFrame(rows)
    df_papers = pd.DataFrame(paper_rows)

    if df_chunks.empty:
        raise RuntimeError("没有生成任何 chunk 记录，请检查 corpus 数据是否为空或列名不匹配。")

    # =========================
    # 4) 表格统计输出（Excel 多 sheet）
    # =========================
    overall = summarize_series(df_chunks["chunk_tokens"])
    overall_chars = summarize_series(df_chunks["chunk_chars"])
    overall_papers = summarize_series(df_papers["paper_tokens"]) if not df_papers.empty else pd.Series()

    df_overall = pd.DataFrame([
        {"metric": "chunk_tokens", **overall.to_dict()},
        {"metric": "chunk_chars", **overall_chars.to_dict()},
        {"metric": "paper_tokens", **overall_papers.to_dict()} if len(overall_papers) else {"metric": "paper_tokens"},
    ])

    by_domain = (
        df_chunks.groupby("domain")["chunk_tokens"]
        .apply(summarize_series)
        .reset_index()
        .rename(columns={"index": "stat"})
    )
    # groupby+apply 返回的结构是多列 stat，直接合并更清晰
    by_domain = df_chunks.groupby("domain")["chunk_tokens"].apply(summarize_series).reset_index()

    # 额外：按 domain 的 chunk 数、paper 数、avg chunks/paper
    count_extra = (
        df_chunks.groupby("domain")
        .agg(chunk_count=("chunk_tokens", "count"), paper_count=("paper_id", "nunique"))
        .reset_index()
    )
    count_extra["avg_chunks_per_paper"] = count_extra["chunk_count"] / count_extra["paper_count"].replace(0, np.nan)

    # 合并展示
    by_domain_full = count_extra.merge(by_domain, on="domain", how="left")

    # 原始明细（可选：抽样，避免太大）
    df_samples = df_chunks.sample(n=min(5000, len(df_chunks)), random_state=42).sort_values(["domain", "paper_id", "chunk_index"])

    # 写 Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df_overall.to_excel(w, sheet_name="overall_summary", index=False)
        by_domain_full.to_excel(w, sheet_name="by_domain_summary", index=False)
        df_papers.head(20000).to_excel(w, sheet_name="paper_level_head", index=False)  # 防止极大文件
        df_samples.to_excel(w, sheet_name="chunk_samples", index=False)

        # 写入参数与 tokenizer 信息（审稿可解释性）
        df_meta = pd.DataFrame([{
            "tokenizer_mode": token_counter.mode,
            "tokenizer_name": token_counter.name,
            "min_len_chars": chunk_cfg.min_len,
            "max_len_chars": chunk_cfg.max_len,
            "hard_max_len_chars": chunk_cfg.hard_max_len,
            "corpus_dir": str(CORPUS_DIR),
            "n_papers": int(df_papers.shape[0]),
            "n_chunks": int(df_chunks.shape[0]),
        }])
        df_meta.to_excel(w, sheet_name="meta", index=False)

    print(f"[OK] Excel saved: {OUT_XLSX}")

    # =========================
    # 5) 作图（美观：清晰、留白、可用于论文）
    # =========================

    # 5.1 总体分布：直方图 + CDF
    x = df_chunks["chunk_tokens"].values
    x = x[np.isfinite(x)]
    x_sorted = np.sort(x)
    y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    plt.figure(figsize=(10, 6), dpi=160)
    # bins：用分位数范围防止极端长尾毁图
    hi = np.percentile(x, 99.5)
    bins = int(min(80, max(20, round(np.sqrt(len(x))))))
    plt.hist(x[x <= hi], bins=bins, alpha=0.85, edgecolor="white", linewidth=0.6)
    plt.title("Chunk Token Distribution (Histogram + CDF)")
    plt.xlabel("Tokens per chunk")
    plt.ylabel("Count")

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x_sorted, y_cdf, linewidth=1.8)
    ax2.set_ylabel("CDF")

    # 标注关键分位数（审稿友好）
    for q in [50, 75, 90, 95]:
        v = np.percentile(x, q)
        ax1.axvline(v, linewidth=1.0, linestyle="--")
        ax1.text(v, ax1.get_ylim()[1] * 0.92, f"P{q}={v:.0f}", rotation=90, va="top", ha="right")

    plt.tight_layout()
    plt.savefig(FIG_HIST_CDF, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure saved: {FIG_HIST_CDF}")

    # 5.2 按 domain 箱线图（展示差异、发现异常域）
    domains = sorted(df_chunks["domain"].unique().tolist())
    data = [df_chunks.loc[df_chunks["domain"] == d, "chunk_tokens"].values for d in domains]

    plt.figure(figsize=(12, 6), dpi=160)
    plt.boxplot(
        data,
        labels=domains,
        showfliers=False,  # 不画离群点，避免长尾影响美观
        patch_artist=True,
        medianprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    plt.title("Chunk Token Distribution by Domain (Boxplot)")
    plt.ylabel("Tokens per chunk")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_BOXPLOT, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure saved: {FIG_BOXPLOT}")

    print("\n=== Done ===")
    print(f"Tokenizer: {token_counter.mode} | {token_counter.name}")
    print(f"Chunks: {len(df_chunks)} | Papers: {len(df_papers)}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
run_build_bm25.py

作者：Accilia
创建时间：2026-02-23
用途说明：BM25 建库入口（一次建好，后面融合检索直接用）

特点：
- 中文停用词：读取 prepare_data/stopwords/ 目录下所有txt（文件名不敏感）
- 英文停用词：sklearn 直接调用 + nltk 直接调用（缺失自动下载）
- 连续性：读取 A1/A2 的 SQLite chunks 表（chunk_id + text），不再重新切分

执行：
python -m deepsea_qa.run.run_build_bm25
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import os
import time
from typing import Iterator, Tuple, List

from deepsea_qa.retrieval.utils import build_stopwords, mixed_tokenize
from deepsea_qa.index.bm25_index import BM25Index, iter_chunks_from_sqlite, save_tokenizer_meta


# ========== 路径配置（对齐项目路径） ==========
from deepsea_qa.configs import paths
from deepsea_qa.configs.index_config import BM25Config

SQLITE_PATH = str(paths.SQLITE_PATH)
CN_STOPWORDS_DIR = str(paths.STOPWORDS_DIR)
OUT_DIR = str(paths.BM25_DIR)
BM25_PKL = str(paths.BM25_PKL_PATH)
TOKENIZER_META_JSON = str(paths.BM25_TOKENIZER_META_PATH)

# SQLite schema（与你 A1/A2 一致即可）
TABLE_NAME = "chunks"
CHUNK_ID_COL = "chunk_id"
TEXT_COL = "text"
# =====================================================
# BM25 配置
bm25_cfg = BM25Config()

# 分词参数
MIN_TOKEN_LEN = bm25_cfg.min_token_len
MAX_TOKENS_PER_DOC = bm25_cfg.max_tokens_per_doc

# “保留词白名单”（避免误删领域核心词）
KEEP_TERMS = {
    # English core
    "deep", "sea", "ocean", "marine", "subsea", "underwater", "seabed",
    "rov", "auv", "sonar", "acoustic", "hydrothermal", "pipeline",
    # Chinese core
    "深海", "海底", "海洋", "水下", "传感", "通信", "装备", "矿产", "可再生能源", "油气",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Build BM25 Index ===")
    print("[SQLite]", SQLITE_PATH)
    print("[CN stopwords dir]", CN_STOPWORDS_DIR)
    print("[Output]", OUT_DIR)

    # 1) 构建停用词（中文目录 + sklearn + nltk）
    #    nltk 若缺 stopwords，会自动下载（默认可联网）
    t0 = time.time()
    stopwords = build_stopwords(
        cn_stopwords_dir=CN_STOPWORDS_DIR,
        use_sklearn_en=True,
        use_nltk_en=True,
        nltk_download_if_missing=True,
    )
    print(f"[Stopwords] loaded={len(stopwords)} | seconds={time.time() - t0:.2f}")

    # 2) 流式读取 chunks，并 token 化
    def tokenized_docs() -> Iterator[Tuple[str, List[str]]]:
        for chunk_id, text in iter_chunks_from_sqlite(
            sqlite_path=SQLITE_PATH,
            table_name=TABLE_NAME,
            chunk_id_col=CHUNK_ID_COL,
            text_col=TEXT_COL,
        ):
            if not text.strip():
                continue

            tokens = mixed_tokenize(
                text=text,
                stopwords=stopwords,
                min_token_len=MIN_TOKEN_LEN,
                max_tokens=MAX_TOKENS_PER_DOC,
                keep_terms=KEEP_TERMS,
            )
            yield chunk_id, tokens

    # 3) 建 BM25
    t1 = time.time()
    bm25 = BM25Index.build(tokenized_docs=tokenized_docs(), cfg=bm25_cfg)
    bm25.save(BM25_PKL)
    print(f"[BM25] docs={len(bm25.art.doc_ids)} avgdl={bm25.art.avgdl:.2f} seconds={time.time() - t1:.2f}")

    # 4) 记录 tokenizer meta（用于论文复现）
    meta = {
        "bm25": {"k1": bm25_cfg.k1, "b": bm25_cfg.b},
        "tokenizer": {
            "type": "jieba+regex_mixed",
            "min_token_len": MIN_TOKEN_LEN,
            "max_tokens_per_doc": MAX_TOKENS_PER_DOC,
            "keep_terms_size": len(KEEP_TERMS),
            "stopwords_sources": {
                "cn_stopwords_dir": CN_STOPWORDS_DIR,
                "en_sklearn": True,
                "en_nltk": True,
                "nltk_download_if_missing": True,
            },
            "stopwords_total_size": len(stopwords),
        },
        "source_sqlite": {
            "path": SQLITE_PATH,
            "table": TABLE_NAME,
            "chunk_id_col": CHUNK_ID_COL,
            "text_col": TEXT_COL,
        },
        "artifacts": {
            "bm25_pkl": BM25_PKL,
            "tokenizer_meta_json": TOKENIZER_META_JSON,
        },
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_tokenizer_meta(meta, TOKENIZER_META_JSON)

    print("[OK] BM25 建库完成：")
    print(" -", BM25_PKL)
    print(" -", TOKENIZER_META_JSON)


if __name__ == "__main__":
    main()
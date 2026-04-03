# -*- coding: utf-8 -*-
"""
run_build_kb.py

作者：Accilia
创建时间：2026-02-22
用途说明: 一键运行创建知识库
使用说明:
  python -m deepsea_qa.run.run_build_kb
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

from deepsea_qa.configs.paths import CHUNKS_ALL_JSONL, SQLITE_PATH, DIR_CORPUS
from deepsea_qa.index.build_kb import build_chunks_jsonl_and_sqlite, sqlite_count


def main():
    print("=== DeepSeaQA | Build KB ===")
    print(f"[Input] DIR_CORPUS: {DIR_CORPUS}")

    stats = build_chunks_jsonl_and_sqlite()

    print("\n=== Done: Chunk stats by domain ===")
    total = 0
    for d, n in stats.items():
        print(f"  - {d}: {n}")
        total += n
    print(f"  => TOTAL chunks: {total}")

    n_db = sqlite_count(SQLITE_PATH)
    print("\n=== Outputs ===")
    print(f"[JSONL] {CHUNKS_ALL_JSONL}")
    print(f"[SQLITE] {SQLITE_PATH} (rows={n_db})")


if __name__ == "__main__":
    main()
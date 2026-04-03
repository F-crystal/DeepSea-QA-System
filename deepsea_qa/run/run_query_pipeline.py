# -*- coding: utf-8 -*-
"""
run_query_pipeline.py

作者：Accilia
创建时间：2026-02-23
用途说明：
  Query 调试入口（CLI壳），核心逻辑在 deepsea_qa.query.api 中。
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import os
# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.api_keys import set_api_key_env
set_api_key_env()

import time
import json
import argparse
import asyncio
from datetime import datetime

from deepsea_qa.query.api import build_query_bundle
from deepsea_qa.configs import paths
from deepsea_qa.configs.query_config import QueryConfig


OUT_DIR = str(paths.QUERY_ARTIFACTS_DIR)
DEBUG_DIR = str(paths.QUERY_DEBUG_DIR)


def _parse_args() -> argparse.Namespace:
    # 获取默认配置值
    default_cfg = QueryConfig()
    
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, default="", help="用户查询文本")
    p.add_argument("--save", action="store_true", help="是否保存输出到 artifacts/query/debug")

    p.add_argument("--no_cls", action="store_true", help="关闭分类（消融）")
    p.add_argument("--no_rewrite", action="store_true", help="关闭 rewrite+expand（消融）")
    p.add_argument("--max_sparse", type=int, default=default_cfg.pipeline.max_sparse_queries, help="BM25 候选 query 数量上限")

    p.add_argument("--llm_provider", type=str, default="zhipu", help="LLM provider，如 zhipu")
    p.add_argument("--llm_model", type=str, default="glm-4-plus", help="LLM model，如 glm-4-plus")
    return p.parse_args()


def main():
    args = _parse_args()
    query = (args.query or "").strip()
    if not query:
        raise ValueError("Empty --query. Please provide a query string.")

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.save:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    print("=== Build Query Bundle ===")
    print("[Query]", query)
    print("[Enable classification]", not args.no_cls)
    print("[Enable rewrite+expand]", not args.no_rewrite)
    print("[Max sparse queries]", args.max_sparse)
    print("[LLM provider]", args.llm_provider)
    print("[LLM model]", args.llm_model)
    print("[Output]", OUT_DIR)
    if args.save:
        print("[Debug]", DEBUG_DIR)

    t0 = time.time()
    obj = asyncio.run(
        build_query_bundle(
            query=query,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            enable_classification=(not args.no_cls),
            enable_rewrite_expand=(not args.no_rewrite),
            max_sparse_queries=args.max_sparse,
        )
    )
    print(f"[OK] done | seconds={time.time() - t0:.2f}")
    print(json.dumps(obj, ensure_ascii=False, indent=2))

    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(DEBUG_DIR, f"query_bundle_{ts}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print("[Saved]")
        print(" -", fp)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
run_retrieval_pipeline.py

作者：Accilia
创建时间：2026-02-24
用途说明：
  本地 CLI 壳（端到端）：
    query -> query_bundle（本地LLM） -> OpenBayes /retrieve（服务器检索） -> chunk_ids/evidence

运行示例：

# 方式A：端到端（推荐）
export OPENBAYES_API_KEY="sk-xxxxxxxxxxxxxxxx"
python -m deepsea_qa.run.run_retrieval_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --llm_provider zhipu --llm_model glm-4-plus \
  --final_top_n 10 \
  --save
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

# 方式B：复现（用已有 query_bundle.json）
python -m deepsea_qa.run.run_retrieval_pipeline \
  --query_bundle_path deepsea_qa/artifacts/query/debug/query_bundle_20260223_232929.json \
  --final_top_n 10 \
  --debug \
  --save
"""

from __future__ import annotations

import os
import time
import json
import argparse
from datetime import datetime

from deepsea_qa.configs import paths

from deepsea_qa.query.api import build_query_bundle_sync
from deepsea_qa.retrieval.api import retrieve_chunks_sync

# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.api_keys import set_api_key_env
set_api_key_env()

OUT_DIR = str(getattr(paths, "RETRIEVAL_ROOT", paths.ARTIFACTS_ROOT / "retrieval"))
DEBUG_DIR = str(getattr(paths, "RETRIEVAL_DEBUG_DIR", paths.ARTIFACTS_ROOT / "retrieval" / "debug"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # 二选一：query 或 query_bundle_path
    p.add_argument("--query", type=str, default="", help="用户查询文本（端到端模式）")
    p.add_argument("--query_bundle_path", type=str, default="", help="query_bundle.json（调试复现模式）")

    # OpenBayes /retrieve 的鉴权（推荐用环境变量 OPENBAYES_API_KEY）
    p.add_argument("--api_key", type=str, default="", help="OpenBayes API Key（可选，优先级高于环境变量）")
    p.add_argument("--api_path", type=str, default="/retrieve", help="服务端点路径（默认 /retrieve）")

    # QueryPipeline 参数（端到端时使用）
    p.add_argument("--no_cls", action="store_true", help="关闭分类（消融）")
    p.add_argument("--no_rewrite", action="store_true", help="关闭 rewrite+expand（消融）")
    p.add_argument("--max_sparse", type=int, default=8, help="BM25 候选 query 数量上限")
    p.add_argument("--llm_provider", type=str, default="zhipu", help="LLM provider")
    p.add_argument("--llm_model", type=str, default="glm-4-plus", help="LLM model")

    # 检索输出控制
    p.add_argument("--final_top_n", type=int, default=10)
    p.add_argument("--delta", type=float, default=0.1, help="动态截断阈值（默认0.1）：保留得分在top1-delta区间内的结果")
    p.add_argument("--no_evidence", action="store_true")
    p.add_argument("--debug", action="store_true")

    # 保存/网络
    p.add_argument("--save", action="store_true")
    p.add_argument("--timeout", type=int, default=300)

    return p.parse_args()


def _load_query_bundle(args: argparse.Namespace) -> dict:
    if args.query_bundle_path:
        with open(args.query_bundle_path, "r", encoding="utf-8") as f:
            return json.load(f)

    query = (args.query or "").strip()
    if not query:
        raise ValueError("Empty input: please provide --query OR --query_bundle_path")

    print("=== Build Query Bundle (End-to-End) ===")
    print("[Query]", query)
    print("[Enable classification]", not args.no_cls)
    print("[Enable rewrite+expand]", not args.no_rewrite)
    print("[Max sparse queries]", args.max_sparse)
    print("[LLM provider]", args.llm_provider)
    print("[LLM model]", args.llm_model)

    t0 = time.time()
    bundle = build_query_bundle_sync(
        query=query,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        enable_classification=(not args.no_cls),
        enable_rewrite_expand=(not args.no_rewrite),
        max_sparse_queries=args.max_sparse,
    )
    print(f"[OK] query bundle done | seconds={time.time() - t0:.2f}")
    return bundle


def main():
    args = _parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.save:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # 1) query -> bundle（或从文件读）
    bundle = _load_query_bundle(args)

    # 2) bundle -> OpenBayes /retrieve
    print("\n=== OpenBayes Retrieval ===")
    print("[API path]", args.api_path)
    print("[Final top N]", args.final_top_n)

    t0 = time.time()
    out = retrieve_chunks_sync(
        query_bundle=bundle,
        api_key=args.api_key,
        api_path=args.api_path,
        final_top_n=args.final_top_n,
        delta=args.delta,
        return_evidence=(not args.no_evidence),
        return_debug=args.debug,
        timeout=args.timeout,
    )
    print(f"[OK] retrieve done | seconds={time.time() - t0:.2f}")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 3) 保存
    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(DEBUG_DIR, f"retrieval_result_{ts}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("[Saved]")
        print(" -", fp)


if __name__ == "__main__":
    main()
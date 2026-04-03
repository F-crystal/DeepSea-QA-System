# -*- coding: utf-8 -*-
"""
run_generation_pipeline.py

作者：Accilia
创建时间：2026-02-25
用途说明：
生成阶段 CLI 壳

示例：
python -m deepsea_qa.run.run_generation_pipeline \
  --retrieval_result_path deepsea_qa/artifacts/retrieval/debug/retrieval_result_20260224_225107.json \
  --query_bundle_path deepsea_qa/artifacts/query/debug/query_bundle_20260223_232929.json \
  --llm_provider zhipu --llm_model glm-4-plus \
  --max_evidence 6 --max_retries 1 \
  --save

可选：--stream 先打印 evidence + 草稿答案的增量输出（LLM streaming体验）
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import os
from datetime import datetime

from deepsea_qa.configs import paths
from deepsea_qa.configs.generation_config import GenerationConfig
from deepsea_qa.generation.api import generate_answer_sync, stream_generation_events


def _parse_args():
    # 使用集中管理的默认配置
    default_cfg = GenerationConfig()
    
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval_result_path", type=str, required=True)
    p.add_argument("--query_bundle_path", type=str, required=True)

    p.add_argument("--llm_provider", type=str, default="zhipu")
    p.add_argument("--llm_model", type=str, default="glm-4-plus")

    p.add_argument("--max_evidence", type=int, default=default_cfg.max_evidence)
    p.add_argument("--max_retries", type=int, default=default_cfg.max_retries)
    p.add_argument("--final_top_n", type=int, default=default_cfg.final_top_n)

    p.add_argument("--stream", action="store_true", help="打印 evidence + 草稿答案 streaming 输出")
    p.add_argument("--save", action="store_true")

    return p.parse_args()


def main():
    args = _parse_args()

    with open(args.retrieval_result_path, "r", encoding="utf-8") as f:
        rr = json.load(f)

    with open(args.query_bundle_path, "r", encoding="utf-8") as f:
        qb = json.load(f)

    query = (rr.get("query", {}) or {}).get("original", "") or (qb.get("variants", {}) or {}).get("original", "")
    query = (query or "").strip()
    if not query:
        raise ValueError("Cannot find query from retrieval_result/query_bundle")

    if args.stream:
        for ev in stream_generation_events(
            query=query,
            retrieval_result=rr,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            max_evidence=args.max_evidence,
        ):
            print(json.dumps(ev, ensure_ascii=False))

    out = generate_answer_sync(
        query=query,
        query_bundle=qb,
        retrieval_result=rr,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        max_evidence=args.max_evidence,
        max_retries=args.max_retries,
        final_top_n=args.final_top_n,
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save:
        os.makedirs(str(paths.GEN_DEBUG_DIR), exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(str(paths.GEN_DEBUG_DIR), f"generation_result_{ts}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("[Saved]")
        print(" -", fp)


if __name__ == "__main__":
    main()
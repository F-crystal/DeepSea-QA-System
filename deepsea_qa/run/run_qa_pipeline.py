# -*- coding: utf-8 -*-
"""
run_qa_pipeline.py

作者：Accilia
创建时间：2026-02-25
用途说明：
端到端问答助手 CLI 壳：
query -> query_bundle -> retrieval(server) -> generation(local) -> verification -> retry

示例（非流式）：
python -m deepsea_qa.run.run_qa_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --llm_provider zhipu --llm_model glm-4-plus \
  --save

示例（流式：先证据卡片 + 草稿答案增量，再输出 final JSON）：
python -m deepsea_qa.run.run_qa_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --stream \
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
from deepsea_qa.configs.query_config import QueryConfig
from deepsea_qa.qa.pipeline import QAPipeline, QAPipelineConfig

# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.api_keys import set_api_key_env
set_api_key_env()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True)

    p.add_argument("--llm_provider", type=str, default="zhipu")
    p.add_argument("--llm_model", type=str, default="glm-4-plus")

    # 消融/调参入口（保持在 py 内部也行，但 run 提供更方便）
    p.add_argument("--no_cls", action="store_true")
    p.add_argument("--no_rewrite", action="store_true")
    p.add_argument("--max_sparse", type=int, default=QueryConfig().pipeline.max_sparse_queries)

    p.add_argument("--final_top_n", type=int, default=GenerationConfig().final_top_n)
    p.add_argument("--max_evidence", type=int, default=GenerationConfig().max_evidence)
    p.add_argument("--max_retries", type=int, default=GenerationConfig().max_retries)

    p.add_argument("--stream", action="store_true")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    query = args.query.strip()

    cfg = QAPipelineConfig(
        enable_classification=(not args.no_cls),
        enable_rewrite_expand=(not args.no_rewrite),
        max_sparse_queries=args.max_sparse,
        final_top_n=args.final_top_n,
        max_evidence=args.max_evidence,
        max_retries=args.max_retries,
    )
    pipe = QAPipeline(llm_provider=args.llm_provider, llm_model=args.llm_model, cfg=cfg)

    if args.stream:
        final_result = None
        for ev in pipe.stream(query):
            print(json.dumps(ev, ensure_ascii=False))
            # 捕获最终结果
            if ev.get("event") == "final":
                final_result = ev.get("data")
        
        # 保存最终结果
        if args.save and final_result:
            try:
                # 调试信息：打印保存路径
                save_dir = getattr(paths, "QA_DEBUG_DIR", paths.ARTIFACTS_ROOT / "qa" / "debug")
                save_dir_str = str(save_dir)
                print(f"[DEBUG] 保存目录: {save_dir_str}")
                
                # 创建目录
                os.makedirs(save_dir_str, exist_ok=True)
                print(f"[DEBUG] 目录创建成功")
                
                # 生成文件名
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fp = os.path.join(save_dir_str, f"qa_result_{ts}.json")
                print(f"[DEBUG] 保存文件路径: {fp}")
                
                # 检查final_result对象类型
                print(f"[DEBUG] final_result类型: {type(final_result)}")
                
                # 写入文件
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, ensure_ascii=False, indent=2)
                print("[Saved]")
                print(" -", fp)
            except Exception as e:
                print(f"[ERROR] 保存失败: {str(e)}")
                import traceback
                traceback.print_exc()
        return

    out = pipe.run(query)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save:
        try:
            # 调试信息：打印保存路径
            save_dir = getattr(paths, "QA_DEBUG_DIR", paths.ARTIFACTS_ROOT / "qa" / "debug")
            save_dir_str = str(save_dir)
            print(f"[DEBUG] 保存目录: {save_dir_str}")
            
            # 创建目录
            os.makedirs(save_dir_str, exist_ok=True)
            print(f"[DEBUG] 目录创建成功")
            
            # 生成文件名
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(save_dir_str, f"qa_result_{ts}.json")
            print(f"[DEBUG] 保存文件路径: {fp}")
            
            # 检查out对象类型
            print(f"[DEBUG] out类型: {type(out)}")
            
            # 写入文件
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print("[Saved]")
            print(" -", fp)
        except Exception as e:
            print(f"[ERROR] 保存失败: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
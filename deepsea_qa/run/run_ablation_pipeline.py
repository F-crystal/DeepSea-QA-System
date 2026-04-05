# -*- coding: utf-8 -*-
"""
run_ablation_pipeline.py

作者：Accilia
创建时间：2026-03-30
用途说明：
消融实验流水线 CLI 壳：
对消融实验数据集进行端到端评估，计算THELMA指标、分类指标和时间指标。

示例：
python run_ablation_pipeline.py --ablation_type no_cls
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from deepsea_qa.configs import paths
from deepsea_qa.configs.eval_config import EvalConfig
from deepsea_qa.configs.generation_config import GenerationConfig
from deepsea_qa.configs.query_config import QueryConfig
from deepsea_qa.eval.evaluator import EndToEndEvaluator
from deepsea_qa.qa.pipeline import QAPipeline, QAPipelineConfig

# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.api_keys import set_api_key_env
set_api_key_env()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="qa_post/qa_ablation_dataset.jsonl")
    
    p.add_argument("--llm_provider", type=str, default="zhipu", help="LLM provider, e.g., zhipu, dashscope")
    p.add_argument("--llm_model", type=str, default="glm-4-plus", help="LLM model name, e.g., glm-4-plus, qwen-plus")
    
    # 消融实验类型
    p.add_argument("--ablation_type", type=str, required=True, 
                   choices=["no_cls", "no_rewrite", "no_sparse", "no_rerank", "no_reverse_verification", "no_cls_rewrite"],
                   help="消融实验类型")
    
    # 消融/调参入口
    p.add_argument("--no_cls", action="store_true")
    p.add_argument("--no_rewrite", action="store_true")
    p.add_argument("--max_sparse", type=int, default=QueryConfig().pipeline.max_sparse_queries)
    p.add_argument("--no_rerank", action="store_true")
    p.add_argument("--no_reverse_verification", action="store_true")
    p.add_argument("--cls_strategy", type=str, default="llm", choices=["rules", "llm", "hybrid"], help="分类策略: rules(规则), llm(LLM), hybrid(混合)")
    
    p.add_argument("--final_top_n", type=int, default=GenerationConfig().final_top_n)
    p.add_argument("--max_evidence", type=int, default=GenerationConfig().max_evidence)
    
    # 限制评估的样本数量
    p.add_argument("--limit", type=int, default=None, help="限制评估的样本数量")
    p.add_argument("--max_retries", type=int, default=GenerationConfig().max_retries)
    
    # 断点续传
    p.add_argument("--start_from", type=int, default=0, help="从哪个索引开始评估")
    p.add_argument("--resume", action="store_true", help="从上次中断的地方继续评估")
    
    # 评估指标配置
    p.add_argument("--no_answer_quality", action="store_true", help="禁用回答质量指标（BLEU、精确率、召回率、BERTScore等）")
     
    return p.parse_args()


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """加载问答数据集"""
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                dataset.append(data)
            except Exception as e:
                print(f"错误：加载数据失败: {e}")
                continue
    return dataset


def main():
    args = _parse_args()
    
    # 根据消融类型设置参数
    if args.ablation_type == "no_cls":
        args.no_cls = True
    elif args.ablation_type == "no_rewrite":
        args.no_rewrite = True
    elif args.ablation_type == "no_sparse":
        args.max_sparse = 0  # 禁用稀疏检索，使RRF融合退化
    elif args.ablation_type == "no_rerank":
        args.no_rerank = True  # 禁用语义重排，但保留分类增强和截断逻辑
    elif args.ablation_type == "no_reverse_verification":
        args.no_reverse_verification = True
    elif args.ablation_type == "no_cls_rewrite":
        args.no_cls = True  # 同时禁用查询分类
        args.no_rewrite = True  # 同时禁用查询改写与拓展
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"数据集大小: {len(dataset)}")
    
    # 初始化配置
    # 强制使用LLM分类策略
    from deepsea_qa.configs.query_config import ClassifierConfig
    cls_cfg = ClassifierConfig(strategy=args.cls_strategy)
    
    cfg = QAPipelineConfig(
        enable_classification=(not args.no_cls),
        enable_rewrite_expand=(not args.no_rewrite),
        max_sparse_queries=args.max_sparse,
        enable_rerank=(not args.no_rerank),
        enable_reverse_verification=(not args.no_reverse_verification),
        final_top_n=args.final_top_n,
        max_evidence=args.max_evidence,
        max_retries=args.max_retries,
    )
    
    # 覆盖分类器配置
    cfg.classifier = cls_cfg
    
    # 初始化QA流水线
    pipe = QAPipeline(llm_provider=args.llm_provider, llm_model=args.llm_model, cfg=cfg)
    
    # 初始化评估器
    eval_cfg = EvalConfig()
    # 配置评估指标
    eval_cfg.answer_quality_metrics = not args.no_answer_quality
    evaluator = EndToEndEvaluator(eval_cfg)
    
    # 为不同消融实验类型创建不同目录
    ablation_output_dir = Path("qa_eval_ablation") / args.ablation_type
    ablation_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"创建/使用输出目录: {ablation_output_dir.absolute()}")
    
    # 断点续传逻辑 - 自动检查是否存在之前的评估结果
    start_idx = args.start_from
    
    # 自动检查是否需要续传
    auto_resume = False
    if start_idx == 0 and not args.resume:  # 只有当用户没有指定start_from且没有指定resume时才自动检查
        # 查找最新的评估结果文件
        if ablation_output_dir.exists():
            result_files = list(ablation_output_dir.glob("qa_eval_results_*.json"))
            if result_files:
                # 按修改时间排序，获取最新的文件
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_results = json.load(f)
                        # 只有当使用完整数据集时才自动续传
                        if "qa_ablation_dataset.jsonl" in args.dataset:
                            start_idx = len(existing_results)
                            if start_idx > 0:
                                auto_resume = True
                                print(f"检测到之前的评估结果，自动从上次中断的地方继续评估，已完成 {start_idx} 个样本")
                        else:
                            print("使用的是未评估数据集，从0开始评估")
                    except Exception as e:
                        print(f"错误：加载现有结果失败: {e}")
    
    # 如果用户明确指定了resume参数，也执行续传逻辑
    if args.resume and start_idx == 0:
        # 查找最新的评估结果文件
        if ablation_output_dir.exists():
            result_files = list(ablation_output_dir.glob("qa_eval_results_*.json"))
            if result_files:
                # 按修改时间排序，获取最新的文件
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_results = json.load(f)
                        # 只有当使用完整数据集时才续传
                        if "qa_ablation_dataset.jsonl" in args.dataset:
                            start_idx = len(existing_results)
                            print(f"从上次中断的地方继续评估，已完成 {start_idx} 个样本")
                        else:
                            print("使用的是未评估数据集，从0开始评估")
                    except Exception as e:
                        print(f"错误：加载现有结果失败: {e}")
    
    # 执行评估
    results = []
    total = len(dataset)
    eval_count = 0  # 累计评估样本数
    
    # 计算最大评估索引
    if args.limit:
        # 当指定limit时，在start_idx的基础上评估limit条数据
        max_eval_idx = start_idx + args.limit
    else:
        # 当未指定limit时，评估从start_idx到末尾的所有数据
        max_eval_idx = len(dataset)
    
    for i, item in enumerate(dataset[start_idx:], start=start_idx):
        # 检查是否超过最大评估索引
        if i >= max_eval_idx:
            break
            
        query = item.get('instruction', '').strip()
        ground_truth = item.get('output', '').strip()
        meta = item.get('meta', {})
        llm_label = meta.get('llm_label', '')
        
        # 组合ground_truth和llm_label
        combined_ground_truth = f"{ground_truth}|||{llm_label}"
        
        print(f"\n评估进度: {i+1 - start_idx}/{args.limit if args.limit else len(dataset[start_idx:])}")
        print(f"查询: {query}")
        
        try:
            result = evaluator.evaluate(query, combined_ground_truth, pipe, item)
            results.append(result)
            eval_count += 1  # 增加评估计数
            
            # 打印评估结果
            if result.metrics:
                print("评估指标:")
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    elif isinstance(value, list):
                        # 处理列表类型，如secondary_labels和candidates
                        print(f"  {key}:")
                        for item in value:
                            if isinstance(item, dict):
                                # 处理candidates中的字典
                                print(f"    - domain: {item.get('domain', '')}, label: {item.get('label', '')}, score: {item.get('score', 0.0):.4f}")
                            else:
                                print(f"    - {item}")
                    else:
                        print(f"  {key}: {value}")
            print(f"耗时: {result.timing.get('total_time', 0):.2f}秒")
            
            # 每评估5个样本保存一次中间结果
            if (i + 1) % 5 == 0:
                # 保存中间结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 加载已有的结果（如果存在）
                existing_results = []
                # 尝试加载之前的结果，包括中间结果
                result_files = list(ablation_output_dir.glob("qa_eval_results_*.json"))
                if result_files:
                    # 按修改时间排序，获取最新的文件
                    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        try:
                            existing_results = json.load(f)
                        except Exception as e:
                            print(f"错误：加载现有结果失败: {e}")
                
                # 合并结果
                all_results = existing_results + [result.__dict__ for result in results]
                
                # 保存合并后的结果
                intermediate_path = ablation_output_dir / f"qa_eval_results_{timestamp}_intermediate.json"
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                
                print(f"中间结果已保存至: {intermediate_path}")
                # 清空results，只保存新的结果
                results = []
                
        except Exception as e:
            print(f"错误：评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存批量评估的中间结果和最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载已有的结果（如果存在）
    existing_results = []
    # 尝试加载之前的结果，包括中间结果
    result_files = list(ablation_output_dir.glob("qa_eval_results_*.json"))
    if result_files:
        # 按修改时间排序，获取最新的文件
        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                print(f"加载了已有的结果，长度: {len(existing_results)}")
            except Exception as e:
                print(f"错误：加载现有结果失败: {e}")
    
    # 合并结果
    all_results = existing_results + [result.__dict__ for result in results]

    # 保存详细的中间结果
    intermediate_results = []
    for i, result in enumerate(all_results):
        intermediate_result = {
            "query": result.get('query', ''),
            "ground_truth": result.get('ground_truth', ''),
            "predicted_answer": result.get('predicted_answer', ''),
            "retrieval_results": result.get('retrieval_results', {}),
            "classification_result": result.get('classification_result', {}),
            "metrics": result.get('metrics', {}),
            "timing": result.get('timing', {})
        }
        intermediate_results.append(intermediate_result)
    
    # 保存详细中间结果
    intermediate_path = ablation_output_dir / f"qa_intermediate_results_{timestamp}.json"
    with open(intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(intermediate_results, f, ensure_ascii=False, indent=2)
    print(f"详细中间结果已保存至: {intermediate_path}")
    
    # 保存最终结果（与单独评估相同的格式）
    final_path = ablation_output_dir / f"qa_eval_results_{timestamp}.json"
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"最终结果已保存至: {final_path}")
    
    # 保存为Excel（用于文章）
    excel_path = ablation_output_dir / f"qa_eval_results_{timestamp}.xlsx"
    
    # 准备Excel数据
    rows = []
    for result in all_results:
        row = {
            "query": result.get('query', ''),
            "ground_truth": result.get('ground_truth', ''),
            "predicted_answer": result.get('predicted_answer', ''),
            "total_time": result.get('timing', {}).get("total_time", 0)
        }
        # 添加指标
        if result.get('metrics', {}):
            row.update(result.get('metrics', {}))
        rows.append(row)
    
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False, encoding='utf-8')
    print(f"Excel结果已保存至: {excel_path}")
    
    # 清理临时中间文件（只清理带有_intermediate后缀的文件）
    temp_intermediate_files = list(ablation_output_dir.glob("*_intermediate.json"))
    for file in temp_intermediate_files:
        try:
            file.unlink()
            print(f"已清理临时中间文件: {file}")
        except Exception as e:
            print(f"清理临时中间文件失败: {e}")
    
    # 使用累计的评估计数
    print(f"\n评估完成！共评估 {eval_count} 个样本")
    print(f"详细中间结果保存至: {intermediate_path}")
    print(f"最终结果保存至: {final_path}")
    print(f"Excel结果保存至: {excel_path}")


if __name__ == "__main__":
    main()

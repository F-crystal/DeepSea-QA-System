# -*- coding: utf-8 -*-
"""
evaluator.py

作者：Accilia
创建时间：2026-02-27
用途说明：评估器类，用于计算THELMA指标和其他评估指标。
"""

from __future__ import annotations

import json
import os
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# 导入中文分词
import jieba

from deepsea_qa.configs.eval_config import EvalConfig
from deepsea_qa.configs.paths import EVAL_OUTPUT_DIR, EVAL_LOGS_DIR
from deepsea_qa.qa.pipeline import QAPipeline
from deepsea_qa.eval.classification_utils import evaluate_classification
from deepsea_qa.eval.thelma.evaluator import ThelmaEvaluator as NewThelmaEvaluator
from deepsea_qa.llm.registry import get_llm, LLMSettings
from deepsea_qa.eval.api import compute_bertscore_sync, OPENBAYES_BASE_URL



@dataclass
class EvalResult:
    """评估结果"""
    query: str
    ground_truth: str
    predicted_answer: str
    retrieval_results: Dict[str, Any]
    classification_result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    timing: Optional[Dict[str, float]] = None


class EndToEndEvaluator:
    """端到端评估器"""
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        # 初始化LLM客户端（使用deepseek作为评估模型）
        settings = LLMSettings(provider="deepseek", model="deepseek-chat")
        self.llm_client = get_llm(settings)
        # 初始化新的THELMA评估器
        self.thelma_evaluator = NewThelmaEvaluator(cfg, self.llm_client)
        
        # 创建输出目录
        os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(EVAL_LOGS_DIR, exist_ok=True)
    
    def evaluate(self, query: str, ground_truth: str, qa_pipeline: QAPipeline, item: dict = None) -> EvalResult:
        """评估单个查询"""
        # 记录时间
        start_time = time.time()
        
        # 执行端到端问答
        result = qa_pipeline.run(query)
        
        # 计算时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 提取答案和检索结果
        # 尝试多种方式提取答案
        predicted_answer = result.get('answer', '')
        if not predicted_answer:
            predicted_answer = result.get('generation', {}).get('answer', '')
        
        # 提取来源
        sources = result.get('sources', [])
        if not sources:
            retrieval_results = result.get('retrieval_result', {})
            sources = retrieval_results.get('evidence', [])
        
        # 提取分类结果
        classification_result = result.get('classification', {})
        if not classification_result:
            classification_result = result.get('query_bundle', {}).get('classification', {})
        
        # 打印详细调试信息
        print(f"调试信息:")
        print(f"  查询: {query}")
        print(f"  回答: '{predicted_answer}'")
        print(f"  回答长度: {len(predicted_answer)}")
        print(f"  来源数量: {len(sources)}")
        
        # 打印来源详情
        for i, source in enumerate(sources):
            source_text = source.get('text', '') if isinstance(source, dict) else str(source)
            print(f"  来源 {i+1}: '{source_text[:50]}...'")
            print(f"  来源长度: {len(source_text)}")
        
        print(f"  分类结果: {classification_result}")
        
        # 分离ground_truth和llm_label
        if '|||' in ground_truth:
            gt_answer, gt_label = ground_truth.split('|||', 1)
        else:
            gt_answer, gt_label = ground_truth, ''
        
        # 从item中获取domain和副标签（如果有）
        gt_domain = ''
        gt_secondary_labels = []
        if item and isinstance(item, dict):
            meta = item.get('meta', {})
            if isinstance(meta, dict):
                gt_domain = meta.get('domain', '').strip()
                # 获取副标签
                secondary_labels = meta.get('secondary_labels', [])
                if isinstance(secondary_labels, list):
                    gt_secondary_labels = secondary_labels
        
        print(f"  真实标签: '{gt_label}'")
        print(f"  真实领域: '{gt_domain}'")
        if gt_secondary_labels:
            print(f"  真实副标签: {gt_secondary_labels}")
        
        # 计算指标
        metrics = {}
        
        # THELMA指标
        if self.cfg.thelma_metrics:
            thelma_metrics = self.thelma_evaluator.compute_thelma_metrics(query, predicted_answer, sources)
            metrics.update(thelma_metrics)
        
        # 分类指标
        if self.cfg.classification_metrics and classification_result:
            # 从item中获取副领域（如果有）
            gt_secondary_domains = []
            if item and isinstance(item, dict):
                meta = item.get('meta', {})
                if isinstance(meta, dict):
                    secondary_domains = meta.get('secondary_domains', [])
                    if isinstance(secondary_domains, list):
                        gt_secondary_domains = secondary_domains
            # 使用新的分类评估函数，传递副标签和副领域
            classification_metrics = evaluate_classification(
                classification_result, 
                gt_domain, 
                gt_label,
                gt_secondary_labels,
                gt_secondary_domains
            )
            metrics.update(classification_metrics)
        
        # 添加真实的副标签和副领域
        metrics['ground_truth_domain'] = gt_domain
        metrics['ground_truth_label'] = gt_label
        metrics['ground_secondary_label'] = gt_secondary_labels
        # 从item中获取副领域（如果有）
        gt_secondary_domains = []
        if item and isinstance(item, dict):
            meta = item.get('meta', {})
            if isinstance(meta, dict):
                secondary_domains = meta.get('secondary_domains', [])
                if isinstance(secondary_domains, list):
                    gt_secondary_domains = secondary_domains
        metrics['ground_secondary_domain'] = gt_secondary_domains
        
        # 回答质量指标
        if self.cfg.answer_quality_metrics:
            # 1. BLEU分数（支持中文）
            try:
                def calculate_bleu_chinese(predicted_answer, gt_answer):
                    """
                    计算中文 BLEU 分数
                    策略：使用 jieba 进行词语分词，然后利用 sacrebleu 计算
                    """
                    metrics = {'bleu_score': 0.0}

                    # 1. 数据校验
                    if not isinstance(predicted_answer, str) or not isinstance(gt_answer, str):
                        return metrics
                    
                    pred_clean = predicted_answer.strip()
                    gt_clean = gt_answer.strip()

                    if not pred_clean or not gt_clean:
                        return metrics

                    try:
                        # 2. 中文分词 (关键：必须按词分，不能按字分)
                        # 使用 jieba 精确模式分词
                        pred_tokens = " ".join(jieba.cut(pred_clean))
                        ref_tokens = " ".join(jieba.cut(gt_clean))
                        
                        # 3. 初始化 BLEU 对象
                        # effective_order=True: 对于单句计算非常重要，它会根据句子长度自动调整 n-gram 的最大阶数
                        # 避免因为句子太短（比如只有2个字）而强行计算 4-gram 导致得分为 0
                        from sacrebleu.metrics import BLEU
                        bleu = BLEU(effective_order=True)
                        
                        # 4. 计算分数
                        # sentence_score 接受 candidate (str) 和 references (List[str])
                        # 注意：sacrebleu 期望输入的是已经用空格分隔好词的字符串，或者是原始字符串配合 tokenize 参数
                        # 这里我们传入已经分好词并用空格连接的字符串
                        score_obj = bleu.sentence_score(pred_tokens, [ref_tokens])
                        
                        # score 属性返回的是 0-100 之间的值，通常我们需要归一化到 0-1
                        bleu_score = score_obj.score / 100.0
                        
                        metrics['bleu_score'] = float(bleu_score)
                        
                    except Exception as e:
                        print(f"计算 BLEU 分数时出错: {e}")
                        # 保持默认 0.0
                        
                    return metrics

                # 调用函数计算BLEU分数
                bleu_metrics = calculate_bleu_chinese(predicted_answer, gt_answer)
                # 将计算结果更新到metrics字典中
                metrics.update(bleu_metrics)
            except Exception as e:
                print(f"计算BLEU分数时出错: {e}")
                metrics['bleu_score'] = 0.0
            
            # 2. ROUGE分数（支持中文）
            try:
                def calculate_rouge_chinese(predicted_answer, gt_answer):
                    """
                    计算中文 ROUGE 分数
                    策略：先使用 jieba 分词，再用空格连接，最后传给标准 rouge 库
                    """
                    metrics = {
                        'rouge_1_f1': 0.0,
                        'rouge_2_f1': 0.0,
                        'rouge_l_f1': 0.0
                    }

                    # 1. 数据清洗与校验
                    # 处理 None 或 非字符串类型
                    if not isinstance(predicted_answer, str) or not isinstance(gt_answer, str):
                        return metrics
                    
                    pred_clean = predicted_answer.strip()
                    gt_clean = gt_answer.strip()

                    # 如果任意一方为空，直接返回 0
                    if not pred_clean or not gt_clean:
                        return metrics

                    try:
                        from rouge import Rouge
                        rouge = Rouge()
                        
                        # 2. 中文分词处理 (关键步骤)
                        # 使用 jieba 切分中文，然后用空格连接，模拟英文单词间隔
                        pred_tokens = " ".join(jieba.cut(pred_clean))
                        gt_tokens = " ".join(jieba.cut(gt_clean))
                        
                        # 再次检查分词后是否为空（防止极端情况）
                        if not pred_tokens or not gt_tokens:
                            return metrics

                        # 3. 计算分数
                        # avg=True 会直接返回字典形式的平均分
                        scores = rouge.get_scores(pred_tokens, gt_tokens, avg=True)
                        
                        # 4. 提取 F1 分数
                        # 注意：rouge 库返回的键是小写的 'rouge-1', 'rouge-2', 'rouge-l'
                        metrics['rouge_1_f1'] = float(scores['rouge-1']['f'])
                        metrics['rouge_2_f1'] = float(scores['rouge-2']['f'])
                        metrics['rouge_l_f1'] = float(scores['rouge-l']['f'])
                        
                    except Exception as e:
                        print(f"计算 ROUGE 分数时出错: {e}")
                        # 出错时保持默认值 0.0，避免程序崩溃
                        
                    return metrics

                # 调用函数计算ROUGE分数
                rouge_metrics = calculate_rouge_chinese(predicted_answer, gt_answer)
                # 将计算结果更新到metrics字典中
                metrics.update(rouge_metrics)
            except Exception as e:
                print(f"计算ROUGE分数时出错: {e}")
                metrics['rouge_1_f1'] = 0.0
                metrics['rouge_2_f1'] = 0.0
                metrics['rouge_l_f1'] = 0.0
            
            # 3. 回答长度指标
            metrics['answer_length'] = len(predicted_answer)
            metrics['ground_truth_length'] = len(gt_answer)
            
            # 4. BERTScore
            try:
                # 使用eval/api.py中的函数计算BERTScore
                bertscore_result = compute_bertscore_sync([predicted_answer], [gt_answer])
                if bertscore_result and bertscore_result.get('scores'):
                    scores = bertscore_result.get('scores', [])
                    if scores:
                        metrics['bert_score_precision'] = scores[0].get('precision', 0.0)
                        metrics['bert_score_recall'] = scores[0].get('recall', 0.0)
                        metrics['bert_score_f1'] = scores[0].get('f1', 0.0)
                    else:
                        # 如果服务器调用失败，使用默认值
                        print(f"计算BERTScore时出错: {bertscore_result.get('error')}")
                        metrics['bert_score_precision'] = 0.0
                        metrics['bert_score_recall'] = 0.0
                        metrics['bert_score_f1'] = 0.0
                else:
                    # 如果服务器调用失败，使用默认值
                    print(f"计算BERTScore时出错: {bertscore_result.get('error')}")
                    metrics['bert_score_precision'] = 0.0
                    metrics['bert_score_recall'] = 0.0
                    metrics['bert_score_f1'] = 0.0
            except Exception as e:
                print(f"计算BERTScore时出错: {e}")
                metrics['bert_score_precision'] = 0.0
                metrics['bert_score_recall'] = 0.0
                metrics['bert_score_f1'] = 0.0
        
        # 时间指标
        timing = {
            "total_time": total_time
        }
        
        return EvalResult(
            query=query,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
            retrieval_results=result.get('retrieval_result', {}),
            classification_result=classification_result,
            metrics=metrics,
            timing=timing
        )
    
    def save_results(self, results: List[EvalResult], timestamp: str):
        """保存评估结果"""
        # 保存为JSON（详细）
        json_path = EVAL_OUTPUT_DIR / f"eval_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([result.__dict__ for result in results], f, ensure_ascii=False, indent=2)
        
        # 保存为Excel（用于文章）
        excel_path = EVAL_OUTPUT_DIR / f"eval_results_{timestamp}.xlsx"
        
        # 准备Excel数据
        rows = []
        for result in results:
            row = {
                "query": result.query,
                "ground_truth": result.ground_truth,
                "predicted_answer": result.predicted_answer,
                "total_time": result.timing.get("total_time", 0)
            }
            # 添加指标
            if result.metrics:
                row.update(result.metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False, encoding='utf-8')
        
        # 计算并保存汇总统计
        summary_path = EVAL_OUTPUT_DIR / f"eval_summary_{timestamp}.json"
        summary = self.compute_summary(results)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存：")
        print(f"- JSON: {json_path}")
        print(f"- Excel: {excel_path}")
        print(f"- 汇总: {summary_path}")
    
    def compute_summary(self, results: List[EvalResult]) -> Dict[str, float]:
        """计算汇总统计"""
        summary = {}
        
        if not results:
            return summary
        
        # 计算平均指标
        metric_sums = {}
        for result in results:
            if result.metrics:
                for key, value in result.metrics.items():
                    if key not in metric_sums:
                        metric_sums[key] = 0
                    metric_sums[key] += value
        
        for key, value in metric_sums.items():
            summary[f"avg_{key}"] = value / len(results)
        
        # 计算平均时间
        total_time = sum(result.timing.get("total_time", 0) for result in results)
        summary["avg_total_time"] = total_time / len(results)
        
        return summary

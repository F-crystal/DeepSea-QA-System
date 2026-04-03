# -*- coding: utf-8 -*-
"""
evaluator.py

作者：Accilia
创建时间：2026-02-27
用途说明：
THELMA评估器
整合分解模块、匹配模块和聚合模块
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

from deepsea_qa.configs.eval_config import EvalConfig
from deepsea_qa.llm.base import BaseLLM
from deepsea_qa.eval.thelma.modules.decompose import DecomposeModule
from deepsea_qa.eval.thelma.modules.match import MatchModule
from deepsea_qa.eval.thelma.modules.aggregate import AggregateModule


class ThelmaEvaluator:
    """THELMA评估器"""
    
    def __init__(self, cfg: EvalConfig, llm_client: Optional[BaseLLM] = None):
        self.cfg = cfg
        self.llm_client = llm_client
        
        # 初始化模块
        self.decompose_module = DecomposeModule(llm_client)
        self.match_module = MatchModule(llm_client)
        self.aggregate_module = AggregateModule()
    
    def compute_thelma_metrics(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算所有THELMA指标"""
        metrics = {}
        
        # 提取源文本
        source_texts = []
        for source in sources:
            if isinstance(source, dict):
                text = source.get('text', '')
            else:
                text = str(source)
            if text:
                source_texts.append(text)
        
        if not source_texts:
            return metrics
        
        # 1. 源精度 (Source Precision)
        for variant in self.cfg.sp_variants:
            sp_score = self._compute_source_precision(source_texts, query, variant)
            metrics[f"source_precision_{variant}"] = sp_score
        
        # 2. 落地性 (Groundedness)
        groundedness_score = self._compute_groundedness(answer, source_texts)
        metrics["groundedness"] = groundedness_score
        
        # 3. 源查询覆盖率 (Source Query Coverage)
        source_query_coverage_score = self._compute_source_query_coverage(source_texts, query)
        metrics["source_query_coverage"] = source_query_coverage_score
        
        # 4. 响应查询覆盖率 (Response Query Coverage)
        response_query_coverage_score = self._compute_response_query_coverage(answer, query)
        metrics["response_query_coverage"] = response_query_coverage_score
        
        # 5. 响应精度 (Response Precision)
        response_precision_score = self._compute_response_precision(answer, query)
        metrics["response_precision"] = response_precision_score
        
        # 6. 响应自区分度 (Response Self-Distinctness)
        response_self_distinctness_score = self._compute_response_self_distinctness(answer)
        metrics["response_self_distinctness"] = response_self_distinctness_score
        
        return metrics
    
    def _compute_source_precision(self, source_texts: List[str], query: str, variant: str = "sp1") -> float:
        """计算源精度"""
        if not source_texts:
            return 0.0
        
        scores = []
        
        if variant == "sp1":
            # SP1: 整体源判断 (D_{id})
            for text in source_texts:
                # 恒等分解
                units = self.decompose_module.identity_decompose(text)
                for unit in units:
                    # 判断必要性
                    is_essential = self.match_module.judge_essentiality(query, unit)
                    scores.append(1.0 if is_essential else 0.0)
        elif variant == "sp2":
            # SP2: 分解为原子事实 (D_{text})
            for text in source_texts:
                # 分解为原子事实
                claims = self.decompose_module.decompose_text(text)
                for claim in claims:
                    # 判断必要性
                    is_essential = self.match_module.judge_essentiality(query, claim)
                    scores.append(1.0 if is_essential else 0.0)
        else:
            return 0.0
        
        # 聚合
        return self.aggregate_module.source_precision_aggregate(scores)
    
    def _compute_groundedness(self, answer: str, source_texts: List[str]) -> float:
        """计算落地性"""
        if not answer:
            return 0.0
        
        # 分解回答为声明 (D_{text})
        claims = self.decompose_module.decompose_text(answer)
        if not claims:
            return 0.0
        
        scores = []
        for claim in claims:
            # 判断声明是否被源支持
            is_supported = self.match_module.judge_support(claim, source_texts)
            scores.append(1.0 if is_supported else 0.0)
        
        # 聚合
        return self.aggregate_module.groundedness_aggregate(scores)
    
    def _compute_source_query_coverage(self, source_texts: List[str], query: str) -> float:
        """计算源查询覆盖率"""
        # 分解查询为子查询 (D_{qcov})
        sub_questions = self.decompose_module.decompose_query(query)
        if not sub_questions:
            return 0.0
        
        scores = []
        for sub_q in sub_questions:
            # 检查是否有任何来源包含这个子查询的答案
            max_score = 0.0
            for source in source_texts:
                contains = self.match_module.contains_answer(sub_q, source)
                if contains:
                    max_score = 1.0
                    break
            scores.append(max_score)
        
        # 聚合
        return self.aggregate_module.source_query_coverage_aggregate(scores)
    
    def _compute_response_query_coverage(self, answer: str, query: str) -> float:
        """计算响应查询覆盖率"""
        # 分解查询为子查询 (D_{qcov})
        sub_questions = self.decompose_module.decompose_query(query)
        if not sub_questions:
            return 0.0
        
        if not answer:
            return 0.0
        
        scores = []
        for sub_q in sub_questions:
            # 验证回答是否解决了子问题的意图
            covers = self.match_module.covers_intent(sub_q, answer)
            scores.append(1.0 if covers else 0.0)
        
        # 聚合
        return self.aggregate_module.response_query_coverage_aggregate(scores)
    
    def _compute_response_precision(self, answer: str, query: str) -> float:
        """计算响应精度"""
        if not answer:
            return 0.0
        
        # 分解回答为声明 (D_{text})
        claims = self.decompose_module.decompose_text(answer)
        if not claims:
            return 0.0
        
        scores = []
        for claim in claims:
            # 判断必要性
            is_essential = self.match_module.judge_essentiality(query, claim)
            scores.append(1.0 if is_essential else 0.0)
        
        # 聚合
        return self.aggregate_module.response_precision_aggregate(scores)
    
    def _compute_response_self_distinctness(self, answer: str) -> float:
        """计算响应自区分度"""
        # 分解回答为句子 (D_{sen})
        sentences = self.decompose_module.decompose_sentences(answer)
        if len(sentences) <= 1:
            return 1.0
        
        similarity_scores = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                # 计算相似度
                similarity = self.match_module.compute_similarity(sentences[i], sentences[j])
                similarity_scores.append(similarity)
        
        # 聚合
        return self.aggregate_module.response_self_distinctness_aggregate(similarity_scores)

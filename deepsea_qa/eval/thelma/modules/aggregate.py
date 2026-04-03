# -*- coding: utf-8 -*-
"""
aggregate.py

作者：Accilia
创建时间：2026-02-27
用途说明：
聚合模块 (Aggregate Module)
实现THELMA框架中的各种聚合策略
"""

from typing import List, Callable, Any


class AggregateModule:
    """聚合模块"""
    
    def macro_average(self, scores: List[float]) -> float:
        """宏观平均
        
        Args:
            scores: 分数列表
            
        Returns:
            平均分数
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    
    def source_precision_aggregate(self, scores: List[float]) -> float:
        """源精度聚合
        
        Args:
            scores: 必要性判断分数列表
            
        Returns:
            源精度分数
        """
        return self.macro_average(scores)
    
    def groundedness_aggregate(self, scores: List[float]) -> float:
        """落地性聚合
        
        Args:
            scores: 支持性判断分数列表
            
        Returns:
            落地性分数
        """
        return self.macro_average(scores)
    
    def source_query_coverage_aggregate(self, sub_query_scores: List[float]) -> float:
        """源查询覆盖率聚合
        
        Args:
            sub_query_scores: 子查询覆盖分数列表
            
        Returns:
            源查询覆盖率分数
        """
        return self.macro_average(sub_query_scores)
    
    def response_query_coverage_aggregate(self, sub_query_scores: List[float]) -> float:
        """响应查询覆盖率聚合
        
        Args:
            sub_query_scores: 子查询覆盖分数列表
            
        Returns:
            响应查询覆盖率分数
        """
        return self.macro_average(sub_query_scores)
    
    def response_precision_aggregate(self, scores: List[float]) -> float:
        """响应精度聚合
        
        Args:
            scores: 必要性判断分数列表
            
        Returns:
            响应精度分数
        """
        return self.macro_average(scores)
    
    def response_self_distinctness_aggregate(self, similarity_scores: List[float]) -> float:
        """响应自区分度聚合
        
        Args:
            similarity_scores: 句子对相似度分数列表
            
        Returns:
            响应自区分度分数
        """
        if not similarity_scores:
            return 1.0
        
        # 计算平均互异度
        avg_distinctness = sum(1 - score for score in similarity_scores) / len(similarity_scores)
        return avg_distinctness

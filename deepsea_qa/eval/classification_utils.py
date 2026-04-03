# -*- coding: utf-8 -*-
"""
classification_utils.py

作者：Accilia
创建时间：2026-02-27
用途说明：
分类判断工具，用于将label_id映射到中文名称，并实现分类评估逻辑。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any

from deepsea_qa.configs import paths


class LabelMapper:
    """标签映射器，用于将label_id映射到中文名称"""
    
    def __init__(self):
        """初始化标签映射器"""
        self.label_mapping: Dict[str, str] = {}
        self.domain_mapping: Dict[str, str] = {
            "SENSOR_COMM": "深海感知与通信装备",
            "RENEWABLE": "深海可再生能源",
            "MINERALS": "深海矿产",
            "OIL_GAS": "深水油气"
        }
        self._load_label_cards()
    
    def _load_label_cards(self):
        """加载分类标签文件"""
        label_cards_dir = os.path.join("prepare_data", "分类标签")
        if not os.path.exists(label_cards_dir):
            print(f"警告：分类标签目录不存在: {label_cards_dir}")
            return
        
        # 加载所有分类标签文件
        for filename in os.listdir(label_cards_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(label_cards_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "labels" in data:
                            for label in data["labels"]:
                                label_id = label.get("label_id", "")
                                label_name_zh = label.get("label_name_zh", "")
                                if label_id and label_name_zh:
                                    self.label_mapping[label_id] = label_name_zh
                except Exception as e:
                    print(f"警告：加载标签文件失败 {file_path}: {e}")
    
    def get_label_name(self, label_id: str) -> Optional[str]:
        """根据label_id获取中文名称"""
        return self.label_mapping.get(label_id)
    
    def get_domain_name(self, domain_id: str) -> Optional[str]:
        """根据domain_id获取中文名称"""
        return self.domain_mapping.get(domain_id)
    
    def map_labels(self, labels: List[str]) -> List[str]:
        """将label_id列表映射为中文名称列表"""
        return [self.get_label_name(label) or label for label in labels]
    
    def is_match(self, predicted_label_id: str, ground_truth_name: str) -> bool:
        """判断预测的label_id是否与真实的中文名称匹配"""
        predicted_name = self.get_label_name(predicted_label_id)
        return predicted_name == ground_truth_name


class ClassificationEvaluator:
    """分类评估器"""
    
    def __init__(self):
        """初始化分类评估器"""
        self.label_mapper = LabelMapper()
    
    def evaluate_classification(
        self,
        predicted: Dict[str, Any],
        ground_truth_domain: str,
        ground_truth_label: str,
        ground_truth_secondary_labels: List[str] = None,
        ground_truth_secondary_domains: List[str] = None
    ) -> Dict[str, float]:
        """评估分类结果"""
        metrics = {}
        
        if not predicted:
            return metrics
        
        # 提取预测结果
        predicted_domain_id = predicted.get('domain_id', '').strip()
        predicted_primary_label_id = predicted.get('primary_label_id', '').strip()
        secondary_label_ids = predicted.get('secondary_label_ids', [])
        candidates = predicted.get('candidates', [])
        
        # 映射预测的domain和label
        predicted_domain_name = self.label_mapper.get_domain_name(predicted_domain_id) or predicted_domain_id
        predicted_primary_label_name = self.label_mapper.get_label_name(predicted_primary_label_id) or predicted_primary_label_id
        
        # 计算domain匹配得分
        domain_correct = 0.0
        if ground_truth_domain:
            # 检查主领域是否命中
            if predicted_domain_name == ground_truth_domain:
                domain_correct = 1.0
            # 检查副领域是否命中
            elif ground_truth_secondary_domains and isinstance(ground_truth_secondary_domains, list):
                if predicted_domain_name in ground_truth_secondary_domains:
                    domain_correct = 0.8
        else:
            domain_correct = 1.0
        
        # 计算label匹配得分
        label_correct = 0.0
        best_rank_score = 0.0
        
        # 定义位置分数
        position_scores = {
            'primary': 1.0,                # 主标签位置
            'secondary': lambda i: max(0.0, 0.8 - 0.1 * i),  # 副标签位置
            'candidate': lambda i: max(0.0, 0.5 - 0.1 * i)   # 候选标签位置
        }
        
        # 首先检查真实主标签的匹配情况（优先级最高）
        if ground_truth_label:
            # 1. 检查预测主标签是否匹配真实主标签
            if self.label_mapper.is_match(predicted_primary_label_id, ground_truth_label):
                label_correct = 1.0
                # 主标签位置得分
                best_rank_score = 1.0
            # 2. 检查预测副标签是否匹配真实主标签
            elif isinstance(secondary_label_ids, list):
                for i, label_id in enumerate(secondary_label_ids):
                    if self.label_mapper.is_match(label_id, ground_truth_label):
                        label_correct = 1.0
                        # 副标签位置得分
                        current_score = position_scores['secondary'](i)
                        if current_score > best_rank_score:
                            best_rank_score = current_score
            # 3. 检查预测候选标签是否匹配真实主标签
            elif isinstance(candidates, list):
                for i, candidate in enumerate(candidates):
                    if isinstance(candidate, dict):
                        candidate_label_id = candidate.get('label_id', '').strip()
                        if self.label_mapper.is_match(candidate_label_id, ground_truth_label):
                            label_correct = 1.0
                            # 候选标签位置得分
                            current_score = position_scores['candidate'](i)
                            if current_score > best_rank_score:
                                best_rank_score = current_score
                    elif hasattr(candidate, 'label_id'):
                        candidate_label_id = str(getattr(candidate, 'label_id', '')).strip()
                        if self.label_mapper.is_match(candidate_label_id, ground_truth_label):
                            label_correct = 1.0
                            # 候选标签位置得分
                            current_score = position_scores['candidate'](i)
                            if current_score > best_rank_score:
                                best_rank_score = current_score
        
        # 如果真实主标签没有匹配，检查真实副标签的匹配情况
        if label_correct == 0.0 and ground_truth_secondary_labels and isinstance(ground_truth_secondary_labels, list):
            for gt_label in ground_truth_secondary_labels:
                # 1. 检查预测主标签是否匹配真实副标签
                if self.label_mapper.is_match(predicted_primary_label_id, gt_label):
                    label_correct = 0.8
                    # 主标签位置得分
                    best_rank_score = 1.0
                    break
                # 2. 检查预测副标签是否匹配真实副标签
                elif isinstance(secondary_label_ids, list):
                    for i, label_id in enumerate(secondary_label_ids):
                        if self.label_mapper.is_match(label_id, gt_label):
                            label_correct = 0.8
                            # 副标签位置得分
                            current_score = position_scores['secondary'](i)
                            if current_score > best_rank_score:
                                best_rank_score = current_score
                            break
                    if label_correct > 0.0:
                        break
                # 3. 检查预测候选标签是否匹配真实副标签
                elif isinstance(candidates, list):
                    for i, candidate in enumerate(candidates):
                        if isinstance(candidate, dict):
                            candidate_label_id = candidate.get('label_id', '').strip()
                            if self.label_mapper.is_match(candidate_label_id, gt_label):
                                label_correct = 0.8
                                # 候选标签位置得分
                                current_score = position_scores['candidate'](i)
                                if current_score > best_rank_score:
                                    best_rank_score = current_score
                                break
                        elif hasattr(candidate, 'label_id'):
                            candidate_label_id = str(getattr(candidate, 'label_id', '')).strip()
                            if self.label_mapper.is_match(candidate_label_id, gt_label):
                                label_correct = 0.8
                                # 候选标签位置得分
                                current_score = position_scores['candidate'](i)
                                if current_score > best_rank_score:
                                    best_rank_score = current_score
                                break
                    if label_correct > 0.0:
                        break
        
        # 计算分类准确率
        classification_accuracy = domain_correct * label_correct
        
        # 使用之前计算的最佳排名分数（位置权重）
        rank_score = best_rank_score
        
        # 计算综合分数
        # 公式：0.7 * Classification Accuracy + 0.3 * rankscore
        weighted_score = 0.7 * classification_accuracy + 0.3 * rank_score
        
        metrics["classification_accuracy"] = classification_accuracy
        metrics["rank_score"] = rank_score
        metrics["weighted_score"] = weighted_score
        metrics["domain_correct"] = domain_correct
        metrics["label_correct"] = label_correct
        
        # 添加上下文信息
        metrics["predicted_domain"] = predicted_domain_name
        metrics["predicted_primary_label"] = predicted_primary_label_name
        metrics["ground_truth_domain"] = ground_truth_domain
        metrics["ground_truth_label"] = ground_truth_label
        
        # 添加完整的分类结果信息
        if secondary_label_ids:
            metrics["secondary_labels"] = self.label_mapper.map_labels(secondary_label_ids)
        
        if candidates:
            mapped_candidates = []
            for candidate in candidates:
                if isinstance(candidate, dict):
                    mapped_candidate = {
                        "domain": self.label_mapper.get_domain_name(candidate.get('domain_id', '')) or candidate.get('domain_id', ''),
                        "label": self.label_mapper.get_label_name(candidate.get('label_id', '')) or candidate.get('label_id', ''),
                        "score": candidate.get('score', 0.0)
                    }
                    mapped_candidates.append(mapped_candidate)
            if mapped_candidates:
                metrics["candidates"] = mapped_candidates
        
        return metrics


# 全局实例
label_mapper = LabelMapper()
classification_evaluator = ClassificationEvaluator()


def evaluate_classification(
    predicted: Dict[str, Any],
    ground_truth_domain: str,
    ground_truth_label: str,
    ground_truth_secondary_labels: List[str] = None,
    ground_truth_secondary_domains: List[str] = None
) -> Dict[str, float]:
    """评估分类结果"""
    return classification_evaluator.evaluate_classification(
        predicted, 
        ground_truth_domain, 
        ground_truth_label,
        ground_truth_secondary_labels,
        ground_truth_secondary_domains
    )


def get_label_name(label_id: str) -> Optional[str]:
    """根据label_id获取中文名称"""
    return label_mapper.get_label_name(label_id)


def get_domain_name(domain_id: str) -> Optional[str]:
    """根据domain_id获取中文名称"""
    return label_mapper.get_domain_name(domain_id)

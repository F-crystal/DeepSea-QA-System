# -*- coding: utf-8 -*-
"""
eval_config.py

作者：Accilia
创建时间：2026-02-27
用途说明：
评估系统配置
"""

from __future__ import annotations

from typing import List, Optional

from deepsea_qa.configs import paths


class EvalConfig:
    """评估系统配置"""
    
    def __init__(self):
        # 数据集配置
        self.dataset_path: str = str(paths.QA_DATASET_PATH)
        
        # 输出配置
        self.output_dir: str = str(paths.EVAL_OUTPUT_DIR)
        self.logs_dir: str = str(paths.EVAL_LOGS_DIR)
        
        # 评估指标配置
        self.thelma_metrics: bool = True
        self.classification_metrics: bool = True
        self.timing_metrics: bool = True
        self.answer_quality_metrics: bool = True  # 回答质量指标（BLEU、精确率、召回率、BERTScore等）
        
        # THELMA指标配置
        self.sp_variants: List[str] = ["sp1", "sp2"]
        
        # LLM配置
        self.llm_provider: str = "zhipu"
        self.llm_model: str = "glm-4-plus"
        
        # 服务器配置
        self.server_url: str = "http://localhost:8000"

# -*- coding: utf-8 -*-
"""
generation_config.py

作者：Accilia
创建时间：2026-02-26
用途说明：
生成阶段的默认参数配置
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """生成阶段配置"""
    # 最大证据数量
    max_evidence: int = 6
    # 最大重试次数（失败后触发二次检索）
    max_retries: int = 3
    # 最终返回的 top N 结果数
    final_top_n: int = 10
    # LLM 参数
    temperature_draft: float = 0.3  # 草稿答案的温度参数
    temperature_final: float = 0.2  # 最终答案的温度参数
    max_tokens: int = 1200  # 最大生成长度
    # 验证参数
    quote_match_threshold: float = 0.75  # 引用匹配阈值
    # 反向验证
    enable_reverse_verification: bool = True

# -*- coding: utf-8 -*-
"""
query_config.py

作者：Accilia
创建时间：2026-02-25
用途说明：
  查询阶段默认参数配置（集中管理，便于调参、消融实验）。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


# -------------------------
# 分类器配置
# -------------------------
@dataclass
class ClassifierConfig:
    strategy: Literal["rules", "llm", "hybrid"] = "hybrid"
    max_secondary: int = 2
    llm_temperature: float = 0.0
    top_k: int = 5                      # ★跨方向 Top-K
    use_negative_keywords: bool = True  # ★用 negative_keywords 降权（更抗误判）


# -------------------------
# 重写扩展配置
# -------------------------
@dataclass
class RewriteExpandConfig:
    max_rewrites: int = 4
    max_expands: int = 6
    temperature: float = 0.4
    strategy: Literal["llm", "rules", "hybrid"] = "hybrid"


# -------------------------
# 查询管线配置
# -------------------------
@dataclass
class QueryPipelineConfig:
    max_sparse_queries: int = 8
    enable_classification: bool = True
    enable_rewrite_expand: bool = True


# -------------------------
# 查询阶段总配置
# -------------------------
@dataclass
class QueryConfig:
    classifier: ClassifierConfig = ClassifierConfig()
    rewrite_expand: RewriteExpandConfig = RewriteExpandConfig()
    pipeline: QueryPipelineConfig = QueryPipelineConfig()

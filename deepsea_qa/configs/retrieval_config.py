# -*- coding: utf-8 -*-
"""
retrieval_config.py

作者：Accilia
创建时间：2026-02-24
用途说明：
  检索阶段默认参数配置（集中管理，便于调参、消融实验）。
"""

from __future__ import annotations
from dataclasses import dataclass


# -------------------------
# 稀疏检索（BM25）
# -------------------------
@dataclass
class SparseConfig:
    top_k: int = 200
    # 多 query（queries_sparse）合并策略
    # - "sum": 直接把多个 query 的 bm25 score 累加（快）
    # - "rrf": 多列表用 RRF 融合（更稳，但需要 rank）
    merge: str = "sum"


# -------------------------
# 稠密检索（FAISS）
# -------------------------
@dataclass
class DenseConfig:
    top_k: int = 200
    batch_size: int = 32  # embedding 批大小


# -------------------------
# 融合（BM25 + Dense）
# -------------------------
@dataclass
class FusionConfig:
    # - "rrf": 推荐默认（不依赖分数尺度）
    # - "linear": 论文友好：alpha*bm25 + (1-alpha)*cosine
    strategy: str = "rrf"
    alpha: float = 0.5                 # 仅 linear 用
    normalize: str = "minmax"          # 仅 linear 用：minmax / zscore / none


# -------------------------
# 分类软加权（soft boost）
# -------------------------
@dataclass
class ClassificationBoostConfig:
    enable: bool = True

    # boost 作用对象：
    # - "domain": 仅按 domain_id boost
    # - "label": 需要 chunk 有 label_id（如果 sqlite 没有，先别开）
    mode: str = "domain"

    # 使用 Top-K candidates 的 domain/label 做 boost
    use_top_k: int = 2

    # 权重：对匹配到的 chunk 进行 score *= (1 + boost_weight * cand_score)
    # cand_score 来自 classification.candidates[i].score（0-1）
    boost_weight: float = 0.25

    # 只在融合后应用（推荐）；也可以前置到 sparse/dense（先不做）
    apply_stage: str = "post_fusion"


# -------------------------
# 重排（Cross-Encoder / Reranker）
# -------------------------
@dataclass
class RerankConfig:
    enable: bool = True
    model_name: str = "bge-reranker-v2-m3"
    device: str = "cuda"
    fp16: bool = True

    # 从融合结果中取 TopM 做 rerank（控制成本）
    top_m: int = 100

    # rerank 批大小（关键：batch 推理加速）
    batch_size: int = 16

    # tokenizer 截断长度（根据模型和你的 chunk 长度调整）
    max_length: int = 512

    # 最终返回 TopN
    final_top_n: int = 10
    
    # 最少返回结果数
    min_final_top_n: int = 3
    
    # 动态截断阈值：保留得分在top1-delta区间内的结果
    delta: float = 0.15


# -------------------------
# 评估接口（hook）
# -------------------------
@dataclass
class EvalConfig:
    enable: bool = False
    # 预留：后续可接入你的 qrels/人工标注集
    # 当前实现只提供 hook，不强制你立刻建评测集
    record_debug: bool = True


# -------------------------
# 总配置
# -------------------------
@dataclass
class RetrievalConfig:
    sparse: SparseConfig = SparseConfig()
    dense: DenseConfig = DenseConfig()
    fusion: FusionConfig = FusionConfig()
    boost: ClassificationBoostConfig = ClassificationBoostConfig()
    rerank: RerankConfig = RerankConfig()
    eval: EvalConfig = EvalConfig()
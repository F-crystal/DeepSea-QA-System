# -*- coding: utf-8 -*-
"""
index_config.py

作者：Accilia
创建时间：2026-02-25
用途说明：
  索引构建阶段默认参数配置（集中管理，便于调参、消融实验）。
"""

from __future__ import annotations
from dataclasses import dataclass


# -------------------------
# BM25 索引配置
# -------------------------
@dataclass
class BM25Config:
    # BM25 核心参数
    k1: float = 1.5      # 词频饱和参数
    b: float = 0.75      # 文档长度归一化参数
    
    # 构建相关
    min_doc_length: int = 1  # 最小文档长度（词数）
    min_token_len: int = 2   # 最小词长
    max_tokens_per_doc: int = 4096  # 每文档最大词数


# -------------------------
# FAISS 索引配置
# -------------------------
@dataclass
class FAISSConfig:
    model_name: str = "bge-m3"  # 嵌入模型名称
    batch_size: int = 64        # 编码批大小
    normalize: bool = True      # 是否归一化嵌入向量（归一化后内积=余弦相似度）
    max_length: int = 512       # 分词器最大序列长度
    device: str = "cuda"        # 设备："cuda" / "cpu"
    
    # 索引类型
    index_type: str = "IndexFlatIP"  # 内积索引（适合归一化向量）


# -------------------------
# 知识库构建配置
# -------------------------
@dataclass
class KBConfig:
    # SQLite 相关
    table_name: str = "chunks"        # 表名
    chunk_id_col: str = "chunk_id"    # chunk_id 列名
    text_col: str = "text"            # 文本列名
    
    # 语义单元切分相关（与build_kb.py一致）
    chunk_size: int = 256     # 切分大小（词数）
    chunk_overlap: int = 40   # 重叠大小（词数）


# -------------------------
# 索引构建总配置
# -------------------------
@dataclass
class IndexConfig:
    bm25: BM25Config = BM25Config()
    faiss: FAISSConfig = FAISSConfig()
    kb: KBConfig = KBConfig()

# -*- coding: utf-8 -*-
"""
paths.py

作者: 冯冉
创建时间：2026-02-22
用途说明: 统一管理数据输入/输出路径。
"""

from __future__ import annotations

from pathlib import Path


# ======================================================
# 一、根目录结构
# ======================================================

# deepsea_qa 项目根目录
DEEPSEA_QA_ROOT = Path(__file__).resolve().parents[1]

# 工作区根目录（prepare_data 与 deepsea_qa 同级）
WORKSPACE_ROOT = DEEPSEA_QA_ROOT.parent

# 预处理数据根目录
PREPARE_DATA_ROOT = WORKSPACE_ROOT / "prepare_data"

# ======================================================
# 二、预处理数据目录
# ======================================================

DIR_CORPUS = PREPARE_DATA_ROOT / "题录信息_中间结果"
DIR_LABEL_CARDS = PREPARE_DATA_ROOT / "分类标签"

# 中文停用词目录
STOPWORDS_DIR = PREPARE_DATA_ROOT / "stopwords"


# ======================================================
# 三、知识库输出目录
# ======================================================

ARTIFACTS_ROOT = DEEPSEA_QA_ROOT / "artifacts"
KB_ROOT = ARTIFACTS_ROOT / "kb"

# ---------- 语义单元切分 ----------
CHUNKS_DIR = KB_ROOT / "chunks"
CHUNKS_ALL_JSONL = CHUNKS_DIR / "chunks_all.jsonl"

# ---------- SQLite 文本+元数据 ----------
STORE_DIR = KB_ROOT / "store"
SQLITE_PATH = STORE_DIR / "chunks.sqlite"

# ---------- FAISS 向量索引 ----------
FAISS_DIR = KB_ROOT / "faiss"
FAISS_INDEX_PATH = FAISS_DIR / "index_bge-m3_ip.faiss"
FAISS_META_PATH = FAISS_DIR / "index_bge-m3_ip.meta.json"
FAISS_IDMAP_PATH = FAISS_DIR / "index_bge-m3_ip.id_map.jsonl"

# ---------- BM25 稀疏索引 ----------
BM25_DIR = KB_ROOT / "bm25"
BM25_PKL_PATH = BM25_DIR / "bm25.pkl"
BM25_INDEX_PATH = BM25_PKL_PATH
BM25_TOKENIZER_META_PATH = BM25_DIR / "tokenizer_meta.json"

# ---------- Query ----------
QUERY_ARTIFACTS_DIR = ARTIFACTS_ROOT / "query"
QUERY_LOG_DIR = QUERY_ARTIFACTS_DIR / "logs"
QUERY_DEBUG_DIR = QUERY_ARTIFACTS_DIR / "debug"

# ---------- Retriever ----------
RETRIEVAL_ROOT = ARTIFACTS_ROOT / "retrieval"
RETRIEVAL_DEBUG_DIR = RETRIEVAL_ROOT / "debug"

# ---------- Generator ----------
GEN_ROOT = ARTIFACTS_ROOT / "generation"
GEN_DEBUG_DIR = GEN_ROOT / "debug"
GEN_OUTPUT_DIR = GEN_ROOT / "outputs"

# ---------- QA ----------
QA_ROOT = ARTIFACTS_ROOT / "qa"
QA_DEBUG_DIR = QA_ROOT / "debug"

# ---------- Evaluation ----------
EVAL_ROOT = ARTIFACTS_ROOT / "eval"
EVAL_OUTPUT_DIR = EVAL_ROOT / "outputs"
EVAL_LOGS_DIR = EVAL_ROOT / "logs"

# 评估数据集路径
QA_DATASET_PATH = WORKSPACE_ROOT / "qa_post" / "qa_sampled_dataset.jsonl"

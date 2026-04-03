# -*- coding: utf-8 -*-
"""
pipeline.py

作者：Accilia
创建时间：2026-02-23
用途说明：
  QueryPipeline：将 “分类 + rewrite/expand” 组合为一个可复用的管线。
  输出 QueryBundle（queries集合 + 分类元数据）。

设计要点：
  1) 依赖注入：可传 llm 实例，或用 llm_settings 由 registry 构建（便于对比实验）
  2) 可消融：enable_classification / enable_rewrite_expand
  3) 下游友好：返回 QueryBundle，后续检索/生成模块可直接复用其中字段
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any

from deepsea_qa.query.types import QueryBundle, QueryVariants, QueryClassification
from deepsea_qa.query.label_cards import LabelCardsStore
from deepsea_qa.query.classifier import QueryClassifier
from deepsea_qa.query.rewrite_expand import QueryRewriteExpander

from deepsea_qa.configs.query_config import ClassifierConfig, RewriteExpandConfig, QueryPipelineConfig

from deepsea_qa.llm.base import BaseLLM
from deepsea_qa.llm.registry import build_llm, LLMSettings


class QueryPipeline:
    def __init__(
        self,
        store: Optional[LabelCardsStore] = None,
        llm: Optional[BaseLLM] = None,
        llm_settings: Optional[LLMSettings] = None,
        cls_cfg: Optional[ClassifierConfig] = None,
        re_cfg: Optional[RewriteExpandConfig] = None,
        pipe_cfg: Optional[QueryPipelineConfig] = None,
    ):
        self.store = store or LabelCardsStore()
        self.store.load_all()

        self.llm = llm or build_llm(llm_settings or LLMSettings())

        self.classifier = QueryClassifier(self.store, self.llm, cls_cfg or ClassifierConfig())
        self.rewriter = QueryRewriteExpander(self.store, self.llm, re_cfg or RewriteExpandConfig())
        self.cfg = pipe_cfg or QueryPipelineConfig()

    async def process(self, query: str) -> QueryBundle:
        query = (query or "").strip()

        # 1) 分类（可消融）
        async def _do_cls():
            if not self.cfg.enable_classification:
                return QueryClassification(
                    domain_id="UNKNOWN",
                    domain_name_zh="UNKNOWN",
                    primary_label_id="UNKNOWN_OTHER",
                    secondary_label_ids=[],
                    confidence=0.0,
                    evidence_spans=[],
                    method="rules",
                )
            return await asyncio.to_thread(self.classifier.classify, query)

        classification = await _do_cls()

        # 2) rewrite/expand（可消融）
        async def _do_rewrite():
            if not self.cfg.enable_rewrite_expand:
                return QueryVariants(original=query, rewrites=[], expands=[])
            return await asyncio.to_thread(self.rewriter.run, query, classification)

        variants = await _do_rewrite()

        queries_sparse = variants.unique_sparse_queries(max_n=self.cfg.max_sparse_queries)
        queries_dense = variants.dense_queries()  # 默认 original

        meta: Dict[str, Any] = {
            "note": "classification用于稀疏+稠密检索过滤；rewrite/expand默认仅用于稀疏检索",
            "llm": {
                "provider": getattr(self.llm, "__class__", type(self.llm)).__name__,
                "model": getattr(self.llm, "model", None),
            }
        }

        return QueryBundle(
            variants=variants,
            classification=classification,  # ★包含 candidates
            queries_sparse=queries_sparse,
            queries_dense=queries_dense,
            meta=meta,
        )
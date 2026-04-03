# -*- coding: utf-8 -*-
"""
api.py

作者：Accilia
创建时间：2026-02-23
用途说明：
  提供 Query 阶段的统一“可复用API”，供后续检索/生成/服务端调用.
  
特点：
  - 异步：支持 asyncio，推荐在服务端使用
  - 同步：也提供同步版，给检索/生成模块直接用
"""

from __future__ import annotations

from deepsea_qa.query.pipeline import QueryPipeline
from deepsea_qa.configs.query_config import QueryPipelineConfig, ClassifierConfig, RewriteExpandConfig
from deepsea_qa.llm.registry import LLMSettings


async def build_query_bundle(
    query: str,
    llm_provider: str = "zhipu",
    llm_model: str = "glm-4-plus",
    enable_classification: bool = True,
    enable_rewrite_expand: bool = True,
    max_sparse_queries: int = 8,
    cls_strategy: str = "llm",  # 默认使用LLM分类
) -> dict:
    """
    异步版（推荐）：返回 dict（便于序列化、落盘、RPC传输）
    """
    pipe = QueryPipeline(
        llm_settings=LLMSettings(provider=llm_provider, model=llm_model),
        cls_cfg=ClassifierConfig(strategy=cls_strategy),
        re_cfg=RewriteExpandConfig(strategy="hybrid"),
        pipe_cfg=QueryPipelineConfig(
            max_sparse_queries=max_sparse_queries,
            enable_classification=enable_classification,
            enable_rewrite_expand=enable_rewrite_expand,
        ),
    )
    bundle = await pipe.process(query)
    return bundle.to_dict()


def build_query_bundle_sync(
    query: str,
    llm_provider: str = "zhipu",
    llm_model: str = "glm-4-plus",
    enable_classification: bool = True,
    enable_rewrite_expand: bool = True,
    max_sparse_queries: int = 8,
    cls_strategy: str = "llm",  # 默认使用LLM分类
) -> dict:
    """
    同步版：给后续检索/生成模块直接用（不需要 async）
    """
    import asyncio
    return asyncio.run(
        build_query_bundle(
            query=query,
            llm_provider=llm_provider,
            llm_model=llm_model,
            enable_classification=enable_classification,
            enable_rewrite_expand=enable_rewrite_expand,
            max_sparse_queries=max_sparse_queries,
            cls_strategy=cls_strategy,
        )
    )
# -*- coding: utf-8 -*-
"""
generation/api.py

作者：Accilia
创建时间：2026-02-25
用途说明：
- generate_answer_sync：同步返回最终 GenerationResult(dict)
- stream_generation_events：同步事件流（证据卡片 + LLM输出增量）
"""

from __future__ import annotations

from typing import Dict, Any, Generator, Optional

from deepsea_qa.llm.registry import LLMSettings
from deepsea_qa.generation.pipeline import GenerationPipeline
from deepsea_qa.configs.generation_config import GenerationConfig


def stream_generation_events(
    query: str,
    retrieval_result: Dict[str, Any],
    llm_provider: str = "zhipu",
    llm_model: str = "glm-4-plus",
    max_evidence: int = 6,
) -> Generator[Dict[str, Any], None, None]:
    pipe = GenerationPipeline(
        llm_settings=LLMSettings(provider=llm_provider, model=llm_model),
        cfg=GenerationConfig(max_evidence=max_evidence),
    )
    for ev in pipe.stream_answer_events(query=query, retrieval_result=retrieval_result):
        yield ev


def generate_answer_sync(
    query: str,
    query_bundle: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    llm_provider: str = "zhipu",
    llm_model: str = "glm-4-plus",
    max_evidence: int = 6,
    max_retries: int = 1,
    final_top_n: int = 10,
    enable_reverse_verification: bool = True,
    enable_rerank: bool = True,
) -> Dict[str, Any]:
    pipe = GenerationPipeline(
        llm_settings=LLMSettings(provider=llm_provider, model=llm_model),
        cfg=GenerationConfig(
            max_evidence=max_evidence,
            max_retries=max_retries,
            final_top_n=final_top_n,
            enable_reverse_verification=enable_reverse_verification,
        ),
        enable_rerank=enable_rerank,
    )
    res = pipe.run(query=query, query_bundle=query_bundle, retrieval_result=retrieval_result)
    return res.to_dict()
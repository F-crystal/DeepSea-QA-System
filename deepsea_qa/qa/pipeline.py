# -*- coding: utf-8 -*-
"""
qa/pipeline.py

端到端问答助手 Pipeline：
query -> query_bundle -> retrieval_result -> constrained generation -> verification -> retry

说明：
- 检索在服务器（OpenBayes）完成；
- 生成在本地（调用智谱 glm-4-plus）完成；
- 反向验证失败会触发“二次检索+重答”（闭环自纠错）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator

from deepsea_qa.query.api import build_query_bundle_sync
from deepsea_qa.retrieval.api import retrieve_chunks_sync
from deepsea_qa.generation.api import generate_answer_sync, stream_generation_events


@dataclass
class QAPipelineConfig:
    # query
    enable_classification: bool = True
    enable_rewrite_expand: bool = True
    max_sparse_queries: int = 8

    # retrieval
    final_top_n: int = 10
    return_debug: bool = False
    enable_rerank: bool = True

    # generation
    max_evidence: int = 6
    max_retries: int = 1
    enable_reverse_verification: bool = True


class QAPipeline:
    def __init__(
        self,
        llm_provider: str = "zhipu",
        llm_model: str = "glm-4-plus",
        cfg: QAPipelineConfig | None = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.cfg = cfg or QAPipelineConfig()

    def _build_query_bundle(self, query: str) -> Dict[str, Any]:
        return build_query_bundle_sync(
            query=query,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            enable_classification=self.cfg.enable_classification,
            enable_rewrite_expand=self.cfg.enable_rewrite_expand,
            max_sparse_queries=self.cfg.max_sparse_queries,
            cls_strategy="llm",  # 强制使用LLM分类
        )

    def _retrieve(self, query_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return retrieve_chunks_sync(
            query_bundle=query_bundle,
            final_top_n=self.cfg.final_top_n,
            return_evidence=True,      # 生成阶段需要 evidence
            return_debug=self.cfg.return_debug,
            enable_rerank=self.cfg.enable_rerank,
            timeout=300,
        )

    def _generate_final(
        self,
        query: str,
        query_bundle: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        draft_answer: str = "",
    ) -> Dict[str, Any]:
        return generate_answer_sync(
            query=query,
            query_bundle=query_bundle,
            retrieval_result=retrieval_result,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            max_evidence=self.cfg.max_evidence,
            max_retries=self.cfg.max_retries,
            final_top_n=self.cfg.final_top_n,
            enable_reverse_verification=self.cfg.enable_reverse_verification,
            enable_rerank=self.cfg.enable_rerank,
            draft_answer=draft_answer,
        )

    def run(self, query: str) -> Dict[str, Any]:
        qb = self._build_query_bundle(query)
        rr = self._retrieve(qb)
        out = self._generate_final(query, qb, rr)

        return {
            "query": query,
            "query_bundle": qb,
            "retrieval_result": rr,
            "generation": out,
        }

    def stream(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """
        事件流（同步 generator）：
          - 先构建 query_bundle + retrieval_result
          - 然后产出 generation 的 evidence/answer_delta 事件
          - 最后产出 final（完整结构化结果）
        """
        qb = self._build_query_bundle(query)
        rr = self._retrieve(qb)
        draft_parts = []

        # streaming：证据卡片 + LLM 增量输出
        for ev in stream_generation_events(
            query=query,
            retrieval_result=rr,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            max_evidence=self.cfg.max_evidence,
        ):
            if ev.get("event") == "answer_delta":
                draft_parts.append(str(ev.get("delta", "")))
            yield ev

        # 最终结构化答案（含验证/引用/必要时二检重答）
        final = self._generate_final(query, qb, rr, draft_answer="".join(draft_parts))
        yield {"event": "final", "data": final}

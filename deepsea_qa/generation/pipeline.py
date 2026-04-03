# -*- coding: utf-8 -*-
"""
pipeline.py

作者：Accilia
创建时间：2026-02-25
用途说明：
GenerationPipeline：约束生成 + 反向验证 + 失败触发二次检索闭环

输入：
- query_bundle（用于二次检索）
- retrieval_result（第一次检索结果，含 evidence）

输出：
- GenerationResult（含 verify_report）

注意：
- “验证控制”由 verifier.py 独立模块实现，便于作为创新点单独描述与消融。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from deepsea_qa.generation.types import EvidenceItem, GenerationResult, CitationItem
from deepsea_qa.generation.meta_resolver import ExcelMetaResolver
from deepsea_qa.generation.gbt7714 import format_gbt_7714
from deepsea_qa.generation.answerer import stream_draft_answer, generate_structured_answer_json
from deepsea_qa.generation.verifier import verify_answer
from deepsea_qa.llm.registry import LLMSettings, build_llm
from deepsea_qa.configs.generation_config import GenerationConfig

from deepsea_qa.retrieval.api import retrieve_chunks_sync


class GenerationPipeline:
    def __init__(
        self,
        llm_settings: LLMSettings,
        cfg: Optional[GenerationConfig] = None,
        meta_resolver: Optional[ExcelMetaResolver] = None,
        enable_rerank: bool = True,
    ):
        self.llm_settings = llm_settings
        self.cfg = cfg or GenerationConfig()
        self.meta_resolver = meta_resolver or ExcelMetaResolver()
        self.llm_client = build_llm(llm_settings)
        self.enable_rerank = enable_rerank

    def _parse_evidences(self, retrieval_result: Dict[str, Any]) -> List[EvidenceItem]:
        out = []
        for e in (retrieval_result.get("evidence", []) or []):
            out.append(EvidenceItem(
                chunk_id=str(e.get("chunk_id", "")).strip(),
                paper_id=str(e.get("paper_id", "")).strip(),
                text=str(e.get("text", "")).strip(),
                domain=str(e.get("domain", "")).strip(),
                source_xlsx=str(e.get("source_xlsx", "")).strip(),
            ))
        return out
    
    def _filter_relevant_evidences(self, query: str, evidences: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        筛选与查询相关的证据
        
        Args:
            query: 用户查询
            evidences: 证据列表
            
        Returns:
            筛选后的相关证据列表
        """
        from deepsea_qa.eval.thelma.modules.match import MatchModule
        
        match_module = MatchModule(self.llm_client)
        relevant_evidences = []
        
        for evidence in evidences:
            # 检查证据与查询的相关性
            is_relevant = match_module.judge_essentiality(query, evidence.text)
            if is_relevant:
                relevant_evidences.append(evidence)
        
        # 如果没有相关证据，返回原始证据（避免空列表）
        return relevant_evidences if relevant_evidences else evidences

    def _attach_refs(self, ans_json: Dict[str, Any], meta_map: Dict[str, Any]) -> List[str]:
        """
        给 citations 生成 ref_gbt，并返回去重 refs 列表
        """
        refs = []
        citations = ans_json.get("citations", []) or []
        for c in citations:
            pid = str(c.get("paper_id", "")).strip()
            m = meta_map.get(pid)
            ref = format_gbt_7714(m) if m else ""
            c["ref_gbt"] = ref
            if ref:
                refs.append(ref)

        # 去重保持顺序
        seen = set()
        uniq = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq

    # ---------- streaming：用于“展示证据+生成过程” ----------
    def stream_answer_events(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
    ):
        """
        事件流（同步 generator）：
          - {"event":"evidence", ...}
          - {"event":"answer_delta", "delta": "..."}
          - {"event":"done"}（草稿结束）
        """
        evidences = self._parse_evidences(retrieval_result)
        meta_map = self.meta_resolver.resolve(evidences)

        # 先把 evidence 卡片吐出去（可解释性展示，不是思维链）
        for i, e in enumerate(evidences[: self.cfg.max_evidence], 1):
            m = meta_map.get(e.paper_id)
            yield {
                "event": "evidence",
                "idx": i,
                "chunk_id": e.chunk_id,
                "paper_id": e.paper_id,
                "title": (m.title if m else ""),
                "venue": (m.venue if m else ""),
                "year": (m.year if m else None),
                "snippet": (e.text[:500] + "…") if len(e.text) > 500 else e.text,
            }

        # 再流式输出草稿答案
        for delta in stream_draft_answer(
            query=query,
            evidences=evidences,
            meta_map=meta_map,
            llm_settings=self.llm_settings,
            max_evidence=self.cfg.max_evidence,
            cfg=self.cfg,
        ):
            yield {"event": "answer_delta", "delta": delta}

        yield {"event": "done"}

    # ---------- 最终：结构化答案 + 闭环验证 + 必要时二检 ----------
    def run(
        self,
        query: str,
        query_bundle: Dict[str, Any],
        retrieval_result: Dict[str, Any],
    ) -> GenerationResult:
        cur_rr = retrieval_result
        last_ans_json: Dict[str, Any] = {}
        last_report: Dict[str, Any] = {}

        for attempt in range(self.cfg.max_retries + 1):
            evidences = self._parse_evidences(cur_rr)
            # 筛选与查询相关的证据
            relevant_evidences = self._filter_relevant_evidences(query, evidences)
            meta_map = self.meta_resolver.resolve(relevant_evidences)

            # 1) 生成严格 JSON（可验证）
            ans_json = generate_structured_answer_json(
                query=query,
                evidences=relevant_evidences,
                meta_map=meta_map,
                llm_settings=self.llm_settings,
                max_evidence=self.cfg.max_evidence,
                cfg=self.cfg,
            )

            # 2) refs（GB/T）
            refs_gbt = self._attach_refs(ans_json, meta_map)
            ans_json["refs_gbt"] = refs_gbt

            # 3) 验证（独立模块：创新点闭环）
            last_ans_json = ans_json
            
            if self.cfg.enable_reverse_verification:
                report = verify_answer(ans_json, evidences, cfg=self.cfg, llm_client=self.llm_client)
                last_report = report
                
                if report.get("verified", False):
                    break

                # 4) 未通过：触发二次检索再生成
                if attempt < self.cfg.max_retries:
                    cur_rr = retrieve_chunks_sync(
                        query_bundle=query_bundle,
                        final_top_n=self.cfg.final_top_n,
                        return_evidence=True,     # 生成阶段需要 evidence
                        return_debug=False,
                        enable_rerank=self.enable_rerank,
                        timeout=300,
                    )
                    continue
            else:
                # 禁用反向验证，直接使用第一次生成的结果
                last_report = {"verified": True, "reason": "Reverse verification disabled"}
                break

        # 输出结构化 result
        citations_out: List[CitationItem] = []
        for c in (last_ans_json.get("citations", []) or []):
            citations_out.append(CitationItem(
                paper_id=str(c.get("paper_id", "")),
                chunk_id=str(c.get("chunk_id", "")),
                quote=str(c.get("quote", "")),
                ref_gbt=str(c.get("ref_gbt", "")),
                ok=None,
                reason="",
            ))

        return GenerationResult(
            query=query,
            answer=str(last_ans_json.get("answer", "")).strip(),
            citations=citations_out,
            refs_gbt=list(last_ans_json.get("refs_gbt", []) or []),
            verified=bool(last_report.get("verified", False)),
            verify_report=last_report,
            raw=last_ans_json,
        )
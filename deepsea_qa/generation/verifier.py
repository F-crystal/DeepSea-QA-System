# -*- coding: utf-8 -*-
"""
verifier.py

作者：Accilia
创建时间：2026-02-25
用途说明：
“反向验证 + 自纠错触发”控制模块：
输入：LLM 结构化输出（citations里包含 chunk_id + quote）
对每条引用执行：
  - chunk_id 是否属于原始 evidence 列表？
  - quote 是否能在对应 evidence.text 中找到（严格子串 + 近似兜底）
  - 使用大模型分解内容检验：问题与来源对应性、来源与回答对应性

输出：verify_report（包含是否通过 + 失败原因）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import difflib

from deepsea_qa.generation.types import EvidenceItem
from deepsea_qa.configs.generation_config import GenerationConfig
from deepsea_qa.llm.base import BaseLLM
from deepsea_qa.eval.thelma.modules.decompose import DecomposeModule
from deepsea_qa.eval.thelma.modules.match import MatchModule


def _quote_match(text: str, quote: str, threshold: float = 0.75) -> bool:
    if not quote:
        return False
    t = text or ""
    q = quote.strip()

    # 1) 严格子串
    if q in t:
        return True

    # 2) 近似匹配兜底：避免空格/标点差异导致误判
    # 注意：阈值别太低，否则会误过；也别太高，否则会误判失败
    r = difflib.SequenceMatcher(None, t, q).ratio()
    return r >= threshold


def verify_answer(answer_json: Dict[str, Any], evidences: List[EvidenceItem], cfg: Optional[GenerationConfig] = None, llm_client: Optional[BaseLLM] = None) -> Dict[str, Any]:
    cfg = cfg or GenerationConfig()
    ev_map = {e.chunk_id: e for e in evidences}
    citations = answer_json.get("citations", []) or []
    answer = answer_json.get("answer", "")
    query = answer_json.get("query", "")

    details = []
    ok_all = True

    # 基础验证：chunk_id 和 quote 匹配
    for c in citations:
        chunk_id = str(c.get("chunk_id", "")).strip()
        quote = str(c.get("quote", "")).strip()

        if not chunk_id or chunk_id not in ev_map:
            ok_all = False
            details.append({
                "chunk_id": chunk_id,
                "ok": False,
                "reason": "chunk_id_not_in_original_evidence",
            })
            continue

        e = ev_map[chunk_id]
        if not _quote_match(e.text, quote, threshold=cfg.quote_match_threshold):
            ok_all = False
            details.append({
                "chunk_id": chunk_id,
                "ok": False,
                "reason": "quote_not_found_in_evidence_text",
                "quote_head": quote[:200],
            })
            continue

        details.append({
            "chunk_id": chunk_id,
            "ok": True,
            "reason": "verified_pass",
        })

    # 高级验证：使用大模型进行分解检验
    if llm_client and query and answer:
        decompose_module = DecomposeModule(llm_client)
        match_module = MatchModule(llm_client)

        # 1. 分解回答为原子事实
        answer_claims = decompose_module.decompose_text(answer)
        
        # 2. 分解查询为子问题
        sub_questions = decompose_module.decompose_query(query)
        
        # 3. 检查每个原子事实是否被来源支持
        source_texts = [e.text for e in evidences]
        supported_claims = []
        for claim in answer_claims:
            is_supported = match_module.judge_support(claim, source_texts)
            supported_claims.append({
                "claim": claim,
                "supported": is_supported
            })
        
        # 4. 检查每个子问题是否在来源中找到答案
        source_coverage = []
        for sub_q in sub_questions:
            covered = False
            for source in source_texts:
                if match_module.contains_answer(sub_q, source):
                    covered = True
                    break
            source_coverage.append({
                "sub_question": sub_q,
                "covered": covered
            })
        
        # 5. 检查每个来源与查询的相关性
        source_relevance = []
        for i, source in enumerate(source_texts):
            is_relevant = match_module.judge_essentiality(query, source)
            source_relevance.append({
                "source_idx": i,
                "relevant": is_relevant
            })
        
        # 6. 检查回答与查询主题的相关性
        answer_relevance = match_module.judge_essentiality(query, answer)
        
        # 7. 检查每个原子事实与查询主题的相关性
        claim_relevance = []
        for claim in answer_claims:
            is_relevant = match_module.judge_essentiality(query, claim)
            claim_relevance.append({
                "claim": claim,
                "relevant": is_relevant
            })
        
        # 生成验证结果
        advanced_verification = {
            "answer_claims": supported_claims,
            "source_coverage": source_coverage,
            "source_relevance": source_relevance,
            "answer_relevance": answer_relevance,
            "claim_relevance": claim_relevance,
            "n_answer_claims": len(answer_claims),
            "n_sub_questions": len(sub_questions),
            "n_sources": len(source_texts)
        }
    else:
        advanced_verification = {
            "answer_claims": [],
            "source_coverage": [],
            "source_relevance": [],
            "answer_relevance": False,
            "claim_relevance": [],
            "n_answer_claims": 0,
            "n_sub_questions": 0,
            "n_sources": len(evidences)
        }

    # 增加主题相关性检查到验证通过条件
    if llm_client and query and answer:
        # 检查回答与查询主题的相关性
        if not advanced_verification.get("answer_relevance", False):
            ok_all = False
        
        # 检查原子事实与查询主题的相关性（至少80%的事实相关）
        claim_relevance = advanced_verification.get("claim_relevance", [])
        if claim_relevance:
            relevant_claims = sum(1 for item in claim_relevance if item.get("relevant", False))
            relevance_ratio = relevant_claims / len(claim_relevance)
            if relevance_ratio < 0.8:
                ok_all = False
        
        # 检查来源与查询的相关性（至少50%的来源相关）
        source_relevance = advanced_verification.get("source_relevance", [])
        if source_relevance:
            relevant_sources = sum(1 for item in source_relevance if item.get("relevant", False))
            relevance_ratio = relevant_sources / len(source_relevance)
            if relevance_ratio < 0.5:
                ok_all = False

    return {
        "verified": bool(ok_all),
        "n_citations": len(citations),
        "details": details,
        "advanced_verification": advanced_verification
    }
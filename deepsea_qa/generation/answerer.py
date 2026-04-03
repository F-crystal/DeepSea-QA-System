# -*- coding: utf-8 -*-
"""
answerer.py

作者：Accilia
创建时间：2026-02-25
用途说明：
生成阶段 LLM 调用：
- stream_draft_answer：提供 streaming 体验（输出草稿答案的增量文本）
- generate_structured_answer_json：生成可验证 JSON（供 verifier 闭环）

说明：
- streaming 不做 JSON（跨 provider 很难稳定 token-stream JSON）
- 最终 JSON 用 response_format=json_object 保证稳定
"""

from __future__ import annotations

from typing import Any, Dict, List, Generator, Optional
import json
import re

from deepsea_qa.generation.types import EvidenceItem, PaperMeta
from deepsea_qa.llm.registry import build_llm, LLMSettings
from deepsea_qa.configs.generation_config import GenerationConfig


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    t = _strip_json_fence(text)
    if not t:
        return {}
    try:
        return json.loads(t)
    except Exception:
        # 兜底：截取 {} 主体
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def _build_evidence_blocks(evidences: List[EvidenceItem], meta_map: Dict[str, PaperMeta], max_evidence: int) -> List[str]:
    blocks = []
    for i, e in enumerate(evidences[:max_evidence], 1):
        m = meta_map.get(e.paper_id)
        title = (m.title if m else "") or ""
        venue = (m.venue if m else "") or ""
        year = "" if (not m or m.year is None) else str(m.year)

        blocks.append(
            f"[E{i}] chunk_id={e.chunk_id}\n"
            f"paper_id={e.paper_id}\n"
            f"title={title}\n"
            f"venue={venue}\n"
            f"year={year}\n"
            f"text={e.text}\n"
        )
    return blocks


def stream_draft_answer(
    query: str,
    evidences: List[EvidenceItem],
    meta_map: Dict[str, PaperMeta],
    llm_settings: LLMSettings,
    max_evidence: int = 6,
    cfg: Optional[GenerationConfig] = None,
) -> Generator[str, None, None]:
    """
    流式输出草稿答案（增量文本）。
    - 不要求 JSON，仅用于"streaming体验"
    """
    cfg = cfg or GenerationConfig()
    llm = build_llm(llm_settings)
    blocks = _build_evidence_blocks(evidences, meta_map, max_evidence=max_evidence)

    system = (
        "你是深海科技问答助手。请严格基于给定 evidence 回答，不要编造。\n"
        "输出要求：\n"
        "1) 用要点/小标题回答。\n"
        "2) 回答不要过于简略。\n"
        "3) 不要输出 JSON，只输出可读答案草稿。\n"
    )

    user = json.dumps({"query": query, "evidence_blocks": blocks}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # 使用 provider 的 stream_chat；若 provider 不支持，会自动 fallback 切片流式
    for delta in llm.stream_chat(messages=messages, temperature=cfg.temperature_draft, max_tokens=cfg.max_tokens):
        # 处理delta，去掉空格、markdown的"**"、分段和制表符
        processed_delta = delta.replace('**', '').replace('\n', '').replace('\t', '').strip()
        yield processed_delta


def generate_structured_answer_json(
    query: str,
    evidences: List[EvidenceItem],
    meta_map: Dict[str, PaperMeta],
    llm_settings: LLMSettings,
    max_evidence: int = 6,
    cfg: Optional[GenerationConfig] = None,
) -> Dict[str, Any]:
    """
    生成严格 JSON（用于闭环验证）：
    citations: [{paper_id, chunk_id, quote}]
    quote 必须来自 evidence text 原文短摘（<=40词）
    """
    cfg = cfg or GenerationConfig()
    llm = build_llm(llm_settings)
    blocks = _build_evidence_blocks(evidences, meta_map, max_evidence=max_evidence)

    system = (
        "你是严格受证据约束的深海科技问答助手。\n"
        "必须仅依据我给出的 evidence 作答，不得编造证据外内容。\n"
        "必须输出 JSON（不要输出任何多余文字）。\n"
        "answer 字段要求：\n"
        "1) 回答不要过于简略，要涵盖用户问题的所有方面。\n"
        "2) 确保包含用户问题中提到的所有内容。\n"
        "citations 中每条必须包含：paper_id, chunk_id, quote。\n"
        "quote 必须是 evidence 原文短摘（<=40个英文词或<=80中文字符）。\n"
    )

    user_obj = {
        "query": query,
        "evidence_blocks": blocks,
        "output_schema": {
            "answer": "string",
            "citations": [
                {"paper_id": "string", "chunk_id": "string", "quote": "string"}
            ]
        }
    }

    # 关键：response_format json_object
    r = llm.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        temperature=cfg.temperature_final,
        max_tokens=cfg.max_tokens,
        response_format_json=True,
    )

    # 处理生成的答案，去掉空格、markdown的"**"、分段和制表符
    content = r.content.replace('**', '').replace('\n', ' ').replace('\t', ' ').strip()
    return _safe_json_loads(content)
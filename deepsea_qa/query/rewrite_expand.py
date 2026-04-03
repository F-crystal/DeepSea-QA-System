# -*- coding: utf-8 -*-
"""
rewrite_expand.py

作者：Accilia
创建时间：2026-02-23
用途说明：
  Query rewrite + expand（用于 BM25 稀疏检索）
- rewrites：必须非空，且格式多样
- expands：术语扩展（中英混合、同义/缩写/关键对象/方法）

设计要点：
  1) 模块独立：便于消融（llm / rules / hybrid）
  2) 当前阶段：不依赖检索结果（后续可升级为 PRF：基于 topK 文档再扩展）
  3) 约束：禁止编造具体论文题名/作者/年份，仅做术语层面的扩展
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Dict, Any, Literal

from deepsea_qa.query.types import QueryVariants, QueryClassification
from deepsea_qa.query.label_cards import LabelCardsStore
from deepsea_qa.llm.base import BaseLLM

from deepsea_qa.configs.query_config import RewriteExpandConfig


class QueryRewriteExpander:
    def __init__(
        self,
        store: Optional[LabelCardsStore] = None,
        llm: Optional[BaseLLM] = None,
        cfg: Optional[RewriteExpandConfig] = None,
    ):
        self.store = store or LabelCardsStore()
        self.store.load_all()
        self.llm = llm
        self.cfg = cfg or RewriteExpandConfig()

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        
        # 清理常见干扰字符
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    # ---- 规则兜底：保证 rewrites 非空、格式多样 ----
    def _rules_rewrite(self, query: str, cls: QueryClassification) -> List[str]:
        q = query.strip().rstrip("？?。.")
        # 1) 标准化问句
        r1 = f"{q} 的方法有哪些？"
        # 2) 关键词短语（去疑问词）
        r2 = q.replace("怎么", "").replace("如何", "").replace("？", "").replace("?", "").strip()
        # 3) 限定符/布尔式（适合稀疏检索）
        r3 = f"{r2} 多径 抗多径 均衡 纠错 AUV 声学通信"

        out = [r1, r2, r3]
        # 去重
        seen = set()
        res = []
        for x in out:
            x2 = (x or "").strip()
            if x2 and x2 not in seen:
                seen.add(x2)
                res.append(x2)
        return res[: self.cfg.max_rewrites]

    # ---- rules expand：使用 Top-K candidates 注入术语，避免单点误判 ----
    def _rules_expand(self, query: str, cls: QueryClassification) -> List[str]:
        expands: List[str] = []

        # 从 Top-K candidates 抽取关键词（跨方向软注入，降低误判放大）
        hints: List[str] = []
        cand_domains = []

        if cls.candidates:
            # 取前2个候选（可调）
            cand_domains = cls.candidates[:2]
        else:
            # 兼容旧结果
            cand_domains = []

        for c in cand_domains:
            domain_key = c.domain_name_zh
            if domain_key in self.store.list_domains():
                domain = self.store.get_domain(domain_key)
                card = domain.labels.get(c.label_id)
                if not card:
                    continue
                hints.extend(card.positive_keywords[:6])
                hints.extend(card.positive_keywords_en[:6])

        # 去重
        seen = set()
        h2 = []
        for k in hints:
            k2 = (k or "").strip()
            if not k2:
                continue
            key = k2.lower()
            if key in seen:
                continue
            seen.add(key)
            h2.append(k2)

        if h2:
            expands.append(query + " " + " ".join(h2[:10]))
            expands.append(" ".join(h2[:12]))

        return expands[: self.cfg.max_expands]

    def _llm_rewrite_expand(self, query: str, cls: QueryClassification) -> Optional[QueryVariants]:
        if not self.llm:
            return None

        # 给 LLM 的“候选标签提示”：用 Top-K candidates 的 scope/keywords
        label_hints: List[Dict[str, Any]] = []
        for c in (cls.candidates or [])[:2]:
            domain_key = c.domain_name_zh
            if domain_key in self.store.list_domains():
                domain = self.store.get_domain(domain_key)
                card = domain.labels.get(c.label_id)
                if not card:
                    continue
                label_hints.append({
                    "domain": domain_key,
                    "label_id": card.label_id,
                    "label_name_zh": card.label_name_zh,
                    "scope": card.scope_description,
                    "keywords_zh": card.positive_keywords[:10],
                    "keywords_en": card.positive_keywords_en[:10],
                })

        system = (
            "你是深海科技问答系统的查询改写与扩展模块。目标：提升BM25召回率。"
            "必须输出JSON：{rewrites:[...], expands:[...]}。"
            "强制要求 rewrites 至少3条且格式多样："
            "A) 标准问句（更规范、更明确）"
            "B) 关键词短语（适合BM25）"
            "C) 限定符/布尔式（含AND/OR 或中文同义词串）"
            "可选D) 英文术语检索式。"
            "expands 用于术语扩展（同义词/缩写/关键对象/方法，中英混合可）。"
            "禁止编造具体论文题名/作者/年份；只做术语层面的扩展。"
        )

        user = {
            "query": query,
            "top_candidates_hints": label_hints,
            "limits": {
                "max_rewrites": self.cfg.max_rewrites,
                "max_expands": self.cfg.max_expands,
            },
            "format_examples": {
                "A": "深海AUV声学通信抗多径的常用方法有哪些？",
                "B": "深海 AUV 声学通信 抗多径",
                "C": "AUV AND acoustic communication AND (multipath mitigation OR equalization OR channel estimation)",
                "D": "AUV acoustic communication multipath mitigation equalization"
            }
        }

        res = self.llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=self.cfg.temperature,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        obj = self._safe_json_loads(res.content)
        if not obj:
            return None

        rewrites = [str(x).strip() for x in (obj.get("rewrites") or []) if str(x).strip()]
        expands = [str(x).strip() for x in (obj.get("expands") or []) if str(x).strip()]

        # 强制保证 rewrites 非空：LLM输出不够就规则补齐
        if len(rewrites) < 2:
            rewrites = (rewrites + self._rules_rewrite(query, cls))[: self.cfg.max_rewrites]

        # expands 不够也可补齐
        if len(expands) < 2:
            expands = (expands + self._rules_expand(query, cls))[: self.cfg.max_expands]

        # 去重
        def _dedup(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for s in xs:
                s2 = s.strip()
                if s2 and s2 not in seen:
                    seen.add(s2)
                    out.append(s2)
            return out

        return QueryVariants(
            original=query,
            rewrites=_dedup(rewrites)[: self.cfg.max_rewrites],
            expands=_dedup(expands)[: self.cfg.max_expands],
        )

    def run(self, query: str, cls: QueryClassification) -> QueryVariants:
        query = (query or "").strip()
        if not query:
            return QueryVariants(original="")

        if self.cfg.strategy in ("llm", "hybrid"):
            qv = self._llm_rewrite_expand(query, cls)
            if qv:
                return qv

        # fallback：规则版本也保证 rewrites 非空
        rewrites = self._rules_rewrite(query, cls)
        expands = self._rules_expand(query, cls)
        return QueryVariants(original=query, rewrites=rewrites, expands=expands)
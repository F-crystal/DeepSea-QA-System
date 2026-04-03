# -*- coding: utf-8 -*-
"""
classifier.py

作者：Accilia
创建时间：2026-02-23
用途说明：
  用户查询的【方向(domain) + 标签(label)】分类模块。

设计要点：
  1) 模块独立：可单独做消融（rules / llm / hybrid）
  2) 可溯源：输出包含 evidence_spans（证据短语）
  3) 强约束：LLM 输出必须从 label cards 的 domain + label_id 集合中选择
  4) 健壮性：LLM 输出不合法时自动回退到规则分类
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple, Optional, Any

from deepsea_qa.query.types import QueryClassification, DomainId, LabelCandidate
from deepsea_qa.query.label_cards import LabelCardsStore
from deepsea_qa.llm.base import BaseLLM

from deepsea_qa.configs.query_config import ClassifierConfig


DOMAIN_MAP: Dict[str, Tuple[DomainId, str]] = {
    "深海感知与通信装备": ("SENSOR_COMM", "深海感知与通信装备"),
    "深海可再生能源": ("RENEWABLE", "深海可再生能源"),
    "深海矿产": ("MINERALS", "深海矿产"),
    "深水油气": ("OIL_GAS", "深水油气"),
}


class QueryClassifier:
    def __init__(
        self,
        store: Optional[LabelCardsStore] = None,
        llm: Optional[BaseLLM] = None,
        cfg: Optional[ClassifierConfig] = None,
    ):
        self.store = store or LabelCardsStore()
        self.store.load_all()
        self.llm = llm
        self.cfg = cfg or ClassifierConfig()

    # ---------- 工具：鲁棒 JSON 提取 ----------
    @staticmethod
    def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            print(f"Empty text received for JSON parsing")
            return None
        
        # 清理常见干扰字符
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            # 尝试提取JSON部分
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                print("No JSON found in text")
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    # ---------- rules：为每个 label 打分，输出 Top-K ----------
    def _rules_rank_candidates(self, query: str) -> List[LabelCandidate]:
        q = query.lower()

        cands: List[LabelCandidate] = []

        for domain_key in self.store.list_domains():
            domain = self.store.get_domain(domain_key)

            # 映射 domain_id
            if domain_key in DOMAIN_MAP:
                domain_id, domain_name = DOMAIN_MAP[domain_key]
            else:
                domain_id, domain_name = ("UNKNOWN", domain_key)

            for label_id, card in domain.labels.items():
                pos = [*card.positive_keywords, *card.positive_keywords_en]
                neg = card.negative_keywords or []

                # 正向命中
                pos_hits = []
                pos_score = 0
                for kw in pos:
                    kw2 = (kw or "").strip().lower()
                    if kw2 and kw2 in q:
                        pos_score += 1
                        pos_hits.append(kw)

                # 负向命中（降权）
                neg_score = 0
                if self.cfg.use_negative_keywords:
                    for kw in neg:
                        kw2 = (kw or "").strip().lower()
                        if kw2 and kw2 in q:
                            neg_score += 1

                score = pos_score - 0.8 * neg_score
                if score <= 0:
                    continue

                cands.append(
                    LabelCandidate(
                        domain_id=domain_id,
                        domain_name_zh=domain_name,
                        label_id=label_id,
                        score=float(score),
                        evidence_spans=list(dict.fromkeys(pos_hits))[:4],
                        rationale="rules: keyword hits",
                    )
                )

        cands.sort(key=lambda x: x.score, reverse=True)
        return cands[: self.cfg.top_k]

    # ---------- llm：输出 Top-K candidates ----------
    def _llm_rank_candidates(self, query: str) -> Optional[List[LabelCandidate]]:
        if not self.llm:
            print("LLM is None.")
            return None

        print(">>> calling LLM for classification...")

        domains = {}
        for dk in self.store.list_domains():
            d = self.store.get_domain(dk)
            domains[dk] = {
                "domain_zh": d.domain_zh,
                "labels": sorted(list(d.labels.keys())),
            }

        system = (
            "你是深海科技问答系统的查询分类器。"
            "请输出 Top-K 候选类别（允许跨方向），用于后续检索过滤。"
            "必须从给定的 domain 与 label_id 集合中选择，禁止自造类别。"
            "请按照json格式输出，输出必须为 JSON：{candidates:[{domain_zh,label_id,score,evidence_spans,rationale},...] }。"
            "score为0-1之间的小数，按相关性降序。evidence_spans为从query中截取的短语/关键词（<=4）。"
        )

        user = {
            "query": query,
            "domains_and_labels": domains,
            "top_k": self.cfg.top_k,
            "notes": "若无法判断，返回 candidates=[]",
        }

        try:
            res = self.llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                temperature=self.cfg.llm_temperature,
                max_tokens=1000,  # 增加max_tokens值，确保模型能够返回完整的JSON内容
                response_format={"type": "json_object"},
            )
            # print(f"LLM response head: {res.content[:120]}")
        except Exception as e:
            print(f"LLM classification error: {e}")
            return None

        obj = self._safe_json_loads(res.content)
        if not obj or "candidates" not in obj:
            print(f"Json parse error: {obj}")
            return None

        out: List[LabelCandidate] = []
        for it in obj.get("candidates") or []:
            domain_zh = it.get("domain_zh")
            label_id = it.get("label_id")
            score = it.get("score", 0)

            if domain_zh not in DOMAIN_MAP:
                continue
            domain_id, domain_name = DOMAIN_MAP[domain_zh]

            # label 合法性校验
            if domain_zh not in self.store.list_domains():
                continue
            domain_cards = self.store.get_domain(domain_zh)
            if label_id not in domain_cards.labels:
                continue

            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            score_f = max(0.0, min(score_f, 1.0))

            evi = it.get("evidence_spans") or []
            evi = [str(x).strip()[:60] for x in evi if str(x).strip()][:4]

            out.append(
                LabelCandidate(
                    domain_id=domain_id,
                    domain_name_zh=domain_name,
                    label_id=str(label_id),
                    score=score_f,
                    evidence_spans=evi,
                    rationale=str(it.get("rationale") or "")[:200],
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out[: self.cfg.top_k] if out else None

    def classify(self, query: str) -> QueryClassification:
        query = (query or "").strip()
        if not query:
            return QueryClassification(
                domain_id="UNKNOWN",
                domain_name_zh="UNKNOWN",
                primary_label_id="UNKNOWN_OTHER",
                candidates=[],
                confidence=0.0,
                evidence_spans=[],
                method="rules",
            )

        rules_cands = self._rules_rank_candidates(query)

        llm_cands = None
        if self.cfg.strategy in ("llm", "hybrid"):
            llm_cands = self._llm_rank_candidates(query)

        # 选择最终 candidates：llm 优先，否则 rules
        final_cands = llm_cands if llm_cands else rules_cands
        method = "llm" if llm_cands and self.cfg.strategy == "llm" else ("hybrid" if llm_cands else "rules")

        if final_cands:
            top = final_cands[0]
            # secondary：同方向内的其余高分
            same_domain = [c for c in final_cands[1:] if c.domain_id == top.domain_id]
            secondary = [c.label_id for c in same_domain[: self.cfg.max_secondary]]

            # confidence：直接用 top.score（llm为0-1；rules为正数，可压到0-1）
            conf = top.score
            if method == "rules":
                conf = min(0.9, 0.35 + 0.15 * min(4.0, top.score))

            return QueryClassification(
                domain_id=top.domain_id,
                domain_name_zh=top.domain_name_zh,
                primary_label_id=top.label_id,
                secondary_label_ids=secondary,
                confidence=float(conf),
                evidence_spans=top.evidence_spans,
                method=method,  # rules/llm/hybrid
                raw={"llm_used": bool(llm_cands)},
                candidates=final_cands,
            )

        # 完全无候选
        return QueryClassification(
            domain_id="UNKNOWN",
            domain_name_zh="UNKNOWN",
            primary_label_id="UNKNOWN_OTHER",
            candidates=[],
            confidence=0.0,
            evidence_spans=[],
            method="rules",
            raw={"llm_used": bool(llm_cands)},
        )
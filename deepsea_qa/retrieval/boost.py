# -*- coding: utf-8 -*-
"""
retrieval/boost.py

作者：Accilia
创建时间：2026-02-24
用途说明：分类 soft boost 模块：
- 不硬过滤（避免误杀）
- 根据 classification.candidates 的 topK，对匹配域(或标签)的候选 chunk 进行加权
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from deepsea_qa.retrieval.types import FusedChunk


class ClassificationBooster:
    def __init__(self, enable: bool, mode: str, use_top_k: int, boost_weight: float):
        self.enable = enable
        self.mode = mode
        self.use_top_k = use_top_k
        self.boost_weight = boost_weight

    def apply(
        self,
        fused: List[FusedChunk],
        query_bundle: dict,
        chunk_meta_map: Dict[str, Dict[str, str]],
    ) -> List[FusedChunk]:
        """
        chunk_meta_map: {chunk_id: {"domain_id": "...", "label_id": "...(optional)"}}
        query_bundle: query pipeline 输出（含 classification.candidates）
        """
        if not self.enable:
            return fused

        cands = query_bundle.get("classification", {}).get("candidates", []) or []
        cands = cands[: max(0, int(self.use_top_k))]

        # 形成目标集合：domain_id->score 或 label_id->score
        target: Dict[str, float] = {}
        key_name = "domain_id" if self.mode == "domain" else "label_id"

        for c in cands:
            k = str(c.get(key_name, "")).strip()
            if not k:
                continue
            # score 0-1，越大越相关
            try:
                sc = float(c.get("score", 0.0))
            except Exception:
                sc = 0.0
            target[k] = max(target.get(k, 0.0), sc)

        if not target:
            return fused

        for fc in fused:
            meta = chunk_meta_map.get(fc.chunk_id) or {}
            v = str(meta.get(key_name, "")).strip()
            if v and v in target:
                # 核心公式：score *= (1 + boost_weight * cand_score)
                mult = 1.0 + self.boost_weight * float(target[v])
                fc.score_boosted = fc.score_fused * mult
                fc.debug["boost"] = f"{key_name}={v}, cand_score={target[v]:.3f}, mult={mult:.3f}"
            else:
                fc.score_boosted = fc.score_fused

        fused.sort(key=lambda x: x.score_boosted, reverse=True)
        return fused
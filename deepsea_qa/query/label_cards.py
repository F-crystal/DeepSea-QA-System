# -*- coding: utf-8 -*-
"""
label_cards.py

作者：Accilia
创建时间：2026-02-23
用途说明： 读取与缓存分类标签卡（{方向}.json），并提供关键词/规则访问接口
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from deepsea_qa.configs.paths import DIR_LABEL_CARDS


@dataclass(frozen=True)
class LabelCard:
    label_id: str
    label_name_zh: str
    scope_description: str
    inclusion_rules: List[str]
    exclusion_rules: List[str]
    positive_keywords: List[str]
    positive_keywords_en: List[str]
    negative_keywords: List[str]
    priority_rules: List[str]


@dataclass
class DomainLabelCards:
    domain_zh: str
    domain_en: str
    labels: Dict[str, LabelCard]  # label_id -> LabelCard


class LabelCardsStore:
    """
    加载 DIR_LABEL_CARDS 下的 4 个方向 json。
    """
    def __init__(self, label_cards_dir: Path | None = None):
        self.label_cards_dir = label_cards_dir or DIR_LABEL_CARDS
        self._domains: Dict[str, DomainLabelCards] = {}  # domain_key -> DomainLabelCards

    def load_all(self) -> None:
        if self._domains:
            return

        if not self.label_cards_dir.exists():
            raise FileNotFoundError(f"Label cards dir not found: {self.label_cards_dir}")

        json_files = sorted(self.label_cards_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No label card json found in: {self.label_cards_dir}")

        for fp in json_files:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            meta = obj.get("meta", {})
            labels = obj.get("labels", [])
            domain_zh = meta.get("domain", fp.stem)
            domain_en = meta.get("domain_en", "")

            label_map: Dict[str, LabelCard] = {}
            for it in labels:
                label_id = it["label_id"]
                label_map[label_id] = LabelCard(
                    label_id=label_id,
                    label_name_zh=it.get("label_name_zh", ""),
                    scope_description=it.get("scope_description", ""),
                    inclusion_rules=it.get("inclusion_rules", []) or [],
                    exclusion_rules=it.get("exclusion_rules", []) or [],
                    positive_keywords=it.get("positive_keywords", []) or [],
                    positive_keywords_en=it.get("positive_keywords_en", []) or [],
                    negative_keywords=it.get("negative_keywords", []) or [],
                    priority_rules=it.get("priority_rules", []) or [],
                )

            # domain_key 用文件名即可（如：深海矿产 / 深海可再生能源 / 深水油气 / 深海感知与通信装备）
            self._domains[fp.stem] = DomainLabelCards(
                domain_zh=domain_zh,
                domain_en=domain_en,
                labels=label_map,
            )

    def list_domains(self) -> List[str]:
        self.load_all()
        return sorted(self._domains.keys())

    def get_domain(self, domain_key: str) -> DomainLabelCards:
        self.load_all()
        if domain_key not in self._domains:
            raise KeyError(f"Unknown domain_key={domain_key}. Available={self.list_domains()}")
        return self._domains[domain_key]
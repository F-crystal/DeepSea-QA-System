# -*- coding: utf-8 -*-
"""
rerank.py

作者：Accilia
创建时间：2026-02-24
用途说明：Cross-Encoder 重排（batch 推理版）：
- 输入：query + TopM passages
- 输出：按 rerank 分数排序的 TopN chunk_id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from deepsea_qa.retrieval.types import FusedChunk


@dataclass
class RerankOutput:
    chunk_ids: List[str]
    scores: List[float]


class Reranker:
    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True, max_length: int = 512):
        self.model_name = model_name
        self.device = torch.device(device)
        self.fp16 = fp16
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: List[FusedChunk],
        passages: Dict[str, str],
        top_m: int,
        final_top_n: int,
        batch_size: int = 16,
    ) -> List[FusedChunk]:
        """
        passages: {chunk_id: text}（由 sqlite 回查得到）
        candidates: 融合后排序结果
        """
        query = (query or "").strip()
        if not query or not candidates:
            return candidates[:final_top_n]

        # 取 TopM 且必须有文本
        picked: List[FusedChunk] = []
        for fc in candidates[:top_m]:
            if fc.chunk_id in passages and passages[fc.chunk_id]:
                picked.append(fc)

        if not picked:
            return candidates[:final_top_n]

        # batch 推理：构造 pair inputs
        all_scores: List[Tuple[str, float]] = []
        for i in range(0, len(picked), batch_size):
            batch = picked[i:i + batch_size]
            texts = [passages[x.chunk_id] for x in batch]

            enc = self.tokenizer(
                [query] * len(batch),
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # fp16 加速（仅在 cuda 上启用）
            if self.fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            # 通用取 logits：不同模型可能是 (bs,1) 或 (bs,2)
            logits = out.logits
            if logits.dim() == 2 and logits.size(-1) > 1:
                # 二分类时常用取“正类”logit
                scores = logits[:, -1]
            else:
                scores = logits.view(-1)

            for fc, sc in zip(batch, scores.detach().float().cpu().tolist()):
                fc.score_rerank = float(sc)
                all_scores.append((fc.chunk_id, float(sc)))

        # 按 rerank 分数降序
        picked.sort(key=lambda x: x.score_rerank, reverse=True)
        return picked[:final_top_n]
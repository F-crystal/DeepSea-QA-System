# -*- coding: utf-8 -*-
"""
chunking.py

作者：Accilia
创建时间：2026-02-22
用途说明: 将每条 paper 的 text 切分为多个“语义单元”，并保留溯源信息（offset等）。

策略：
- 初步切分：按换行、句末标点、分号等正则切分
- 使用 SciBERT tokenizer 进行 token 计数与长度约束
- 长度约束：过短向后/向前合并；过长再按软标点切；仍过长则硬切
- 输出：[(chunk_text, start_char, end_char), ...]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TokenChunkingConfig:
    # 推荐区间：80-256 tokens
    min_tokens: int = 80
    max_tokens: int = 256

    # 初切语义边界
    split_pattern: str = r"(?:\n+|(?<=[。！？!?；;])\s+|(?<=\.)\s+|(?<=:)\s+|(?<=：)\s+)"

    # tokenizer
    tokenizer_name: str = "scibert_scivocab_uncased"
    local_files_only: bool = True  # 服务器若已缓存可保持 True；没缓存则改 False 让它下载


def _initial_split(text: str, cfg: TokenChunkingConfig) -> List[Tuple[str, int, int]]:
    text = str(text).strip()
    if not text:
        return []

    parts: List[Tuple[str, int, int]] = []
    last = 0
    for m in re.finditer(cfg.split_pattern, text):
        cut = m.start()
        seg = text[last:cut].strip()
        if seg:
            start = text.find(seg, last, cut)
            end = start + len(seg)
            parts.append((seg, start, end))
        last = m.end()

    tail = text[last:].strip()
    if tail:
        start = text.find(tail, last)
        end = start + len(tail)
        parts.append((tail, start, end))

    return parts


def _load_tokenizer(cfg: TokenChunkingConfig):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装 transformers，无法使用 SciBERT tokenizer。"
        ) from e

    try:
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, local_files_only=cfg.local_files_only)
        return tok
    except Exception as e:
        hint = (
            f"无法加载 tokenizer: {cfg.tokenizer_name} (local_files_only={cfg.local_files_only}).\n"
            "如果你在服务器上首次运行且尚未缓存，请把 local_files_only=False 以允许下载。\n"
        )
        raise RuntimeError(hint) from e


def _count_tokens(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    return int(len(ids))


def _pack_by_tokens(
    tok,
    parts: List[Tuple[str, int, int]],
    cfg: TokenChunkingConfig
) -> List[Tuple[str, int, int]]:
    """
    将初切 parts 逐段拼接，直到 token 数接近 max_tokens。
    - 若单段本身超过 max_tokens：按 token 边界切成多个 chunk（严格 token 切）
    - 若拼完仍低于 min_tokens：尽量与下一段合并（保证语义密度）
    """
    if not parts:
        return []

    out: List[Tuple[str, int, int]] = []

    cur_text = ""
    cur_s: Optional[int] = None
    cur_e: Optional[int] = None
    cur_tokens = 0

    def flush():
        nonlocal cur_text, cur_s, cur_e, cur_tokens
        if cur_text.strip():
            out.append((cur_text.strip(), int(cur_s or 0), int(cur_e or 0)))
        cur_text, cur_s, cur_e, cur_tokens = "", None, None, 0

    for seg, s, e in parts:
        seg = seg.strip()
        if not seg:
            continue

        seg_tokens = _count_tokens(tok, seg)

        # 情况A：单段就超长 -> 严格按 token 切成多个
        if seg_tokens > cfg.max_tokens:
            # 先把当前累积的 chunk 输出
            flush()

            ids = tok.encode(seg, add_special_tokens=False)
            # 以 max_tokens 为步长切 ids，再 decode 回文本（严格 token 切分）
            start_idx = 0
            while start_idx < len(ids):
                sub_ids = ids[start_idx:start_idx + cfg.max_tokens]
                sub_text = tok.decode(sub_ids, skip_special_tokens=True).strip()
                if not sub_text:
                    start_idx += cfg.max_tokens
                    continue

                # 估算 sub_text 在 seg 中的字符 offset（保底可溯源）
                # decode 后字符串可能与原 seg 有空格差异，因此用 find 近似定位
                local_pos = seg.find(sub_text)
                if local_pos < 0:
                    local_pos = 0
                sub_s = s + local_pos
                sub_e = sub_s + len(sub_text)
                out.append((sub_text, sub_s, sub_e))
                start_idx += cfg.max_tokens
            continue

        # 情况B：正常段，尝试拼接到当前 chunk
        if not cur_text:
            cur_text = seg
            cur_s, cur_e = s, e
            cur_tokens = seg_tokens
        else:
            # 试拼接后的 token 数
            candidate = (cur_text + " " + seg).strip()
            cand_tokens = _count_tokens(tok, candidate)

            if cand_tokens <= cfg.max_tokens:
                cur_text = candidate
                cur_e = e
                cur_tokens = cand_tokens
            else:
                # 当前 chunk 到上限了 -> 输出当前
                flush()
                # 新起一个
                cur_text = seg
                cur_s, cur_e = s, e
                cur_tokens = seg_tokens

        # 尽量保证不要太短：如果当前已成 chunk 但还 < min_tokens，不立刻 flush，继续吃下一段
        #（由上面逻辑自然实现：只有超 max_tokens 才 flush）

    # 最后 flush
    flush()

    # 最后一步：如果最后一个 chunk 太短，且前面有 chunk，可并入前一个（避免尾巴碎片）
    if len(out) >= 2:
        last_text, last_s, last_e = out[-1]
        last_tokens = _count_tokens(tok, last_text)
        if last_tokens < cfg.min_tokens:
            prev_text, prev_s, prev_e = out[-2]
            merged = (prev_text + " " + last_text).strip()
            merged_tokens = _count_tokens(tok, merged)
            if merged_tokens <= cfg.max_tokens:
                out[-2] = (merged, prev_s, last_e)
                out.pop()

    return out


def split_into_chunks_token_based(text: str, cfg: TokenChunkingConfig) -> List[Tuple[str, int, int]]:
    """
    主入口：SciBERT token-based chunking
    """
    text = str(text).strip()
    if not text:
        return []

    tok = _load_tokenizer(cfg)
    parts = _initial_split(text, cfg)
    packed = _pack_by_tokens(tok, parts, cfg)

    # 保底
    if not packed:
        return [(text, 0, len(text))]
    return packed
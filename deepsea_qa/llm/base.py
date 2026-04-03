# -*- coding: utf-8 -*-
"""
base.py

作者：Accilia
创建时间：2026-02-23
用途说明：
  LLM 抽象接口层（统一 chat 调用与返回结构）
  - 支持多模型/多provider对比实验
  - 便于后续：日志、缓存、限流、成本统计、可追溯性
  - 增强：chat_json / async wrappers / stream fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Generator
from abc import ABC, abstractmethod
import asyncio
import json
import re
import time


@dataclass
class ChatResult:
    content: str
    raw: Optional[Any] = None               # 原始对象/字典（debug用）
    usage: Optional[Dict[str, Any]] = None  # token统计等（若可得）
    model: Optional[str] = None
    latency: Optional[float] = None         # 秒


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    t = _strip_json_fence(text)
    if not t:
        return {}
    try:
        return json.loads(t)
    except Exception:
        # 兜底：尝试截取首尾大括号内容
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


class BaseLLM(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        response_format_json: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        """
        kwargs：为不同 provider 预留扩展参数（如 response_format / stream 等）
        """
        raise NotImplementedError

    # -------------------------
    # 1) 同步便捷接口
    # -------------------------
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> ChatResult:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        强制 JSON 输出并解析：
          - 优先用 provider 的 response_format={"type":"json_object"}
          - 兜底：清理 ```json fence + json.loads
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        r = self.chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format_json=True if response_format is None else False,
            response_format=response_format,
            **kwargs,
        )
        return _safe_json_loads(r.content)

    # -------------------------
    # 2) async wrappers（对齐 generation pipeline）
    # -------------------------
    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        response_format_json: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.to_thread(
            self.chat,
            messages,
            temperature,
            top_p,
            max_tokens,
            response_format_json,
            **kwargs,
        )

    async def achat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.chat_json,
            system,
            user,
            temperature,
            top_p,
            max_tokens,
            response_format,
            **kwargs,
        )

    # -------------------------
    # 3) Streaming（默认 fallback）
    # -------------------------
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        chunk_size: int = 80,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        provider 若没实现真正 token stream，则 fallback：
          先拿完整 content，再按 chunk_size 切片 yield。
        """
        r = self.chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )
        text = r.content or ""
        for i in range(0, len(text), int(chunk_size)):
            yield text[i:i + int(chunk_size)]
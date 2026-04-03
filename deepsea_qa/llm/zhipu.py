# -*- coding: utf-8 -*-
"""
zhipu.py

作者：Accilia
创建时间：2026-02-23
用途说明：智谱 LLM provider（通过 zai SDK 调用）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generator

from deepsea_qa.llm.base import BaseLLM, ChatResult


@dataclass
class ZhipuSettings:
    api_key_env: str = "ZHIPU_API_KEY"
    model: str = "glm-4-plus"
    timeout: int = 60
    max_retries: int = 2


class ZhipuLLM(BaseLLM):
    def __init__(self, settings: Optional[ZhipuSettings] = None):
        self.settings = settings or ZhipuSettings()
        api_key = os.getenv(self.settings.api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(f"Missing API key env var: {self.settings.api_key_env}")

        super().__init__(model=self.settings.model)

        try:
            from zai import ZhipuAiClient
        except Exception as e:
            raise ImportError("Cannot import zai. Please ensure zai is installed.") from e

        self.client = ZhipuAiClient(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        response_format_json: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        统一 chat：
          - 支持 response_format={"type":"json_object"}（ZAI structured output）
          - 向后兼容 response_format_json=True
        """
        last_err: Optional[Exception] = None

        for _ in range(self.settings.max_retries + 1):
            t0 = time.time()
            try:
                call_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                # timeout 透传（SDK 是否接受由版本决定，放在 kwargs 中不强制）
                if self.settings.timeout:
                    call_kwargs["timeout"] = self.settings.timeout

                # ✅ response_format：优先显式传入；否则根据 response_format_json
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                elif response_format_json:
                    call_kwargs["response_format"] = {"type": "json_object"}

                # 允许上层额外透传（比如 tools 等）
                call_kwargs.update(kwargs)

                resp = self.client.chat.completions.create(**call_kwargs)
                latency = time.time() - t0

                content = resp.choices[0].message.content

                # usage（不同 SDK 结构不同，尽量读取）
                usage = None
                if hasattr(resp, "usage") and resp.usage:
                    try:
                        usage = dict(resp.usage)
                    except Exception:
                        usage = getattr(resp, "usage", None)

                # raw：尽量转 dict
                raw = None
                try:
                    if hasattr(resp, "model_dump"):
                        raw = resp.model_dump()
                    elif hasattr(resp, "dict"):
                        raw = resp.dict()
                    else:
                        raw = resp
                except Exception:
                    raw = resp

                return ChatResult(
                    content=content,
                    raw=raw,
                    usage=usage,
                    model=self.model,
                    latency=latency,
                )

            except Exception as e:
                last_err = e
                time.sleep(0.6)

        raise RuntimeError(f"ZhipuLLM.chat failed after retries: {last_err}")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        chunk_size: int = 80,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        真流式（优先） + fallback：
        - 优先尝试 stream=True 的 zai SDK
        - 如果 SDK 不支持/结构不一致，就 fallback 到 BaseLLM 的“切片流式”
        """
        try:
            call_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if self.settings.timeout:
                call_kwargs["timeout"] = self.settings.timeout
            if response_format is not None:
                call_kwargs["response_format"] = response_format

            call_kwargs.update(kwargs)

            stream = self.client.chat.completions.create(**call_kwargs)

            # zai 的流式事件结构在不同版本可能不同，这里做最大兼容：
            # 常见：event.choices[0].delta.content 或 event.choices[0].message.content
            for ev in stream:
                delta = ""
                try:
                    ch = ev.choices[0]
                    if hasattr(ch, "delta") and ch.delta and getattr(ch.delta, "content", None):
                        delta = ch.delta.content
                    elif hasattr(ch, "message") and ch.message and getattr(ch.message, "content", None):
                        delta = ch.message.content
                except Exception:
                    delta = ""

                if delta:
                    yield delta

            return

        except Exception:
            # fallback：切片流式（不报错，保证 streaming 体验至少可用）
            for s in super().stream_chat(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                chunk_size=chunk_size,
                **kwargs,
            ):
                yield s
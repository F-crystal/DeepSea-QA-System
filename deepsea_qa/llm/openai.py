# -*- coding: utf-8 -*-
"""
openai.py

作者：Accilia
创建时间：2026-03-09
用途说明：OpenAI LLM provider（通过 openai SDK 调用 GPT 模型）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generator

from deepsea_qa.llm.base import BaseLLM, ChatResult


@dataclass
class OpenAISettings:
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"
    base_url: str = "https://xiaoai.plus/v1"  # 中转站服务器 URL
    timeout: int = 120
    max_retries: int = 5


class OpenAILLM(BaseLLM):
    def __init__(self, settings: Optional[OpenAISettings] = None):
        self.settings = settings or OpenAISettings()
        api_key = os.getenv(self.settings.api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(f"Missing API key env var: {self.settings.api_key_env}")

        super().__init__(model=self.settings.model)

        try:
            import openai
        except Exception as e:
            raise ImportError("Cannot import openai. Please ensure openai is installed (pip install openai).") from e

        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.settings.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        response_format_json: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        统一 chat：
          - 支持 response_format={"type":"json_object"} (OpenAI 原生支持)
          - 向后兼容 response_format_json=True
        """
        last_err: Optional[Exception] = None

        for _ in range(self.settings.max_retries + 1):
            t0 = time.time()
            try:
                # 构建请求参数
                call_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                
                # 只有当 max_tokens 不为 None 时才传递该参数
                if max_tokens is not None:
                    call_kwargs["max_tokens"] = max_tokens
                
                # timeout 透传
                if self.settings.timeout:
                    call_kwargs["timeout"] = self.settings.timeout

                # response_format 处理
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                elif response_format_json:
                    call_kwargs["response_format"] = {"type": "json_object"}
                
                # 允许上层额外透传
                call_kwargs.update(kwargs)

                # 调用 SDK
                resp = self.client.chat.completions.create(**call_kwargs)
                latency = time.time() - t0
                
                # 添加缓冲时间，避免API速率限制
                time.sleep(1.0)

                # 提取内容
                content = resp.choices[0].message.content or ""
                
                # Usage 提取
                usage = None
                if hasattr(resp, "usage") and resp.usage:
                    try:
                        if hasattr(resp.usage, "model_dump"):
                            usage = resp.usage.model_dump()
                        elif hasattr(resp.usage, "__dict__"):
                            usage = resp.usage.__dict__
                        else:
                            usage = dict(resp.usage)
                    except Exception:
                        usage = getattr(resp, "usage", None)

                # Raw 数据提取
                raw = None
                try:
                    if hasattr(resp, "model_dump"):
                        raw = resp.model_dump()
                    elif hasattr(resp, "__dict__"):
                        raw = resp.__dict__
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
                # 网络波动较常见，适当重试
                time.sleep(0.3)

        raise RuntimeError(f"OpenAILLM.chat failed after retries: {last_err}")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        chunk_size: int = 80,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        真流式调用 OpenAI SDK
        """
        try:
            call_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
            }
            
            # 只有当 max_tokens 不为 None 时才传递该参数
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            
            if self.settings.timeout:
                call_kwargs["timeout"] = self.settings.timeout
            
            if response_format is not None:
                call_kwargs["response_format"] = response_format

            call_kwargs.update(kwargs)

            # 调用流式接口
            stream = self.client.chat.completions.create(**call_kwargs)

            # 遍历流式响应
            for chunk in stream:
                delta = ""
                try:
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if choice.delta and choice.delta.content:
                            delta = choice.delta.content
                except Exception:
                    delta = ""

                if delta:
                    yield delta

            # 添加缓冲时间，避免API速率限制
            time.sleep(0.6)
            return

        except Exception:
            # Fallback：如果流式失败，降级为非流式切片
            try:
                result = self.chat(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs
                )
                content = result.content
                # 简单切片模拟流式
                for i in range(0, len(content), chunk_size):
                    yield content[i:i+chunk_size]
            except Exception:
                # 彻底失败，静默退出
                return

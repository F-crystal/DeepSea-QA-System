# -*- coding: utf-8 -*-
"""
dashscope.py

作者：Accilia 
创建时间：2026-03-01
用途说明：阿里云 DashScope LLM provider（通过 dashscope SDK 调用通义千问）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generator

from deepsea_qa.llm.base import BaseLLM, ChatResult


@dataclass
class DashScopeSettings:
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "qwen-plus"  # 推荐默认模型：qwen-plus, qwen-max, qwen-flash
    timeout: int = 120  # 增加超时时间到120秒
    max_retries: int = 5  # 增加重试次数到5次


class DashScopeLLM(BaseLLM):
    def __init__(self, settings: Optional[DashScopeSettings] = None):
        self.settings = settings or DashScopeSettings()
        api_key = os.getenv(self.settings.api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(f"Missing API key env var: {self.settings.api_key_env}")

        super().__init__(model=self.settings.model)

        try:
            import dashscope
            from dashscope import Generation
        except Exception as e:
            raise ImportError("Cannot import dashscope. Please ensure dashscope is installed (pip install dashscope).") from e

        # 设置 API Key (DashScope SDK 全局生效，也可在 call 时传入)
        dashscope.api_key = api_key
        self.client = Generation

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
          - 支持 response_format={"type":"json_object"} (DashScope 原生支持)
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
                    "result_format": "message",  # 关键：返回格式为 message 列表，兼容 OpenAI 结构
                }
                
                # 只有当 max_tokens 不为 None 时才传递该参数
                if max_tokens is not None:
                    call_kwargs["max_tokens"] = max_tokens
                
                # timeout 透传 (DashScope SDK 支持 request_timeout)
                if self.settings.timeout:
                    call_kwargs["request_timeout"] = self.settings.timeout

                # ✅ response_format 处理
                # DashScope 原生支持 result_format 或在 messages 中引导，
                # 新版 SDK 支持 response_format 参数直接控制 JSON 输出
                # 注意：根据官方文档，response_format 应该作为顶层参数
                if response_format is not None:
                    call_kwargs["response_format"] = response_format
                elif response_format_json:
                    call_kwargs["response_format"] = {"type": "json_object"}
                
                # 对于 Qwen3 商业版模型，需要确保 result_format 为 message
                # 参考官方文档：Qwen3 商业版模型默认值为 text，需要将其设置为 message
                if self.model in ["qwen-plus", "qwen-max", "qwen-flash", "qwen-coder"]:
                    call_kwargs["result_format"] = "message"

                # 允许上层额外透传（比如 tools, stop_words 等）
                call_kwargs.update(kwargs)

                # 调用 SDK
                resp = self.client.call(**call_kwargs)
                latency = time.time() - t0

                # ✅ 错误处理 (DashScope 返回对象中包含 status_code)
                if resp.status_code != 200:
                    raise RuntimeError(f"DashScope API error: {resp.code} - {resp.message}")

                # 提取内容 (DashScope 结构: resp.output.choices[0].message.content)
                content = ""
                if hasattr(resp, "output") and resp.output:
                    if hasattr(resp.output, "choices") and len(resp.output.choices) > 0:
                        choice = resp.output.choices[0]
                        if hasattr(choice, "message") and choice.message:
                            content = getattr(choice.message, "content", "")
                        # 兼容旧版结构：直接 output.text (较少见，但在某些模式下存在)
                        elif hasattr(choice, "text"):
                            content = choice.text
                
                if not content and hasattr(resp.output, "text"):
                    content = resp.output.text

                # Usage 提取
                usage = None
                if hasattr(resp, "usage") and resp.usage:
                    try:
                        # DashScope usage 通常是一个对象，转为 dict
                        if hasattr(resp.usage, "__dict__"):
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
                # DashScope 网络波动较常见，适当重试
                time.sleep(0.6)

        raise RuntimeError(f"DashScopeLLM.chat failed after retries: {last_err}")

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
        真流式调用 DashScope SDK
        """
        try:
            call_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "result_format": "message",
                "stream": True,  # ✅ 开启流式
            }
            
            # 只有当 max_tokens 不为 None 时才传递该参数
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            
            if self.settings.timeout:
                call_kwargs["request_timeout"] = self.settings.timeout
            
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            
            # 对于 Qwen3 商业版模型，需要确保 result_format 为 message
            # 参考官方文档：Qwen3 商业版模型默认值为 text，需要将其设置为 message
            if self.model in ["qwen-plus", "qwen-max", "qwen-flash", "qwen-coder"]:
                call_kwargs["result_format"] = "message"

            call_kwargs.update(kwargs)

            # 调用流式接口
            responses = self.client.call(**call_kwargs)

            # 遍历流式响应
            # DashScope 流式返回结构：每个 chunk 是一个 GenerationResponse 对象
            # 内容通常在: chunk.output.choices[0].message.content
            for chunk in responses:
                if chunk.status_code != 200:
                    # 流式中出错，可以选择抛出或跳过
                    continue
                
                delta = ""
                try:
                    if hasattr(chunk, "output") and chunk.output:
                        if hasattr(chunk.output, "choices") and len(chunk.output.choices) > 0:
                            choice = chunk.output.choices[0]
                            if hasattr(choice, "message") and choice.message:
                                delta = getattr(choice.message, "content", "") or ""
                            # 兼容部分模型的 text 字段
                            elif hasattr(choice, "text"):
                                delta = getattr(choice, "text", "") or ""
                except Exception:
                    delta = ""

                if delta:
                    yield delta

            return

        except Exception:
            # Fallback：如果流式失败，降级为非流式切片（保证不报错）
            # 注意：这里需要重新调用一次非流式 chat，为了简单起见，复用父类逻辑或手动调用
            # 由于父类 stream_chat 可能依赖 self.chat，这里直接调用 self.chat 并模拟切片
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
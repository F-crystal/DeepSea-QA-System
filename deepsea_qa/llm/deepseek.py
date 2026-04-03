# -*- coding: utf-8 -*-
"""
deepseek.py

作者：Accilia
创建时间：2026-02-27
用途说明：
  DeepSeek 模型封装（使用 OpenAI 兼容接口）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from deepsea_qa.llm.base import BaseLLM, ChatResult


@dataclass
class DeepSeekSettings:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3


class DeepSeekLLM(BaseLLM):
    """DeepSeek 模型封装"""
    
    def __init__(self, settings: DeepSeekSettings):
        super().__init__(model=settings.model)
        self.settings = settings
        self.api_key = settings.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        self.base_url = settings.base_url
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        response_format_json: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        """调用 DeepSeek 模型"""
        start_time = time.time()
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        
        last_err = None
        for i in range(self.settings.max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.settings.timeout)
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", None)
                
                latency = time.time() - start_time
                
                # 添加缓冲时间，避免API速率限制
                time.sleep(1.0)
                
                return ChatResult(
                    content=content,
                    raw=data,
                    usage=usage,
                    model=self.model,
                    latency=latency
                )
            except Exception as e:
                last_err = e
                print(f"DeepSeek API 调用失败 (尝试 {i+1}/{self.settings.max_retries+1}): {e}")
                if i < self.settings.max_retries:
                    # 指数退避策略
                    time.sleep(2 ** i)
        
        print(f"DeepSeek API 调用最终失败: {last_err}")
        return ChatResult(
            content="",
            raw=None,
            usage=None,
            model=self.model,
            latency=time.time() - start_time
        )

# -*- coding: utf-8 -*-
"""
registry.py

作者：Accilia
创建时间：2026-02-23
用途说明：LLM provider 注册与构造。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from deepsea_qa.llm.base import BaseLLM
from deepsea_qa.llm.zhipu import ZhipuLLM, ZhipuSettings
from deepsea_qa.llm.deepseek import DeepSeekLLM, DeepSeekSettings
from deepsea_qa.llm.dashscope import DashScopeLLM, DashScopeSettings
from deepsea_qa.llm.openai import OpenAILLM, OpenAISettings


Provider = Literal["zhipu", "deepseek", "dashscope", "openai"]


@dataclass
class LLMSettings:
    provider: Provider = "zhipu"
    model: str = "glm-4-plus"


def build_llm(settings: Optional[LLMSettings] = None) -> BaseLLM:
    """
    构造 provider 实例（同步对象；async 由 BaseLLM.achat 包装）
    """
    s = settings or LLMSettings()
    if s.provider == "zhipu":
        return ZhipuLLM(ZhipuSettings(model=s.model))
    elif s.provider == "deepseek":
        return DeepSeekLLM(DeepSeekSettings(model=s.model))
    elif s.provider == "dashscope":
        return DashScopeLLM(DashScopeSettings(model=s.model))
    elif s.provider == "openai":
        return OpenAILLM(OpenAISettings(model=s.model))
    raise ValueError(f"Unknown provider: {s.provider}")


#  统一入口（与你 query/generation 更好对齐）
def get_llm(settings: Optional[LLMSettings] = None) -> BaseLLM:
    return build_llm(settings)
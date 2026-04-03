# -*- coding: utf-8 -*-
"""
retrieval/api.py

作者：Accilia
创建时间：2026-02-24
用途说明：
  提供 Retrieval 阶段的统一“可复用API”，供后续生成/服务端调用。

设计原则：
  - 同步：给本地端到端直接用
  - 轻依赖：不加载索引/模型，检索全在服务器完成
  - 统一输入：优先使用 query_bundle dict（由 query/api.py 产出）
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import os
import requests

from config.api_keys import get_api_key

# 从配置文件读取 OpenBayes 服务地址
OPENBAYES_BASE_URL = get_api_key("OPENBAYES_BASE_URL")

# ✅ app.py 的检索端点（固定）
DEFAULT_API_PATH = "/retrieve"


def _join_url(api_path: str) -> str:
    base = OPENBAYES_BASE_URL.rstrip("/")
    path = "/" + (api_path or DEFAULT_API_PATH).lstrip("/")
    return base + path


def _build_headers(api_key: str) -> Dict[str, str]:
    """
    对齐 OpenBayes curl 示例：
      --header 'Content-Type: application/json'
      --header 'Authorization: Bearer sk-xxxx'
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    api_key = (api_key or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def retrieve_chunks_sync(
    query_bundle: Dict[str, Any],
    api_key: str = "",
    api_path: str = DEFAULT_API_PATH,
    final_top_n: int = 10,
    delta: Optional[float] = None,
    return_evidence: bool = True,
    return_debug: bool = False,
    enable_rerank: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    同步：向 OpenBayes /retrieve 发送请求，返回 JSON dict（字段名由 app.py 固定协议决定）
    """
    # API Key：命令行优先，其次环境变量
    api_key = (api_key or "").strip() or os.getenv("OPENBAYES_API_KEY", "").strip()
    headers = _build_headers(api_key)

    url = _join_url(api_path)

    payload = {
        "query_bundle": query_bundle,
        "final_top_n": int(final_top_n),
        "delta": delta,
        "return_evidence": bool(return_evidence),
        "return_debug": bool(return_debug),
        "enable_rerank": bool(enable_rerank),
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=int(timeout))
    if resp.status_code != 200:
        raise RuntimeError(f"[HTTP {resp.status_code}] {url}\n{resp.text[:2000]}")

    try:
        return resp.json()
    except Exception:
        raise RuntimeError(f"[HTTP OK but JSON decode failed]\n{resp.text[:2000]}")
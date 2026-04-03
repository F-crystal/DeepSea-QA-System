# -*- coding: utf-8 -*-
"""
eval/api.py

作者：Accilia
创建时间：2026-02-27
用途说明：
评估系统API接口，通过服务器调用端口的方式获取服务器计算结果。
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from typing import Dict, Any, List, Optional

import requests

from deepsea_qa.configs import paths
from config.api_keys import get_api_key

# 从配置文件读取 OpenBayes 服务地址
OPENBAYES_BASE_URL = get_api_key("OPENBAYES_BASE_URL")

# ✅ app.py 的评估端点（固定）
DEFAULT_API_PATH_COMPUTE_EMBEDDING = "/compute_embedding"
DEFAULT_API_PATH_COMPUTE_SIMILARITY = "/compute_similarity"
DEFAULT_API_PATH_COMPUTE_BERTSCORE = "/compute_bertscore"
DEFAULT_API_PATH_HEALTH = "/health"


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


class EvalAPIClient:
    """评估系统API客户端"""
    
    def __init__(self, base_url: str = OPENBAYES_BASE_URL, api_key: str = None):
        """初始化客户端"""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENBAYES_API_KEY")
    
    def compute_embedding(self, texts: List[str]) -> Dict[str, Any]:
        """计算文本嵌入"""
        url = f"{self.base_url}{DEFAULT_API_PATH_COMPUTE_EMBEDDING}"
        data = {"texts": texts}
        headers = _build_headers(self.api_key)
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("ok"):
                return result
            else:
                print(f"计算嵌入失败: {result.get('error')}")
                return {"embeddings": [], "count": 0}
        except Exception as e:
            print(f"计算嵌入失败: {e}")
            return {"embeddings": [], "count": 0}
    
    def compute_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """计算文本相似度"""
        url = f"{self.base_url}{DEFAULT_API_PATH_COMPUTE_SIMILARITY}"
        data = {"text1": text1, "text2": text2}
        headers = _build_headers(self.api_key)
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("ok"):
                return result
            else:
                print(f"计算相似度失败: {result.get('error')}")
                return {"similarity": 0.0, "text1": text1, "text2": text2}
        except Exception as e:
            print(f"计算相似度失败: {e}")
            return {"similarity": 0.0, "text1": text1, "text2": text2}
    

    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        url = f"{self.base_url}{DEFAULT_API_PATH_HEALTH}"
        headers = _build_headers(self.api_key)
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"健康检查失败: {e}")
            return {"status": "error", "services": {}}
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """计算BERTScore"""
        url = f"{self.base_url}{DEFAULT_API_PATH_COMPUTE_BERTSCORE}"
        data = {"predictions": predictions, "references": references}
        headers = _build_headers(self.api_key)
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("ok"):
                return result
            else:
                print(f"计算BERTScore失败: {result.get('error')}")
                return {"scores": [], "average": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "count": 0}
        except Exception as e:
            print(f"计算BERTScore失败: {e}")
            return {"scores": [], "average": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "count": 0}


def compute_embedding_sync(
    texts: List[str],
    base_url: str = OPENBAYES_BASE_URL,
) -> Dict[str, Any]:
    """同步计算文本嵌入"""
    client = EvalAPIClient(base_url)
    return client.compute_embedding(texts)


def compute_similarity_sync(
    text1: str,
    text2: str,
    base_url: str = OPENBAYES_BASE_URL,
) -> Dict[str, Any]:
    """同步计算文本相似度"""
    client = EvalAPIClient(base_url)
    return client.compute_similarity(text1, text2)



def health_check_sync(
    base_url: str = OPENBAYES_BASE_URL,
) -> Dict[str, Any]:
    """同步健康检查"""
    client = EvalAPIClient(base_url)
    return client.health_check()


def compute_bertscore_sync(
    predictions: List[str],
    references: List[str],
    base_url: str = OPENBAYES_BASE_URL,
) -> Dict[str, Any]:
    """同步计算BERTScore"""
    client = EvalAPIClient(base_url)
    return client.compute_bertscore(predictions, references)

# -*- coding: utf-8 -*-
"""
retrieval/utils.py

作者：Accilia
创建时间：2026-02-23
用途说明：
- 混合分词（中文jieba + 英文regex）
- 停用词加载：
  - 中文：读取 prepare_data/stopwords/ 目录内所有 .txt 文件
  - 英文：sklearn ENGLISH_STOP_WORDS + nltk stopwords（可选，自动下载）
- 预处理：小写化、最小长度过滤、停用词过滤、保留词白名单（可选）

"""

from __future__ import annotations

import os
import re
from typing import List, Set, Optional

import jieba


# 是否包含中文字符（用于启用jieba）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 英文/数字 token（适合学术文本：保留 subsea/rov/auv/3d 等）
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_\-][A-Za-z0-9]+)*")


def load_cn_stopwords_from_dir(stopwords_dir: str) -> Set[str]:
    """
    读取中文停用词目录内所有 .txt（文件名不敏感）
    约定：一行一个词；允许空行；允许 # 注释
    """
    stops: Set[str] = set()
    if not stopwords_dir or not os.path.isdir(stopwords_dir):
        return stops

    for fn in sorted(os.listdir(stopwords_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        fp = os.path.join(stopwords_dir, fn)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if not w or w.startswith("#"):
                        continue
                    stops.add(w)
        except Exception:
            # 某个文件编码异常也不应阻塞整个流程
            continue

    return stops


def load_en_stopwords_sklearn() -> Set[str]:
    """
    直接调用 sklearn 内置英文停用词（工程上最稳定）
    """
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore
        return set(ENGLISH_STOP_WORDS)
    except Exception:
        return set()


def load_en_stopwords_nltk(download_if_missing: bool = True) -> Set[str]:
    """
    直接调用 nltk.corpus.stopwords
    - 若 stopwords 未下载且 download_if_missing=True，会自动联网下载
    """
    try:
        import nltk  # type: ignore
        from nltk.corpus import stopwords  # type: ignore

        try:
            words = stopwords.words("english")
            return set(words)
        except LookupError:
            if not download_if_missing:
                return set()
            # 自动下载
            nltk.download("stopwords")
            words = stopwords.words("english")
            return set(words)
    except Exception:
        return set()


def build_stopwords(
    cn_stopwords_dir: str,
    use_sklearn_en: bool = True,
    use_nltk_en: bool = True,
    nltk_download_if_missing: bool = True,
) -> Set[str]:
    """
    总停用词表 = 中文目录停用词 ∪ 英文停用词（sklearn + nltk 可选）
    """
    stops = set()
    stops |= load_cn_stopwords_from_dir(cn_stopwords_dir)

    if use_sklearn_en:
        stops |= load_en_stopwords_sklearn()
    if use_nltk_en:
        stops |= load_en_stopwords_nltk(download_if_missing=nltk_download_if_missing)

    # 统一小写（英文停用词）——中文不受影响
    stops = {w.lower() if _EN_TOKEN_RE.fullmatch(w) else w for w in stops}
    return stops


def mixed_tokenize(
    text: str,
    stopwords: Set[str],
    min_token_len: int = 2,
    max_tokens: int = 4096,
    keep_terms: Optional[Set[str]] = None,
) -> List[str]:
    """
    混合分词（BM25用）：
    - 如果文本含中文：jieba分词 + 额外抽取英文token（防止术语丢失）
    - 如果文本不含中文：仅英文regex token
    - 英文 token 小写化
    - 停用词过滤 + 最小长度过滤
    - keep_terms：保留词白名单（不会被停用词过滤）
    """
    if not text:
        return []

    text = str(text).strip()
    if not text:
        return []

    keep_terms = keep_terms or set()

    has_cjk = bool(_CJK_RE.search(text))
    tokens: List[str] = []

    if has_cjk:
        zh_tokens = [t.strip() for t in jieba.lcut(text, cut_all=False) if t.strip()]
        en_tokens = _EN_TOKEN_RE.findall(text)
        tokens = zh_tokens + en_tokens
    else:
        tokens = _EN_TOKEN_RE.findall(text)

    out: List[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue

        # 英文小写化
        if _EN_TOKEN_RE.fullmatch(t):
            t = t.lower()

        # 最小长度过滤（例如 "a"、"b" 这种噪声）
        if len(t) < min_token_len and t not in keep_terms:
            continue

        # 停用词过滤（保留词白名单优先）
        if t in stopwords and t not in keep_terms:
            continue

        out.append(t)
        if len(out) >= max_tokens:
            break

    return out
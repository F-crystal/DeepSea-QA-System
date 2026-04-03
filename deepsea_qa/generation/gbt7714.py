# -*- coding: utf-8 -*-
"""
gbt7714.py

作者：Accilia
创建时间：2026-02-25
用途说明：
GB/T 7714-2015：
作者. 题名[J]. 刊名, 年, 卷(期): 页码. DOI: xxx

缺字段则省略，不阻塞主流程。
"""

from __future__ import annotations

from typing import List
from deepsea_qa.generation.types import PaperMeta


def _norm_authors(s: str) -> str:
    """规范化作者姓名格式"""
    s = (s or "").strip()
    s = s.replace("；", ";").replace("，", ",")
    
    # 按";"拆分作者
    authors_list = [author.strip() for author in s.split(";") if author.strip()]
    
    # 处理每个作者的姓名格式
    processed_authors = []
    for author in authors_list:
        # 处理姓名中的","分隔符，将其替换为空格
        name_parts = [part.strip() for part in author.split(",") if part.strip()]
        # 转成全大写
        processed_name = " ".join(name_parts).upper()
        processed_authors.append(processed_name)
    
    # 处理作者数量，超过3人时只保留前3人，使用"et al"
    if len(processed_authors) > 3:
        s = ", ".join(processed_authors[:3]) + ", et al"
    else:
        s = ", ".join(processed_authors)
    
    return s


def _title_case(s: str) -> str:
    """将字符串转换为首字母大写格式"""
    if not s:
        return s
    words = s.split()
    capitalized_words = [word.capitalize() for word in words]
    return " ".join(capitalized_words)


def format_gbt_7714(meta: PaperMeta) -> str:
    """格式化参考文献为GB/T 7714-2015标准格式"""
    authors = _norm_authors(meta.authors)
    title = (meta.title or "").strip()
    venue = (meta.venue or "").strip()
    # 期刊名首字母大写
    venue = _title_case(venue)
    year = "" if meta.year is None else str(meta.year)
    vol = (meta.volume or "").strip()
    iss = (meta.issue or "").strip()
    pages = (meta.pages or "").strip()
    doi = (meta.doi or "").strip()

    # 确定文献类型标识
    # 默认使用[J]，因为大多数文献是期刊论文
    doc_type = "[J]"

    vi = ""
    if vol and iss:
        vi = f"{vol}({iss})"
    elif vol:
        vi = vol
    elif iss:
        vi = f"({iss})"

    parts: List[str] = []
    
    # 作者
    if authors:
        parts.append(f"{authors}.")
    
    # 题名和文献类型
    if title:
        parts.append(f"{title}{doc_type}.")
    else:
        parts.append(f"{doc_type}.")

    # 刊名/会议名、年份、卷期
    tail = []
    if venue:
        tail.append(venue)
    if year:
        tail.append(year)
    if vi:
        tail.append(vi)

    if tail:
        line = ", ".join(tail)
        if pages:
            line += f": {pages}."
        else:
            # 如果没有页码，也要在末尾添加点号
            line += "."
        parts.append(line)
    else:
        if pages:
            parts.append(f"{pages}.")

    # DOI
    if doi:
        parts.append(f"DOI: {doi}.")

    return " ".join([p for p in parts if p]).strip()
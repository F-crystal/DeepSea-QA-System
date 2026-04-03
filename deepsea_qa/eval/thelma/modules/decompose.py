# -*- coding: utf-8 -*-
"""
decompose.py

作者：Accilia
创建时间：2026-02-27
用途说明：
分解模块 (Decompose Module)
实现THELMA框架中的各种分解策略
"""

from typing import List, Optional

from deepsea_qa.llm.base import BaseLLM


class DecomposeModule:
    """分解模块"""
    
    def __init__(self, llm_client: Optional[BaseLLM] = None):
        self.llm_client = llm_client
    
    def decompose_text(self, text: str) -> List[str]:
        """分解文本为原子事实 (D_{text})
        
        Args:
            text: 输入文本
            
        Returns:
            原子事实列表
        """
        if not text:
            return []
        
        # 使用LLM提取原子事实
        if self.llm_client:
            return self._decompose_text_with_llm(text)
        else:
            # 降级使用规则分割
            return self._decompose_text_with_rules(text)
    
    def decompose_query(self, query: str) -> List[str]:
        """分解查询为子问题 (D_{qcov})
        
        Args:
            query: 输入查询
            
        Returns:
            子问题列表
        """
        if not query:
            return []
        
        # 使用LLM分解查询
        if self.llm_client:
            return self._decompose_query_with_llm(query)
        else:
            # 降级使用规则分割
            return self._decompose_query_with_rules(query)
    
    def decompose_sentences(self, text: str) -> List[str]:
        """分解文本为句子 (D_{sen})
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        if not text:
            return []
        
        # 使用规则分割
        return self._decompose_sentences_with_rules(text)
    
    def identity_decompose(self, text: str) -> List[str]:
        """恒等分解 (D_{id})
        
        Args:
            text: 输入文本
            
        Returns:
            包含原始文本的列表
        """
        if not text:
            return []
        return [text]
    
    def _decompose_text_with_llm(self, text: str) -> List[str]:
        """使用LLM提取原子事实"""
        prompt = f"""你是一个专业的声明提取器 (Claim Extractor)。请将以下文本拆解为独立的、可独立验证的原子事实（stand-alone claims）。

要求：
1. 每个声明只包含一件独立的信息。
2. 声明之间应尽可能少重叠（independent）。
3. 必须覆盖输入文本中的所有信息。
4. 输出格式为列表，每个元素是一个原子事实。

输入文本：
{text}

输出："""
        
        response = self.llm_client.generate(prompt)
        # 解析LLM输出
        claims = self._parse_llm_list_output(response.content)
        return claims
    
    def _decompose_query_with_llm(self, query: str) -> List[str]:
        """使用LLM分解查询"""
        prompt = f"""你是一个专业的问题分解器 (Question Decomposer)。请将以下查询拆解为多个独立的子问题。

要求：
1. 解决代词指代问题（Resolve ambiguous pronouns），使每个子问题独立可读。
2. 忽略问候语或非问题的陈述句。
3. 将复合问题拆分为多个独立的短问题。
4. 输出格式为问题列表。

输入查询：
{query}

输出："""
        
        response = self.llm_client.generate(prompt)
        # 解析LLM输出
        sub_questions = self._parse_llm_list_output(response.content)
        return sub_questions
    
    def _decompose_text_with_rules(self, text: str) -> List[str]:
        """使用规则分割文本为原子事实"""
        claims = []
        
        # 按中文句号分割
        sentences = text.split('。')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 进一步分割复杂句子
                if '，' in sentence:
                    parts = sentence.split('，')
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part:
                            if i < len(parts) - 1:
                                claims.append(part + '，')
                            else:
                                claims.append(part)
                else:
                    claims.append(sentence)
        
        if not claims:
            claims.append(text)
        
        return claims
    
    def _decompose_query_with_rules(self, query: str) -> List[str]:
        """使用规则分割查询为子问题"""
        sub_questions = []
        
        # 按常见的中文分隔符分割
        separators = ['。', '？', '！', '；', '；']
        temp_query = query
        
        # 优先使用中文分隔符
        for sep in separators:
            if sep in temp_query:
                parts = temp_query.split(sep)
                for part in parts:
                    part = part.strip()
                    if part:
                        sub_questions.append(part + sep)
                break
        
        # 如果中文分隔符没有分割成功，尝试英文分隔符
        if not sub_questions:
            eng_separators = ['.', '?', '!', ';']
            for sep in eng_separators:
                if sep in temp_query:
                    parts = temp_query.split(sep)
                    for part in parts:
                        part = part.strip()
                        if part:
                            sub_questions.append(part + sep)
                    break
        
        if not sub_questions:
            sub_questions.append(query)
        
        return sub_questions
    
    def _decompose_sentences_with_rules(self, text: str) -> List[str]:
        """使用规则分割文本为句子"""
        sentences = []
        current_sentence = []
        
        # 优先使用中文标点符号
        chinese_punctuation = ['。', '！', '？', '；']
        # 英文标点符号
        english_punctuation = ['.', '!', '?', ';']
        
        for char in text:
            current_sentence.append(char)
            if char in chinese_punctuation or char in english_punctuation:
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []
        
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    def _parse_llm_list_output(self, output: str) -> List[str]:
        """解析LLM输出的列表"""
        import re
        
        # 尝试提取列表
        # 匹配带编号的列表
        numbered_items = re.findall(r'\d+\.\s*(.+?)\s*(?=\d+\.|$)', output, re.DOTALL)
        if numbered_items:
            return [item.strip() for item in numbered_items]
        
        # 匹配带破折号的列表
        dashed_items = re.findall(r'-\s*(.+?)\s*(?=-|$)', output, re.DOTALL)
        if dashed_items:
            return [item.strip() for item in dashed_items]
        
        # 匹配带星号的列表
        starred_items = re.findall(r'\*\s*(.+?)\s*(?=\*|$)', output, re.DOTALL)
        if starred_items:
            return [item.strip() for item in starred_items]
        
        # 尝试按换行符分割
        lines = output.strip().split('\n')
        if len(lines) > 1:
            return [line.strip() for line in lines if line.strip()]
        
        # 兜底：返回原始文本
        return [output.strip()]

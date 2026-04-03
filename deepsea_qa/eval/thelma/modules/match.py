# -*- coding: utf-8 -*-
"""
match.py

作者：Accilia
创建时间：2026-02-27
用途说明：
匹配模块 (Match Module)
实现THELMA框架中的各种匹配逻辑
"""

from typing import List, Optional

from deepsea_qa.llm.base import BaseLLM


class MatchModule:
    """匹配模块"""
    
    def __init__(self, llm_client: Optional[BaseLLM] = None):
        self.llm_client = llm_client
    
    def judge_essentiality(self, query: str, text: str) -> bool:
        """判断文本对于回答查询是否必要 (m_{sp}, m_{rp})
        
        Args:
            query: 用户查询
            text: 待评估文本
            
        Returns:
            是否必要
        """
        if not query or not text:
            return False
        
        # 使用LLM判断必要性
        if self.llm_client:
            llm_result = self._judge_essentiality_with_llm(query, text)
            # 如果LLM判断为非必要，使用相似度作为补充判断
            if not llm_result:
                similarity_result = self._judge_essentiality_with_similarity(query, text)
                return similarity_result
            return llm_result
        else:
            # 降级使用相似度判断
            return self._judge_essentiality_with_similarity(query, text)
    
    def judge_support(self, claim: str, sources: List[str]) -> bool:
        """判断声明是否被源支持 (m_{gr})
        
        Args:
            claim: 待验证声明
            sources: 源文本列表
            
        Returns:
            是否被支持
        """
        if not claim or not sources:
            return False
        
        # 使用LLM判断支持性
        if self.llm_client:
            llm_result = self._judge_support_with_llm(claim, sources)
            # 如果LLM判断为不支持，使用相似度作为补充判断
            if not llm_result:
                similarity_result = self._judge_support_with_similarity(claim, sources)
                return similarity_result
            return llm_result
        else:
            # 降级使用相似度判断
            return self._judge_support_with_similarity(claim, sources)
    
    def contains_answer(self, sub_query: str, source: str) -> bool:
        """判断源是否包含子查询的答案 (m_{sqcov})
        
        Args:
            sub_query: 子查询
            source: 源文本
            
        Returns:
            是否包含答案
        """
        if not sub_query or not source:
            return False
        
        # 使用LLM判断答案存在性
        if self.llm_client:
            llm_result = self._contains_answer_with_llm(sub_query, source)
            # 如果LLM判断为不包含，使用相似度作为补充判断
            if not llm_result:
                similarity_result = self._contains_answer_with_similarity(sub_query, source)
                return similarity_result
            return llm_result
        else:
            # 降级使用相似度判断
            return self._contains_answer_with_similarity(sub_query, source)
    
    def covers_intent(self, sub_query: str, answer: str) -> bool:
        """验证回答是否解决了子问题的意图 (m_{rqcov})
        
        Args:
            sub_query: 子查询
            answer: 回答
            
        Returns:
            是否覆盖意图
        """
        if not sub_query or not answer:
            return False
        
        # 使用LLM判断意图覆盖
        if self.llm_client:
            llm_result = self._covers_intent_with_llm(sub_query, answer)
            # 如果LLM判断为不覆盖，使用相似度作为补充判断
            if not llm_result:
                similarity_result = self._covers_intent_with_similarity(sub_query, answer)
                return similarity_result
            return llm_result
        else:
            # 降级使用相似度判断
            return self._covers_intent_with_similarity(sub_query, answer)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数
        """
        from deepsea_qa.eval.api import compute_similarity_sync
        
        try:
            result = compute_similarity_sync(text1, text2)
            return result.get('similarity', 0.0)
        except Exception as e:
            print(f"相似度计算失败: {e}")
            # 降级使用字符级相似度
            import difflib
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            return similarity
    
    def _judge_essentiality_with_llm(self, query: str, text: str) -> bool:
        """使用LLM判断必要性"""
        prompt = f"""你是一个专业的编辑 (Editor)。请判断以下文本对于回答用户查询是否是必要的 (Essential)。

用户查询：
{query}

待评估文本：
{text}

判断标准：
- 1 (Essential)：该信息是回答核心意图所必需的。如果去掉该信息，回答会不完整或不准确。
- 0 (Extraneous)：该信息虽可能相关，但对于回答核心问题是不必要的冗余细节，属于锦上添花的内容。

请仅返回 "1" 或 "0"，不要添加任何其他内容。

输出："""
        
        response = self.llm_client.generate(prompt)
        return "1" in response.content
    
    def _judge_support_with_llm(self, claim: str, sources: List[str]) -> bool:
        """使用LLM判断支持性"""
        sources_text = '\n'.join(sources)
        prompt = f"""你是一个专业的事实核查员 (Fact Checker)。请严格基于以下源集合，判断给定的声明是否被支持 (Supported)。

源集合：
{sources_text}

待验证声明：
{claim}

判断标准：
- 1 (Supported)：声明肯定被源支持，源中有明确的证据。
- 0 (Not Supported)：声明未被源支持，源中找不到证据，或声明与源矛盾。

注意：请严格依赖提供的源作为“真理”，不考虑模型内部知识。即使该事实在现实中是真的，只要源里没写，也算未被支持。

请仅返回 "1" 或 "0"，不要添加任何其他内容。

输出："""
        
        response = self.llm_client.generate(prompt)
        return "1" in response.content
    
    def _contains_answer_with_llm(self, sub_query: str, source: str) -> bool:
        """使用LLM判断答案存在性"""
        prompt = f"""请判断以下源文本是否包含回答子问题的信息。

子问题：
{sub_query}

源文本：
{source}

判断标准：
- 1 (Contains)：源文本包含回答子问题的相关信息，即使不是非常明确。
- 0 (Not Contains)：源文本完全不包含与子问题相关的信息。

请仅返回 "1" 或 "0"，不要添加任何其他内容。

输出："""
        
        response = self.llm_client.generate(prompt)
        return "1" in response.content
    
    def _covers_intent_with_llm(self, sub_query: str, answer: str) -> bool:
        """使用LLM判断意图覆盖"""
        prompt = f"""请判断以下回答是否在逻辑上和事实上解决了子问题的意图。

子问题：
{sub_query}

回答：
{answer}

判断标准：
- 1 (Covered)：回答涵盖了该子问题的核心意图，提供了有价值的信息。
- 0 (Not Covered)：回答完全没有涉及子问题的核心意图。

请仅返回 "1" 或 "0"，不要添加任何其他内容。

输出："""
        
        response = self.llm_client.generate(prompt)
        return "1" in response.content
    
    def _judge_essentiality_with_similarity(self, query: str, text: str) -> bool:
        """使用相似度判断必要性"""
        similarity = self.compute_similarity(query, text)
        return similarity >= 0.5
    
    def _judge_support_with_similarity(self, claim: str, sources: List[str]) -> bool:
        """使用相似度判断支持性"""
        for source in sources:
            similarity = self.compute_similarity(claim, source)
            if similarity >= 0.5:
                return True
        return False
    
    def _contains_answer_with_similarity(self, sub_query: str, source: str) -> bool:
        """使用相似度判断答案存在性"""
        similarity = self.compute_similarity(sub_query, source)
        return similarity >= 0.5
    
    def _covers_intent_with_similarity(self, sub_query: str, answer: str) -> bool:
        """使用相似度判断意图覆盖"""
        similarity = self.compute_similarity(sub_query, answer)
        return similarity >= 0.5

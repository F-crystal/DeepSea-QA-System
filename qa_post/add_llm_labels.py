# -*- coding: utf-8 -*-
"""
add_llm_labels.py

作者：Accilia
创建时间：2026-02-27
用途说明：为qa_filtered.jsonl添加llm_label标签，基于domain和paper_id匹配
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import json
import os
import pandas as pd
from pathlib import Path

# 工作区根目录
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# 输入文件路径
QA_FILTERED_PATH = WORKSPACE_ROOT / "qa_post" / "qa_filtered.jsonl"

# 输出文件路径
QA_WITH_LABELS_PATH = WORKSPACE_ROOT / "qa_post" / "qa_filtered_with_labels.jsonl"

# llm_labeled文件路径
LLM_LABELED_DIR = WORKSPACE_ROOT / "prepare_data" / "题录信息_中间结果"

# 方向映射
domain_map = {
    "深海矿产": "深海矿产",
    "深海可再生能源": "深海可再生能源",
    "深海感知与通信装备": "深海感知与通信装备",
    "深水油气": "深水油气"
}

def load_llm_labels():
    """加载llm_labeled文件，构建paper_id到llm_label和secondary_labels的映射
    处理同一paper_id在多个领域出现的情况，基于置信度选择主领域和主标签
    """
    # 存储每个paper_id的所有领域信息
    paper_domains = {}
    
    for domain_zh, domain_key in domain_map.items():
        xlsx_path = LLM_LABELED_DIR / f"llm_labeled_{domain_key}.xlsx"
        if not xlsx_path.exists():
            print(f"警告：文件不存在: {xlsx_path}")
            continue
        
        try:
            df = pd.read_excel(xlsx_path)
            # 打印列名，了解文件结构
            print(f"{domain_key} 的列名: {list(df.columns)}")
            
            # 尝试不同的列名
            paper_id_columns = ['paper_id', 'id', 'Paper ID', 'paperId']
            label_columns = ['primary_label_name_zh', 'llm_label', 'label', 'LLM Label', 'llmLabel']
            secondary_label_columns = ['secondary_label_names_zh', 'secondary_labels', 'secondary_label', 'secondary_labels_zh', '副标签']
            confidence_columns = ['confidence', 'conf', 'score', '置信度']
            
            paper_id_col = None
            label_col = None
            secondary_label_col = None
            confidence_col = None
            
            for col in paper_id_columns:
                if col in df.columns:
                    paper_id_col = col
                    break
            
            for col in label_columns:
                if col in df.columns:
                    label_col = col
                    break
            
            for col in secondary_label_columns:
                if col in df.columns:
                    secondary_label_col = col
                    break
            
            for col in confidence_columns:
                if col in df.columns:
                    confidence_col = col
                    break
            
            if not paper_id_col or not label_col:
                print(f"警告：{domain_key} 文件中没有找到合适的列名")
                continue
            
            # 构建映射
            for _, row in df.iterrows():
                paper_id = str(row.get(paper_id_col, '')).strip()
                llm_label = str(row.get(label_col, '')).strip()
                
                # 获取置信度
                confidence = 0.0
                if confidence_col:
                    try:
                        confidence = float(row.get(confidence_col, 0.0))
                    except:
                        pass
                
                # 获取副标签
                secondary_labels = []
                if secondary_label_col:
                    secondary_label_value = row.get(secondary_label_col, '')
                    if secondary_label_value:
                        # 处理不同格式的副标签
                        if isinstance(secondary_label_value, str):
                            # 尝试按分隔符分割
                            if ';' in secondary_label_value:
                                secondary_labels = [label.strip() for label in secondary_label_value.split(';') if label.strip()]
                            elif ',' in secondary_label_value:
                                secondary_labels = [label.strip() for label in secondary_label_value.split(',') if label.strip()]
                            else:
                                secondary_labels = [secondary_label_value.strip()]
                
                if paper_id and llm_label:
                    if paper_id not in paper_domains:
                        paper_domains[paper_id] = []
                    
                    paper_domains[paper_id].append({
                        'domain': domain_zh,
                        'llm_label': llm_label,
                        'secondary_labels': secondary_labels,
                        'confidence': confidence
                    })
            
            print(f"成功加载 {domain_key} 的llm_label，共 {len([p for p in paper_domains if any(d['domain'] == domain_zh for d in paper_domains[p])])} 条")
        except Exception as e:
            print(f"错误：读取 {xlsx_path} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 为每个paper_id选择主领域和主标签（基于置信度）
    label_map = {}
    for paper_id, domains_info in paper_domains.items():
        # 按置信度排序
        domains_info.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 主领域和主标签
        primary_domain = domains_info[0]['domain']
        primary_label = domains_info[0]['llm_label']
        
        # 收集所有副标签（排除主标签）
        # 包括：1. 各领域的副标签 2. 其他领域的主标签
        all_secondary_labels = []
        secondary_domains = []
        
        for info in domains_info:
            if info['domain'] != primary_domain:
                secondary_domains.append(info['domain'])
                # 将其他领域的主标签作为副标签
                if info['llm_label'] != primary_label and info['llm_label'] not in all_secondary_labels:
                    all_secondary_labels.append(info['llm_label'])
            # 收集各领域的副标签
            for label in info['secondary_labels']:
                if label != primary_label and label not in all_secondary_labels:
                    all_secondary_labels.append(label)
        
        label_map[paper_id] = {
            'domain': primary_domain,
            'llm_label': primary_label,
            'secondary_labels': all_secondary_labels,
            'secondary_domains': secondary_domains,
            'all_domains': [info['domain'] for info in domains_info],
            'confidence': domains_info[0]['confidence']
        }
    
    return label_map

def add_labels_to_qa():
    """为qa_filtered.jsonl添加llm_label和secondary_labels标签"""
    # 加载llm_label映射
    label_map = load_llm_labels()
    print(f"总共加载 {len(label_map)} 个paper_id的标签信息")
    
    # 处理qa_filtered.jsonl
    processed_count = 0
    labeled_count = 0
    secondary_labels_count = 0
    
    with open(QA_FILTERED_PATH, 'r', encoding='utf-8') as infile, \
         open(QA_WITH_LABELS_PATH, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                meta = data.get('meta', {})
                paper_id = meta.get('paper_id', '').strip()
                
                # 添加llm_label和secondary_labels
                if paper_id in label_map:
                    label_info = label_map[paper_id]
                    meta['domain'] = label_info.get('domain', '')
                    meta['llm_label'] = label_info.get('llm_label', '')
                    meta['secondary_labels'] = label_info.get('secondary_labels', [])
                    meta['secondary_domains'] = label_info.get('secondary_domains', [])
                    meta['all_domains'] = label_info.get('all_domains', [])
                    meta['confidence'] = label_info.get('confidence', 0.0)
                    labeled_count += 1
                    if label_info.get('secondary_labels'):
                        secondary_labels_count += 1
                else:
                    meta['domain'] = ""
                    meta['llm_label'] = ""
                    meta['secondary_labels'] = []
                    meta['secondary_domains'] = []
                    meta['all_domains'] = []
                    meta['confidence'] = 0.0
                
                data['meta'] = meta
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
            except Exception as e:
                print(f"错误：处理行失败: {e}")
                continue
    
    print(f"处理完成：共 {processed_count} 条数据，其中 {labeled_count} 条添加了llm_label，{secondary_labels_count} 条添加了secondary_labels")
    print(f"输出文件：{QA_WITH_LABELS_PATH}")

if __name__ == "__main__":
    add_labels_to_qa()

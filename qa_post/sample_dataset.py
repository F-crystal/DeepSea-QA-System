# -*- coding: utf-8 -*-
"""
sample_dataset.py

作者：Accilia
创建时间：2026-02-28
用途说明：
  从问答数据集中进行采样：
  - 4个方向（domain）
  - 每个方向11个类别（加上other）
  - 每个类别随机抽取15-20条（min=15，max=20）
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import json
import random
from collections import defaultdict


def load_dataset(file_path):
    """加载数据集"""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                dataset.append(data)
            except Exception as e:
                print(f"错误：加载数据失败: {e}")
                continue
    return dataset


def analyze_dataset(dataset):
    """分析数据集分布"""
    # 统计每个domain和label的分布
    domain_label_count = defaultdict(lambda: defaultdict(int))
    
    for data in dataset:
        domain = data.get('meta', {}).get('domain', 'unknown')
        label = data.get('meta', {}).get('llm_label', 'unknown')
        domain_label_count[domain][label] += 1
    
    # 打印统计结果
    print("数据集分布统计：")
    print("=" * 80)
    
    total = len(dataset)
    print(f"总数据量: {total}")
    print()
    
    for domain, label_counts in domain_label_count.items():
        domain_total = sum(label_counts.values())
        print(f"领域: {domain} (总数: {domain_total})")
        print(f"  类别数: {len(label_counts)}")
        
        # 按数量排序
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            print(f"    {label}: {count}")
        print()
    
    return domain_label_count


def sample_dataset(dataset, domain_label_count, min_samples=15, max_samples=20):
    """采样数据集"""
    # 为每个domain和label创建数据列表
    domain_label_data = defaultdict(lambda: defaultdict(list))
    for data in dataset:
        domain = data.get('meta', {}).get('domain', 'unknown')
        label = data.get('meta', {}).get('llm_label', 'unknown')
        domain_label_data[domain][label].append(data)
    
    # 采样结果
    sampled_data = []
    
    # 遍历每个domain
    for domain, label_data in domain_label_data.items():
        print(f"\n处理领域: {domain}")
        print(f"  原始类别数: {len(label_data)}")
        
        # 确保至少有11个类别（加上other）
        # 如果类别数不足11，我们会将一些小类别合并到other
        labels = list(label_data.keys())
        
        # 按数据量排序
        labels.sort(key=lambda x: len(label_data[x]), reverse=True)
        
        # 选择前10个类别，剩下的合并到other
        selected_labels = labels[:10]
        other_labels = labels[10:]
        
        # 处理前10个类别
        for label in selected_labels:
            data_list = label_data[label]
            count = len(data_list)
            sample_size = min(max_samples, max(1, count))  # 至少采样1条
            sampled = random.sample(data_list, sample_size)
            sampled_data.extend(sampled)
            print(f"  类别 '{label}': 原始 {count}，采样 {sample_size}")
        
        # 处理other类别
        if other_labels:
            other_data = []
            for label in other_labels:
                other_data.extend(label_data[label])
            
            count = len(other_data)
            sample_size = min(max_samples, max(1, count))  # 至少采样1条
            if sample_size > 0:
                sampled = random.sample(other_data, sample_size)
                # 将other类别的数据的llm_label设置为'其他'
                for item in sampled:
                    item['meta']['llm_label'] = '其他'
                sampled_data.extend(sampled)
                print(f"  类别 '其他': 原始 {count}，采样 {sample_size}")
    
    print(f"\n采样完成！总采样数: {len(sampled_data)}")
    return sampled_data


def save_dataset(dataset, output_path):
    """保存采样后的数据集"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    print(f"采样结果已保存至: {output_path}")


def main():
    # 输入文件路径
    input_path = 'qa_post/qa_filtered_with_labels.jsonl'
    # 输出文件路径
    output_path = 'qa_post/qa_sampled_dataset.jsonl'
    
    # 加载数据集
    print(f"加载数据集: {input_path}")
    dataset = load_dataset(input_path)
    
    # 分析数据集
    domain_label_count = analyze_dataset(dataset)
    
    # 采样数据集
    sampled_data = sample_dataset(dataset, domain_label_count)
    
    # 保存采样结果
    save_dataset(sampled_data, output_path)


if __name__ == "__main__":
    main()

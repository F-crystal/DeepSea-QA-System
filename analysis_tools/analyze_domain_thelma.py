"""
analyze_domain_thelma.py
作者：Accilia
创建时间：2026-03-27
用途说明：
分析不同领域的Thelma指标
"""
import pandas as pd
import numpy as np
import os

# 基础目录
base_dir = '/Users/fengran/Desktop/5 毕业论文'
data_dir = os.path.join(base_dir, 'analysis_tools', 'data')

# 模型目录
model_dirs = {
    'GLM': 'qa_eval_output/glm-4-plus/qa_eval_results_glm_deduplicated.xlsx',
    'GPT': 'qa_eval_output/gpt-4o/qa_eval_results_gpt_deduplicated.xlsx',
    'Qwen': 'qa_eval_output/qwen-plus/qa_eval_results_qwen_deduplicated.xlsx'
}

# Thelma指标
thelma_metrics = {
    'source_precision_sp1': 'SP1',
    'source_precision_sp2': 'SP2',
    'groundedness': 'Groundedness',
    'source_query_coverage': 'SQC',
    'response_query_coverage': 'RQC',
    'response_precision': 'RP',
    'response_self_distinctness': 'RSD'
}

# 读取所有数据
all_data = []
for model_name, file_path in model_dirs.items():
    full_path = os.path.join(base_dir, file_path)
    df = pd.read_excel(full_path)
    # 只保留需要的列
    columns = ['ground_truth_domain'] + list(thelma_metrics.keys())
    model_data = df[columns].copy()
    model_data['Model'] = model_name
    all_data.append(model_data)

# 合并数据
data_df = pd.concat(all_data, ignore_index=True)

# 按领域分组计算均值（不区分模型）
domain_thelma = data_df.groupby('ground_truth_domain')[list(thelma_metrics.keys())].mean()

# 重命名列名
domain_thelma = domain_thelma.rename(columns=thelma_metrics)

# 保存结果到Excel
output_excel = os.path.join(data_dir, 'domain_thelma_metrics.xlsx')
domain_thelma.to_excel(output_excel)
print(f'领域Thelma指标结果已保存到: {output_excel}')

# 显示结果
print('\n领域Thelma指标结果:')
print(domain_thelma)

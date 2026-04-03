"""
analyze_response_time.py
作者：Accilia
创建时间：2026-03-27
用途说明：
分析不同模型的响应时间分布
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

# 读取所有数据
all_data = []
for model_name, file_path in model_dirs.items():
    full_path = os.path.join(base_dir, file_path)
    df = pd.read_excel(full_path)
    # 只保留需要的列
    if 'total_time' in df.columns:
        model_data = df[['total_time']].copy()
        model_data['Model'] = model_name
        all_data.append(model_data)

# 合并数据
data_df = pd.concat(all_data, ignore_index=True)

# 计算统计信息
stats = []
print('\n响应时间统计信息:')
for model_name in model_dirs.keys():
    model_data = data_df[data_df['Model'] == model_name]['total_time']
    mean_time = model_data.mean()
    median_time = model_data.median()
    p95_time = np.percentile(model_data, 95)
    min_time = model_data.min()
    max_time = model_data.max()
    
    if model_name == 'GPT':
        # 对于GPT，使用第二大值
        sorted_times = sorted(model_data, reverse=True)
        second_max = sorted_times[1] if len(sorted_times) > 1 else sorted_times[0]
        print(f'{model_name}: 平均值={mean_time:.2f}秒, 中位数={median_time:.2f}秒, P95={p95_time:.2f}秒, 最小值={min_time:.2f}秒, 第二大值={second_max:.2f}秒 (最大值={max_time:.2f}秒)')
    else:
        print(f'{model_name}: 平均值={mean_time:.2f}秒, 中位数={median_time:.2f}秒, P95={p95_time:.2f}秒, 最小值={min_time:.2f}秒, 最大值={max_time:.2f}秒)')
    
    # 保存统计信息
    stats.append({
        '模型': model_name,
        '平均用时(秒)': mean_time,
        '中位数(秒)': median_time,
        'P95(秒)': p95_time,
        '最小值(秒)': min_time,
        '最大值(秒)': max_time
    })

# 转换为DataFrame并保存到Excel
stats_df = pd.DataFrame(stats)
output_excel = os.path.join(data_dir, 'response_time_stats.xlsx')
stats_df.to_excel(output_excel, index=False)
print(f'\n响应时间统计结果已保存到: {output_excel}')

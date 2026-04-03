"""
analyze_domain_metrics.py
作者：Accilia
创建时间：2026-03-27
用途说明：
分析不同领域的评估指标
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

# 颜色配置
model_colors = {
    'GLM': '#F17256',   # 橙红（主模型）
    'GPT': '#3B87C7',   # 蓝色（次模型）
    'Qwen': '#69BD44',  # 绿色（次模型）
}

# 配置字体
matplotlib.rcParams.update({
    'font.sans-serif': ['Songti SC', 'STSong', 'SimSun', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 基础目录
base_dir = '/Users/fengran/Desktop/5 毕业论文'
data_dir = os.path.join(base_dir, 'analysis_tools', 'data')
pic_dir = os.path.join(base_dir, 'analysis_tools', 'pics')
os.makedirs(pic_dir, exist_ok=True)

# 模型目录
model_dirs = {
    'GLM': 'qa_eval_output/glm-4-plus/qa_eval_results_glm_deduplicated.xlsx',
    'GPT': 'qa_eval_output/gpt-4o/qa_eval_results_gpt_deduplicated.xlsx',
    'Qwen': 'qa_eval_output/qwen-plus/qa_eval_results_qwen_deduplicated.xlsx'
}

# 可计算的数值指标
numeric_columns = [
    'total_time', 'source_precision_sp1', 'source_precision_sp2', 'groundedness',
    'source_query_coverage', 'response_query_coverage', 'response_precision',
    'response_self_distinctness', 'classification_accuracy', 'rank_score',
    'weighted_score', 'domain_correct', 'label_correct', 'bleu_score',
    'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1', 'answer_length', 'ground_truth_length',
    'bert_score_precision', 'bert_score_recall', 'bert_score_f1'
]

# 指标名称映射
metric_names = {
    'total_time': '总时间',
    'source_precision_sp1': 'SP1',
    'source_precision_sp2': 'SP2',
    'groundedness': 'Groundedness',
    'source_query_coverage': 'SQC',
    'response_query_coverage': 'RQC',
    'response_precision': 'RP',
    'response_self_distinctness': 'SD',
    'classification_accuracy': '分类准确率',
    'rank_score': '排序分数',
    'weighted_score': '加权分数',
    'domain_correct': '领域正确性',
    'label_correct': '标签正确性',
    'bleu_score': 'BLEU',
    'rouge_1_f1': 'ROUGE-1',
    'rouge_2_f1': 'ROUGE-2',
    'rouge_l_f1': 'ROUGE-L',
    'answer_length': '回答长度',
    'ground_truth_length': '参考答案长度',
    'bert_score_precision': 'BERTScore-P',
    'bert_score_recall': 'BERTScore-R',
    'bert_score_f1': 'BERTScore-F1'
}

# 读取并按领域分组计算均值
domain_results = {}
for model_name, file_path in model_dirs.items():
    full_path = os.path.join(base_dir, file_path)
    df = pd.read_excel(full_path)
    
    # 按ground_truth_domain分组计算均值
    grouped = df.groupby('ground_truth_domain')[numeric_columns].mean()
    # 重命名列名
    grouped = grouped.rename(columns=metric_names)
    domain_results[model_name] = grouped

# 合并结果到一个DataFrame
all_results = []
for model_name, results in domain_results.items():
    model_df = results.reset_index()
    model_df['Model'] = model_name
    all_results.append(model_df)

combined_df = pd.concat(all_results, ignore_index=True)

# 保存结果到excel
output_excel = os.path.join(data_dir, 'domain_metrics.xlsx')
combined_df.to_excel(output_excel, index=False)
print(f'领域指标结果已保存到: {output_excel}')

# 准备绘制Weighted Score的分方向柱状图
# 获取所有唯一的领域
domains = sorted(combined_df['ground_truth_domain'].unique())
models = ['GLM', 'GPT', 'Qwen']

# 提取Weighted Score数据
weighted_score_data = {}
for domain in domains:
    weighted_score_data[domain] = []
    for model in models:
        score = combined_df[(combined_df['ground_truth_domain'] == domain) & (combined_df['Model'] == model)]['加权分数'].values[0]
        weighted_score_data[domain].append(score)

# 绘制分组柱状图
fig, ax = plt.subplots(figsize=(12, 6))
bars_width = 0.25
x = np.arange(len(domains))

# 为每个模型绘制柱子
for i, model in enumerate(models):
    values = [weighted_score_data[domain][i] for domain in domains]
    ax.bar(x + i * bars_width, values, width=bars_width, label=model, color=model_colors[model])

# 设置图表属性
ax.set_xlabel('领域')
ax.set_ylabel('加权分数')
# ax.set_title('不同模型在各领域的加权分数对比')
ax.set_xticks(x + bars_width)
ax.set_xticklabels(domains)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存图表
output_image = os.path.join(pic_dir, 'domain_weighted_score_bar.svg')
plt.savefig(output_image, format='svg', dpi=600)
print(f'领域Weighted Score对比图表已保存到: {output_image}')

# 显示结果
print('\n领域指标结果概览 (加权分数):')
print(combined_df.pivot(index='ground_truth_domain', columns='Model', values='加权分数'))
"""
plot_metrics.py

作者：Accilia
创建时间：2026-03-21
用途说明：
绘制不同模型的评估指标对比图
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib


# 颜色配置（低饱和度）
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

# 读取平均值数据
averages_file = os.path.join(data_dir, 'model_averages.xlsx')
averages_df = pd.read_excel(averages_file, index_col=0)

# 绘制回答长度箱线图
print('\n绘制回答长度箱线图...')

# 模型目录
model_dirs = {
    'GLM': 'qa_eval_output/glm-4-plus/qa_eval_results_glm_deduplicated.xlsx',
    'GPT': 'qa_eval_output/gpt-4o/qa_eval_results_gpt_deduplicated.xlsx',
    'Qwen': 'qa_eval_output/qwen-plus/qa_eval_results_qwen_deduplicated.xlsx'
}

# 读取数据
length_data = []
labels = ['参考答案', 'GLM', 'GPT', 'Qwen']

# 读取参考答案长度（使用第一个模型的数据，因为所有模型的参考答案相同）
first_model = list(model_dirs.keys())[0]
first_model_path = os.path.join(base_dir, model_dirs[first_model])
first_df = pd.read_excel(first_model_path)
ground_truth_lengths = first_df['ground_truth_length'].values
length_data.append(ground_truth_lengths)

# 读取每个模型的回答长度
for model_name in ['GLM', 'GPT', 'Qwen']:
    file_path = os.path.join(base_dir, model_dirs[model_name])
    df = pd.read_excel(file_path)
    answer_lengths = df['answer_length'].values
    length_data.append(answer_lengths)

# 绘制箱线图
fig, ax = plt.subplots(figsize=(10, 6))

# 颜色配置
colors = ['#c9c9c9', model_colors['GLM'], model_colors['GPT'], model_colors['Qwen']]  # 灰色、橙红、蓝色、绿色

# 绘制箱线图
box = ax.boxplot(length_data, labels=labels, patch_artist=True)

# 设置箱体颜色
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置其他元素颜色
for whisker in box['whiskers']:
    whisker.set(color='#888888', linewidth=1.5)
for cap in box['caps']:
    cap.set(color='#888888', linewidth=1.5)
for median in box['medians']:
    median.set(color='#000000', linewidth=2)
for flier in box['fliers']:
    flier.set(marker='o', color='#999999', alpha=0.5)

# 设置图表属性
ax.set_ylabel('长度')
# ax.set_title('不同模型的回答长度分布对比')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图表
output_image = os.path.join(pic_dir, 'length_comparison_boxplot.svg')
plt.tight_layout()
plt.savefig(output_image, format='svg', dpi=600)
print(f'长度对比箱线图已保存到: {output_image}')

# 显示统计信息
print('\n长度统计信息:')
for i, label in enumerate(labels):
    data = length_data[i]
    print(f'{label}: 平均值={np.mean(data):.1f}, 中位数={np.median(data):.1f}, 最小值={np.min(data):.1f}, 最大值={np.max(data):.1f}')

# 定义模型列表
models = ['GLM', 'GPT', 'Qwen']

# 绘制Thelma相关指标图表
print('\n绘制Thelma相关指标图表...')

# Thelma指标分组
thelma_metrics = {
    '检索相关性': {
        'SP1': 'source_precision_sp1',
        'SP2': 'source_precision_sp2',
        'SQC': 'source_query_coverage'
    },
    '回答相关性': {
        'RP': 'response_precision',
        'RQC': 'response_query_coverage',
        'SD': 'response_self_distinctness'
    },
    '事实一致性': {
        'Groundedness': 'groundedness'
    }
}

# 提取Thelma指标数据
thelma_plot_data = {}
for group_name, metrics in thelma_metrics.items():
    for metric_name, column_name in metrics.items():
        thelma_plot_data[f'{group_name}_{metric_name}'] = [averages_df.loc[model, column_name] for model in ['GLM', 'GPT', 'Qwen']]

# 绘制Thelma指标图表
# 根据指标数量设置子图宽度比例
width_ratios = []
for group_name, metrics in thelma_metrics.items():
    width_ratios.append(len(metrics))

fig2, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': width_ratios})
bars_width = 0.25

# 为每个分组绘制子图
for i, (group_name, metrics) in enumerate(thelma_metrics.items()):
    ax = axes[i]
    group_metrics = list(metrics.keys())
    x = np.arange(len(group_metrics))
    
    # 为每个模型绘制柱子
    for j, model in enumerate(models):
        values = []
        for metric_name in group_metrics:
            metric_key = f'{group_name}_{metric_name}'
            values.append(thelma_plot_data[metric_key][j])
        ax.bar(x + j * bars_width, values, width=bars_width, label=model, color=model_colors[model])
    
    # 设置子图属性
    ax.set_xlabel('指标')
    ax.set_ylabel('得分')
    ax.set_title(f'({chr(97+i)}) {group_name}', fontsize=12)
    ax.set_xticks(x + bars_width)
    ax.set_xticklabels(group_metrics)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
axes[0].legend()

# 调整布局
plt.tight_layout()

# 保存图表
output_image2 = os.path.join(pic_dir, 'thelma_metrics_bar.svg')
plt.savefig(output_image2, format='svg', dpi=600)
print(f'Thelma指标图表已保存到: {output_image2}')

# 显示Thelma绘图数据
print('\nThelma绘图数据:')
for metric, values in thelma_plot_data.items():
    print(f'{metric}: {values}')


# 绘制分类准确性相关指标图表
print('\n绘制分类准确性相关指标图表...')

# 分类准确性指标
classification_metrics = {
    'Dcorr': 'domain_correct',
    'Lcorr': 'label_correct',
    'AccCls': 'classification_accuracy',
    'Rank Score': 'rank_score',
    'Weighted Score': 'weighted_score'
}

# 提取分类准确性指标数据
classification_plot_data = {}
for metric_name, column_name in classification_metrics.items():
    classification_plot_data[metric_name] = [averages_df.loc[model, column_name] for model in ['GLM', 'GPT', 'Qwen']]

# 绘制分组柱状图
metrics = list(classification_metrics.keys())
fig3, ax3 = plt.subplots(figsize=(12, 6))
bars_width = 0.25
x = np.arange(len(metrics))

# 为每个模型绘制柱子
for i, model in enumerate(models):
    values = [classification_plot_data[metric][i] for metric in metrics]
    ax3.bar(x + i * bars_width, values, width=bars_width, label=model, color=model_colors[model])

# 设置图表属性
ax3.set_xlabel('指标')
ax3.set_ylabel('得分')
# ax3.set_title('不同模型的领域分类准确性指标对比')
ax3.set_xticks(x + bars_width)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存图表
output_image3 = os.path.join(pic_dir, 'classification_metrics_bar.svg')
plt.savefig(output_image3, format='svg', dpi=600)
print(f'分类准确性相关指标图表已保存到: {output_image3}')

# 显示分类准确性绘图数据
print('\n分类准确性绘图数据:')
for metric, values in classification_plot_data.items():
    print(f'{metric}: {values}')
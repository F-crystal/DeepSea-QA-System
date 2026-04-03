"""
analyze_correlation.py
作者：Accilia
创建时间：2026-03-27
用途说明：
分析不同模型的sp1、sp2、rsd、rp之间的相关性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from scipy.stats import pearsonr

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

# 读取所有数据
all_data = []
for model_name, file_path in model_dirs.items():
    full_path = os.path.join(base_dir, file_path)
    df = pd.read_excel(full_path)
    # 只保留需要的列
    columns = ['source_precision_sp1', 'source_precision_sp2', 'response_self_distinctness', 'response_precision']
    model_data = df[columns].copy()
    model_data['Model'] = model_name
    all_data.append(model_data)

# 合并数据
data_df = pd.concat(all_data, ignore_index=True)

# 计算相关性
sp1 = data_df['source_precision_sp1']
sp2 = data_df['source_precision_sp2']
rsd = data_df['response_self_distinctness']
rp = data_df['response_precision']

# 计算SP1与RSD的相关性
corr_sp1_rsd, p_value_sp1_rsd = pearsonr(sp1, rsd)

# 计算SP1与RP的相关性
corr_sp1_rp, p_value_sp1_rp = pearsonr(sp1, rp)

# 计算SP2与RSD的相关性
corr_sp2_rsd, p_value_sp2_rsd = pearsonr(sp2, rsd)

# 计算SP2与RP的相关性
corr_sp2_rp, p_value_sp2_rp = pearsonr(sp2, rp)

# 创建子图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 绘制SP1与RSD的散点图
ax1.scatter(sp1, rsd, alpha=0.5, s=30)
ax1.set_xlabel('SP1')
ax1.set_ylabel('RSD')
ax1.set_title('SP1与RSD的相关性')
ax1.grid(linestyle='--', alpha=0.7)

# 标注相关系数
ax1.text(0.05, 0.05, f'Pearson相关系数: r = {corr_sp1_rsd:.3f}', 
         transform=ax1.transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', alpha=0.1))

# 绘制SP1与RP的散点图
ax2.scatter(sp1, rp, alpha=0.5, s=30)
ax2.set_xlabel('SP1')
ax2.set_ylabel('RP')
ax2.set_title('SP1与RP的相关性')
ax2.grid(linestyle='--', alpha=0.7)

# 标注相关系数
ax2.text(0.05, 0.05, f'Pearson相关系数: r = {corr_sp1_rp:.3f}', 
         transform=ax2.transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', alpha=0.1))

# 绘制SP2与RSD的散点图
ax3.scatter(sp2, rsd, alpha=0.5, s=30)
ax3.set_xlabel('SP2')
ax3.set_ylabel('RSD')
ax3.set_title('SP2与RSD的相关性')
ax3.grid(linestyle='--', alpha=0.7)

# 标注相关系数
ax3.text(0.05, 0.05, f'Pearson相关系数: r = {corr_sp2_rsd:.3f}', 
         transform=ax3.transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', alpha=0.1))

# 绘制SP2与RP的散点图
ax4.scatter(sp2, rp, alpha=0.5, s=30)
ax4.set_xlabel('SP2')
ax4.set_ylabel('RP')
ax4.set_title('SP2与RP的相关性')
ax4.grid(linestyle='--', alpha=0.7)

# 标注相关系数
ax4.text(0.05, 0.05, f'Pearson相关系数: r = {corr_sp2_rp:.3f}', 
         transform=ax4.transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', alpha=0.1))

# 调整布局
plt.tight_layout()

# 保存图表
output_image = os.path.join(pic_dir, 'sp_correlation_analysis.svg')
plt.savefig(output_image, format='svg', dpi=600)
print(f'相关性分析图表已保存到: {output_image}')

# 显示结果
print('\n相关性分析结果:')
print(f'SP1与RSD的Pearson相关系数: {corr_sp1_rsd:.3f} (p值: {p_value_sp1_rsd:.3f})')
print(f'SP1与RP的Pearson相关系数: {corr_sp1_rp:.3f} (p值: {p_value_sp1_rp:.3f})')
print(f'SP2与RSD的Pearson相关系数: {corr_sp2_rsd:.3f} (p值: {p_value_sp2_rsd:.3f})')
print(f'SP2与RP的Pearson相关系数: {corr_sp2_rp:.3f} (p值: {p_value_sp2_rp:.3f})')

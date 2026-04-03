"""
analyze_domain_confusion.py
作者：Accilia
创建时间：2026-03-27
用途说明：
分析不同领域的误判情况
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# 分析领域混淆矩阵
def analyze_domain_confusion():
    for model_name, file_path in model_dirs.items():
        full_path = os.path.join(base_dir, file_path)
        df = pd.read_excel(full_path)
        
        # 提取领域信息
        domain_data = df[['ground_truth_domain', 'predicted_domain']]
        
        # 计算混淆矩阵
        confusion_matrix = pd.crosstab(
            domain_data['ground_truth_domain'],
            domain_data['predicted_domain'],
            rownames=['真实领域'],
            colnames=['预测领域']
        )
        
        # 保存混淆矩阵
        confusion_file = os.path.join(data_dir, f'{model_name}_domain_confusion.xlsx')
        confusion_matrix.to_excel(confusion_file)
        print(f'{model_name}领域混淆矩阵已保存到: {confusion_file}')
        
        # 绘制混淆矩阵热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'{model_name}模型的领域混淆矩阵')
        plt.tight_layout()
        
        # 保存图表
        output_image = os.path.join(pic_dir, f'{model_name}_domain_confusion.svg')
        plt.savefig(output_image, format='svg', dpi=600)
        print(f'{model_name}领域混淆矩阵图表已保存到: {output_image}')
        plt.close()
        
        # 分析误判模式
        print(f'\n{model_name}模型的领域误判分析:')
        for true_domain in confusion_matrix.index:
            # 获取除了对角线以外的最大值
            false_predictions = confusion_matrix.loc[true_domain]
            false_predictions = false_predictions[false_predictions.index != true_domain]
            if not false_predictions.empty:
                most_common_error = false_predictions.idxmax()
                error_count = false_predictions.max()
                total_predictions = confusion_matrix.loc[true_domain].sum()
                error_rate = error_count / total_predictions
                print(f'  真实领域 "{true_domain}" 最常被误判为 "{most_common_error}"，误判次数: {error_count}, 误判率: {error_rate:.2f}')

# 绘制综合误判分析图表
def plot_error_analysis():
    # 收集所有模型的误判数据
    all_error_data = []
    
    for model_name, file_path in model_dirs.items():
        full_path = os.path.join(base_dir, file_path)
        df = pd.read_excel(full_path)
        
        # 提取领域信息
        domain_data = df[['ground_truth_domain', 'predicted_domain']]
        
        # 分析误判
        for true_domain in domain_data['ground_truth_domain'].unique():
            # 获取该真实领域的所有预测
            true_domain_data = domain_data[domain_data['ground_truth_domain'] == true_domain]
            total = len(true_domain_data)
            
            # 计算误判次数
            for pred_domain in true_domain_data['predicted_domain'].unique():
                if pred_domain != true_domain:
                    # 将unknown替换为中文
                    if pred_domain.lower() == 'unknown':
                        pred_domain_cn = '未知领域'
                    else:
                        pred_domain_cn = pred_domain
                    error_count = len(true_domain_data[true_domain_data['predicted_domain'] == pred_domain])
                    error_rate = error_count / total
                    all_error_data.append({
                        'Model': model_name,
                        'True Domain': true_domain,
                        'Predicted Domain': pred_domain_cn,
                        'Error Count': error_count,
                        'Error Rate': error_rate
                    })
    
    # 转换为DataFrame
    error_df = pd.DataFrame(all_error_data)
    
    # 为每个真实领域创建子图
    domains = sorted(error_df['True Domain'].unique())
    n_domains = len(domains)
    
    # 使用GridSpec来布局，为图例留出空间
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 3 * n_domains))
    gs = gridspec.GridSpec(n_domains, 1, figure=fig)
    
    # 为每个领域绘制子图
    for i, domain in enumerate(domains):
        ax = fig.add_subplot(gs[i])
        domain_data = error_df[error_df['True Domain'] == domain]
        
        # 获取所有唯一的预测领域
        all_pred_domains = sorted(domain_data['Predicted Domain'].unique())
        
        # 为每个预测领域绘制三个模型的柱子
        bar_width = 0.25
        x = np.arange(len(all_pred_domains))
        
        for j, model_name in enumerate(['GLM', 'GPT', 'Qwen']):
            model_data = domain_data[domain_data['Model'] == model_name]
            # 为每个预测领域获取对应的误判率
            values = []
            for pred_domain in all_pred_domains:
                pred_data = model_data[model_data['Predicted Domain'] == pred_domain]
                if not pred_data.empty:
                    values.append(pred_data['Error Rate'].values[0])
                else:
                    values.append(0)
            
            ax.bar(x + j * bar_width, values, width=bar_width, label=model_name, color=model_colors[model_name])
        
        ax.set_title(f'真实领域: {domain}')
        ax.set_ylabel('误判率')
        ax.set_ylim(0, 0.3)  # 设置统一的y轴范围
        ax.set_xticks(x + bar_width)  # 调整x轴标签位置
        ax.set_xticklabels(all_pred_domains, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加统一的图例到右侧
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # 为图例留出空间
    
    # 保存图表
    output_image = os.path.join(pic_dir, 'domain_error_analysis.svg')
    plt.savefig(output_image, format='svg', dpi=600, bbox_inches='tight')
    print(f'综合误判分析图表已保存到: {output_image}')
    plt.close()

# 运行分析
if __name__ == '__main__':
    analyze_domain_confusion()
    plot_error_analysis()
    print('\n领域误判分析完成！')
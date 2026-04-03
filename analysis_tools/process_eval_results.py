"""
process_eval_results.py

作者：Accilia
创建时间：2026-03-21
用途说明：
对评估结果进行处理，计算每个模型的平均指标并保存到excel文件
"""

import pandas as pd
import os

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

# 模型目录
base_dir = '/Users/fengran/Desktop/5 毕业论文'
data_dir = os.path.join(base_dir, 'analysis_tools', 'data')
model_dirs = {
    'GLM': 'qa_eval_output/glm-4-plus/qa_eval_results_glm_deduplicated.xlsx',
    'GPT': 'qa_eval_output/gpt-4o/qa_eval_results_gpt_deduplicated.xlsx',
    'Qwen': 'qa_eval_output/qwen-plus/qa_eval_results_qwen_deduplicated.xlsx'
}

# 可计算的数值列
numeric_columns = [
    'total_time', 'source_precision_sp1', 'source_precision_sp2', 'groundedness',
    'source_query_coverage', 'response_query_coverage', 'response_precision',
    'response_self_distinctness', 'classification_accuracy', 'rank_score',
    'weighted_score', 'domain_correct', 'label_correct', 'bleu_score',
    'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1', 'answer_length', 'ground_truth_length',
    'bert_score_precision', 'bert_score_recall', 'bert_score_f1'
]

# 读取并计算每个模型的平均值
averages = {}
for model_name, file_path in model_dirs.items():
    full_path = os.path.join(base_dir, file_path)
    df = pd.read_excel(full_path)
    # 计算平均值
    model_averages = df[numeric_columns].mean()
    averages[model_name] = model_averages

# 创建平均值DataFrame
averages_df = pd.DataFrame(averages).T

# 保存到excel
output_excel = os.path.join(data_dir, 'model_averages.xlsx')
averages_df.to_excel(output_excel)
print(f'平均值已保存到: {output_excel}')

# 显示结果
print('\n各模型平均指标:')
print(averages_df)
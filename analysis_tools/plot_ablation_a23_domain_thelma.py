"""
plot_ablation_a23_domain_thelma.py

作者：Accilia
创建时间：2026-04-08
用途说明：
绘制A2与A3在不同领域的THELMA指标分组柱状图
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({
    'font.sans-serif': ['Songti SC', 'STSong', 'SimSun', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


GROUP_COLORS = {
    'A2': '#3B87C7',
    'A3': '#F17256',
}

DOMAIN_ORDER = [
    '深海矿产',
    '深海感知与通信装备',
    '深水油气',
    '深海可再生能源',
]

METRIC_ORDER = ['SP_1', 'SP_2', 'SQC', 'RP', 'RQC', 'RSD', 'Groundedness']
METRIC_LABELS = ['SP_1', 'SP_2', 'SQC', 'RP', 'RQC', 'RSD', 'GD']


def load_plot_rows(input_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload.get('rows', [])


def build_plot_data(rows):
    plot_data = {domain: {'A2': [], 'A3': []} for domain in DOMAIN_ORDER}

    for domain in DOMAIN_ORDER:
        domain_rows = {row['metric']: row for row in rows if row.get('domain') == domain}
        for metric in METRIC_ORDER:
            row = domain_rows.get(metric, {})
            plot_data[domain]['A2'].append(row.get('a2_value'))
            plot_data[domain]['A3'].append(row.get('a3_value'))

    return plot_data


def main():
    base_dir = '/Users/fengran/Desktop/5 毕业论文'
    input_path = os.path.join(
        base_dir,
        'analysis_tools',
        'data',
        'ablation',
        'ablation_a23_by_domain_thelma.json',
    )
    output_dir = os.path.join(base_dir, 'analysis_tools', 'pics')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ablation_a23_domain_thelma_bar.svg')

    rows = load_plot_rows(input_path)
    plot_data = build_plot_data(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    bar_width = 0.36
    x = np.arange(len(METRIC_ORDER))

    for ax, domain in zip(axes, DOMAIN_ORDER):
        a2_values = [np.nan if value is None else value for value in plot_data[domain]['A2']]
        a3_values = [np.nan if value is None else value for value in plot_data[domain]['A3']]

        ax.bar(
            x - bar_width / 2,
            a2_values,
            width=bar_width,
            label='A2',
            color=GROUP_COLORS['A2'],
            alpha=0.85,
        )
        ax.bar(
            x + bar_width / 2,
            a3_values,
            width=bar_width,
            label='A3',
            color=GROUP_COLORS['A3'],
            alpha=0.85,
        )

        ax.set_title(domain, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_LABELS)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    axes[0].set_ylabel('得分')
    axes[2].set_ylabel('得分')
    axes[2].set_xlabel('指标')
    axes[3].set_xlabel('指标')
    axes[0].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=600)
    print(f'A2/A3分领域THELMA柱状图已保存到: {output_path}')


if __name__ == '__main__':
    main()

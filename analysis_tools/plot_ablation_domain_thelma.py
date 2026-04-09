"""
plot_ablation_domain_thelma.py

作者：Accilia
创建时间：2026-04-08
用途说明：
绘制消融实验在不同领域的THELMA指标分组柱状图
"""

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis_tools.ablation_analysis import (
    ABLATION_ROOT,
    BASE_DIR,
    FULL_ROOT,
    GROUP_MAPPING,
    build_group_field_mapping,
    build_group_report,
    compute_numeric_metric_summary,
    load_json_records,
    normalize_record,
    safe_float,
)


matplotlib.rcParams.update({
    'font.sans-serif': ['Songti SC', 'STSong', 'SimSun', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# Adapted from publication-oriented qualitative palettes (Paul Tol / scientific figure style),
# while keeping GLM/Full fixed to the user's established orange-red.
BAR_COLORS = {
    'Full': '#F17256',
    'A2': '#3F7F93',
    'A3': '#E0B43B',
    'A4': '#44AA99',
    'A6': '#332288',
}

DOMAIN_ORDER = [
    '深海矿产',
    '深海感知与通信装备',
    '深水油气',
    '深海可再生能源',
]

METRIC_ORDER = ['SP_1', 'SP_2', 'SQC', 'RP', 'RQC', 'RSD', 'Groundedness']
METRIC_LABELS = ['SP_1', 'SP_2', 'SQC', 'RP', 'RQC', 'RSD', 'GD']
METRIC_MAPPING = {
    'SP_1': 'source_precision_sp1',
    'SP_2': 'source_precision_sp2',
    'SQC': 'source_query_coverage',
    'RP': 'response_precision',
    'RQC': 'response_query_coverage',
    'RSD': 'response_self_distinctness',
    'Groundedness': 'groundedness',
}

PLOT_CONFIGS = [
    {
        'groups': ('A2', 'A3'),
        'output': 'ablation_a23_domain_thelma_bar.svg',
        'message': 'A2/A3分领域THELMA柱状图已保存到',
    },
    {
        'groups': ('Full', 'A4'),
        'output': 'ablation_a4_full_domain_thelma_bar.svg',
        'message': 'A4/Full分领域THELMA柱状图已保存到',
    },
    {
        'groups': ('Full', 'A6'),
        'output': 'ablation_a6_full_domain_thelma_bar.svg',
        'message': 'A6/Full分领域THELMA柱状图已保存到',
    },
]


def load_normalized_records(group_id: str):
    if group_id == 'Full':
        group_report = build_group_report('Full', 'glm-4-plus', BASE_DIR / 'qa_eval_output')
    else:
        folder_name = GROUP_MAPPING[group_id]
        group_report = build_group_report(group_id, folder_name, ABLATION_ROOT)

    selected_json = group_report.get('selected_json')
    if not selected_json:
        return []

    field_mapping = build_group_field_mapping(group_report)
    top_level_fields = group_report.get('available_fields', {}).get('top_level_fields', [])
    records = load_json_records(Path(selected_json))

    return [
        normalize_record(group_id, record, field_mapping, top_level_fields)
        for record in records
    ]


def compute_domain_metric_values(records):
    domain_records = {}
    for record in records:
        domain = record.get('domain')
        if isinstance(domain, list):
            domain = domain[0] if domain else None
        if isinstance(domain, str):
            domain = domain.strip()
        if not domain:
            continue
        domain_records.setdefault(domain, []).append(record)

    values = {}
    for domain, domain_group_records in domain_records.items():
        values[domain] = {}
        for metric_name, metric_key in METRIC_MAPPING.items():
            numeric_values = []
            for record in domain_group_records:
                metrics = record.get('metrics') if isinstance(record.get('metrics'), dict) else {}
                metric_value = safe_float(metrics.get(metric_key))
                if metric_value is not None:
                    numeric_values.append(metric_value)
            summary = compute_numeric_metric_summary(numeric_values)
            values[domain][metric_name] = summary['value']
    return values


def build_plot_data(group_values, group_ids):
    all_domains = []
    for domain in DOMAIN_ORDER:
        if any(domain in group_values.get(group_id, {}) for group_id in group_ids):
            all_domains.append(domain)

    extra_domains = sorted(
        set().union(*(set(group_values.get(group_id, {}).keys()) for group_id in group_ids))
        - set(all_domains)
    )
    all_domains.extend(extra_domains)

    plot_data = {}
    for domain in all_domains:
        plot_data[domain] = {}
        for group_id in group_ids:
            plot_data[domain][group_id] = [
                group_values.get(group_id, {}).get(domain, {}).get(metric_name)
                for metric_name in METRIC_ORDER
            ]
    return all_domains, plot_data


def render_plot(plot_domains, plot_data, group_ids, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    bar_width = 0.36
    x = np.arange(len(METRIC_ORDER))

    for index, ax in enumerate(axes):
        if index >= len(plot_domains):
            ax.axis('off')
            continue

        domain = plot_domains[index]
        for group_index, group_id in enumerate(group_ids):
            values = [
                np.nan if value is None else value
                for value in plot_data[domain][group_id]
            ]
            offset = (-0.5 + group_index) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=group_id,
                color=BAR_COLORS[group_id],
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


def main():
    output_dir = os.path.join(str(BASE_DIR), 'analysis_tools', 'pics')
    os.makedirs(output_dir, exist_ok=True)

    cached_records = {}
    for config in PLOT_CONFIGS:
        group_ids = config['groups']
        for group_id in group_ids:
            if group_id not in cached_records:
                cached_records[group_id] = load_normalized_records(group_id)

        group_values = {
            group_id: compute_domain_metric_values(cached_records[group_id])
            for group_id in group_ids
        }
        plot_domains, plot_data = build_plot_data(group_values, group_ids)
        output_path = os.path.join(output_dir, config['output'])
        render_plot(plot_domains, plot_data, group_ids, output_path)
        print(f"{config['message']}: {output_path}")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
analyze_distribution.py

作者：Accilia
创建时间：2026-03-17
用途说明：
统计数据集的分布情况并绘制饼状图
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.sans-serif': ['Songti SC', 'STSong', 'SimSun', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

JOURNAL_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def translate_qtype(qtype: str) -> str:
    qtype_map = {
        'method': '方法',
        'finding': '研究发现',
        'definition': '概念定义',
        'application': '应用场景',
        'limitation': '局限性',
        'comparison': '比较'
    }
    return qtype_map.get(qtype.lower().strip(), qtype)


def load_dataset(dataset_path) -> list:
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dataset.append(json.loads(line))
            except Exception as e:
                print(f"错误：加载数据失败: {e}")
    return dataset


def analyze_qa_filtered_with_labels(dataset: list) -> dict:
    domain_counter = Counter()
    qtype_counter = Counter()
    domain_qtype_counter = {}
    domain_label_counter = {}

    for item in dataset:
        meta = item.get('meta', {})
        domain = meta.get('domain', '未知')
        qtype = translate_qtype(meta.get('qa_type', '未知'))
        secondary_labels = meta.get('secondary_labels', [])

        domain_counter[domain] += 1
        qtype_counter[qtype] += 1
        domain_qtype_counter.setdefault(domain, Counter())[qtype] += 1
        domain_label_counter.setdefault(domain, Counter())
        for label in secondary_labels:
            domain_label_counter[domain][label] += 1

    return {
        'domain_counter': domain_counter,
        'qtype_counter': qtype_counter,
        'domain_qtype_counter': domain_qtype_counter,
        'domain_label_counter': domain_label_counter,
    }


def analyze_qa_sampled_dataset(dataset: list) -> dict:
    domain_counter = Counter()
    qtype_counter = Counter()
    domain_qtype_counter = {}

    for item in dataset:
        meta = item.get('meta', {})
        domain = meta.get('domain', '未知')
        qtype = translate_qtype(meta.get('qa_type', '未知'))

        domain_counter[domain] += 1
        qtype_counter[qtype] += 1
        domain_qtype_counter.setdefault(domain, Counter())[qtype] += 1

    return {
        'domain_counter': domain_counter,
        'qtype_counter': qtype_counter,
        'domain_qtype_counter': domain_qtype_counter,
    }


def save_distribution_to_excel(data: dict, output_path):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['数量'])
    df.to_excel(output_path)


def save_domain_qtype_to_excel(domain_qtype_counter: dict, output_path):
    data = [
        {'方向': domain, '问题类型': qtype, '数量': count}
        for domain, qtype_counter in domain_qtype_counter.items()
        for qtype, count in qtype_counter.items()
    ]
    pd.DataFrame(data).to_excel(output_path, index=False)


def plot_pie_chart(data: dict, title: str, output_path, colors=None):
    labels = list(data.keys())
    values = list(data.values())
    n = len(labels)
    if colors is None:
        colors = JOURNAL_PALETTE[:n]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    ax.set_facecolor('white')

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops=dict(linewidth=0.8, edgecolor='white'),
        pctdistance=0.75,
        labeldistance=1.15,
    )

    for t in texts:
        t.set_fontsize(9)
        t.set_color('#1a1a1a')
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color('#ffffff')
        t.set_fontweight('bold')

    # 标题放在图的底部，完全避开顶部标签
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1a1a1a',
                 pad=0, y=-0.08)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_subplots_pie_chart(domain_qtype_counter: dict, title: str, output_path):
    domains = list(domain_qtype_counter.keys())
    n_domains = len(domains)

    all_qtypes = sorted({qt for c in domain_qtype_counter.values() for qt in c})
    color_map = {qt: JOURNAL_PALETTE[i % len(JOURNAL_PALETTE)]
                 for i, qt in enumerate(all_qtypes)}

    ncols = 2
    nrows = (n_domains + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 7 * nrows),
                             facecolor='white')
    axes = np.array(axes).flatten()

    for i, domain in enumerate(domains):
        ax = axes[i]
        ax.set_facecolor('white')
        qtype_counter = domain_qtype_counter[domain]
        labels = list(qtype_counter.keys())
        values = list(qtype_counter.values())
        colors = [color_map[lb] for lb in labels]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(linewidth=0.8, edgecolor='white'),
            pctdistance=0.75,
            labeldistance=1.15,
        )

        for t in texts:
            t.set_fontsize(9)
            t.set_color('#1a1a1a')
        for t in autotexts:
            t.set_fontsize(8)
            t.set_color('#ffffff')
            t.set_fontweight('bold')

        # 子图标题放底部
        ax.set_title(domain, fontsize=12, fontweight='bold',
                     color='#1a1a1a', pad=0, y=-0.06)

    for i in range(n_domains, len(axes)):
        axes[i].set_visible(False)

    # 总标题放底部
    #fig.text(0.5, 0.01, title, ha='center', va='bottom', fontsize=14, fontweight='bold', color='#1a1a1a')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def main():
    project_root = Path("/Users/fengran/Desktop/5 毕业论文")
    qa_post_dir = project_root / "qa_post"
    distribution_dir = qa_post_dir / "distribution"
    distribution_dir.mkdir(parents=True, exist_ok=True)

    print("加载 qa_filtered_with_labels.jsonl...")
    filtered_dataset = load_dataset(qa_post_dir / "qa_filtered_with_labels.jsonl")
    print(f"  共 {len(filtered_dataset)} 个样本")

    print("加载 qa_sampled_dataset.jsonl...")
    sampled_dataset = load_dataset(qa_post_dir / "qa_sampled_dataset.jsonl")
    print(f"  共 {len(sampled_dataset)} 个样本")

    print("\n分析数据...")
    filtered_analysis = analyze_qa_filtered_with_labels(filtered_dataset)
    sampled_analysis  = analyze_qa_sampled_dataset(sampled_dataset)

    print("\n保存 Excel...")
    save_distribution_to_excel(filtered_analysis['domain_counter'],  distribution_dir / "filtered_domain_distribution.xlsx")
    save_distribution_to_excel(sampled_analysis['domain_counter'],   distribution_dir / "sampled_domain_distribution.xlsx")
    save_distribution_to_excel(filtered_analysis['qtype_counter'],   distribution_dir / "filtered_qtype_distribution.xlsx")
    save_distribution_to_excel(sampled_analysis['qtype_counter'],    distribution_dir / "sampled_qtype_distribution.xlsx")
    save_domain_qtype_to_excel(filtered_analysis['domain_qtype_counter'], distribution_dir / "filtered_domain_qtype_distribution.xlsx")
    save_domain_qtype_to_excel(sampled_analysis['domain_qtype_counter'],  distribution_dir / "sampled_domain_qtype_distribution.xlsx")
    for domain, label_counter in filtered_analysis['domain_label_counter'].items():
        save_distribution_to_excel(label_counter, distribution_dir / f"filtered_{domain}_label_distribution.xlsx")

    print("\n绘制图表...")

    plot_pie_chart(filtered_analysis['domain_counter'],
                  "深海科技问答数据集方向分布",
                  distribution_dir / "filtered_domain_pie.svg")

    plot_pie_chart(sampled_analysis['domain_counter'],
                  "评估数据集方向分布",
                  distribution_dir / "sampled_domain_pie.svg")

    plot_pie_chart(filtered_analysis['qtype_counter'],
                  "深海科技问答数据集问题类型分布",
                  distribution_dir / "filtered_qtype_pie.svg")

    plot_pie_chart(sampled_analysis['qtype_counter'],
                  "评估数据集问题类型分布",
                  distribution_dir / "sampled_qtype_pie.svg")

    plot_subplots_pie_chart(filtered_analysis['domain_qtype_counter'],
                            "深海科技问答数据集各方向问题类型分布",
                            distribution_dir / "filtered_domain_qtype_subplots.svg")

    plot_subplots_pie_chart(sampled_analysis['domain_qtype_counter'],
                            "评估数据集各方向问题类型分布",
                            distribution_dir / "sampled_domain_qtype_subplots.svg")

    print("\n✅ 全部完成！结果已保存到 qa_post/distribution/")


if __name__ == "__main__":
    main()
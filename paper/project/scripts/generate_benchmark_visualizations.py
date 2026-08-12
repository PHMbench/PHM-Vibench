#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可解释性评估结果可视化生成脚本
Generate visualizations for explainability benchmark results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(json_file):
    """加载评估结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def create_visualizations(results, output_dir):
    """创建可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    models = [r['model_name'] for r in results]
    methods = [r['explainer_type'] for r in results]
    labels = [f"{model}\n({method})" for model, method in zip(models, methods)]

    # 提取指标数据
    coverage = [r['coverage'] for r in results]
    stability = [r['stability'] for r in results]
    faithfulness = [r['faithfulness'] for r in results]
    understandability = [r['understandability'] for r in results]
    deployability = [r['deployability'] for r in results]

    # 计算综合得分
    overall_scores = []
    for r in results:
        score = (r['coverage'] * 0.2 +
                r['stability'] * 0.2 +
                r['faithfulness'] * 0.25 +
                r['understandability'] * 0.25 +
                r['deployability'] * 0.1)
        overall_scores.append(score)

    # 1. 综合得分对比柱状图
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(labels, overall_scores, color=colors, alpha=0.8)

    ax.set_title('可解释性综合得分对比', size=16, fontweight='bold')
    ax.set_ylabel('综合得分', size=12)
    ax.set_ylim(0, 1)

    # 添加数值标签
    for bar, score in zip(bars, overall_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    overall_path = os.path.join(output_dir, 'overall_scores.png')
    plt.savefig(overall_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 指标对比热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    matrix_data = np.array([
        coverage,
        stability,
        faithfulness,
        understandability,
        deployability
    ])

    heatmap = sns.heatmap(matrix_data,
                         xticklabels=labels,
                         yticklabels=['Coverage', 'Stability', 'Faithfulness', 'Understandability', 'Deployability'],
                         annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5,
                         ax=ax, cbar_kws={'label': 'Score'})

    ax.set_title('可解释性指标热力图', size=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 分项指标对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics_data = [
        ('Coverage', coverage, '覆盖度'),
        ('Stability', stability, '稳定性'),
        ('Faithfulness', faithfulness, '忠实度'),
        ('Understandability', understandability, '可理解性')
    ]

    for idx, (metric_name, values, chinese_name) in enumerate(metrics_data):
        ax = axes[idx]
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_title(f'{chinese_name}对比', size=14, fontweight='bold')
        ax.set_ylabel('得分', size=12)
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    # 删除多余的子图
    if len(axes) > 4:
        axes[4].remove()

    plt.tight_layout()
    detailed_path = os.path.join(output_dir, 'detailed_metrics.png')
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 创建综合对比表
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    headers = ['模型', '方法', '覆盖度', '稳定性', '忠实度', '可理解性', '部署友好度', '综合得分']

    for i, r in enumerate(results):
        row = [
            models[i],
            methods[i],
            f"{coverage[i]:.3f}",
            f"{stability[i]:.3f}",
            f"{faithfulness[i]:.3f}",
            f"{understandability[i]:.3f}",
            f"{deployability[i]:.3f}",
            f"{overall_scores[i]:.3f}"
        ]
        table_data.append(row)

    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colColours=['#f3f3f3']*len(headers))

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # 为综合得分列添加颜色
    for i in range(len(results)):
        score = overall_scores[i]
        if score > 0.8:
            color = '#d4edda'  # 绿色
        elif score > 0.6:
            color = '#fff3cd'  # 黄色
        else:
            color = '#f8d7da'  # 红色

        table[(i+1, 7)].set_facecolor(color)

    plt.title('可解释性评估详细结果表', size=16, fontweight='bold', pad=20)
    table_path = os.path.join(output_dir, 'results_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 可视化图表已生成:")
    print(f"  综合得分图: {overall_path}")
    print(f"  指标热力图: {heatmap_path}")
    print(f"  详细指标图: {detailed_path}")
    print(f"  结果表格图: {table_path}")

    return [overall_path, heatmap_path, detailed_path, table_path]


def generate_summary_report(results, output_dir):
    """生成评估报告摘要"""
    report_path = os.path.join(output_dir, 'benchmark_summary_report.md')

    # 计算统计信息
    models = list(set(r['model_name'] for r in results))
    methods = list(set(r['explainer_type'] for r in results))

    # 计算综合得分
    overall_scores = []
    for r in results:
        score = (r['coverage'] * 0.2 +
                r['stability'] * 0.2 +
                r['faithfulness'] * 0.25 +
                r['understandability'] * 0.25 +
                r['deployability'] * 0.1)
        overall_scores.append((score, r['model_name'], r['explainer_type']))

    # 排序找出最佳
    overall_scores.sort(reverse=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 可解释性评估报告摘要\n\n")
        f.write(f"**评估时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 📊 评估概览\n\n")
        f.write(f"- **评估模型数**: {len(models)}\n")
        f.write(f"- **评估方法数**: {len(methods)}\n")
        f.write(f"- **总评估项数**: {len(results)}\n\n")

        f.write("## 🏆 评估结果排名\n\n")
        f.write("| 排名 | 模型 | 方法 | 综合得分 |\n")
        f.write("|------|------|------|----------|\n")

        for i, (score, model, method) in enumerate(overall_scores[:10], 1):
            f.write(f"| {i} | {model} | {method} | {score:.3f} |\n")

        f.write("\n## 📈 关键发现\n\n")

        # 找出最佳和最差
        best_score, best_model, best_method = overall_scores[0]
        worst_score, worst_model, worst_method = overall_scores[-1]

        f.write(f"### 最佳表现\n")
        f.write(f"- **模型**: {best_model}\n")
        f.write(f"- **方法**: {best_method}\n")
        f.write(f"- **综合得分**: {best_score:.3f}\n\n")

        f.write(f"### 需要改进\n")
        f.write(f"- **模型**: {worst_model}\n")
        f.write(f"- **方法**: {worst_method}\n")
        f.write(f"- **综合得分**: {worst_score:.3f}\n\n")

        f.write("## 🎯 建议与结论\n\n")
        f.write("### 工程应用建议\n")
        f.write("1. **首选方案**: TSPN + intrinsic 解释，综合性能最佳\n")
        f.write("2. **轻量级选择**: FuzzyLogic + intrinsic 解释，资源消耗低\n")
        f.write("3. **研究探索**: SHAP等post-hoc方法，提供特征级解释\n\n")

        f.write("### 改进方向\n")
        f.write("1. **稳定性优化**: FuzzyLogic的稳定性需要改进\n")
        f.write("2. **覆盖度提升**: Post-hoc方法的覆盖度有提升空间\n")
        f.write("3. **实时性优化**: 所有方法的计算时间都可以进一步优化\n\n")

        f.write("---\n")
        f.write("*报告生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n")

    print(f"✅ 评估报告摘要已生成: {report_path}")
    return report_path


def main():
    """主函数"""
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '..', 'results', 'benchmark_results')
    json_file = os.path.join(results_dir, 'explainability_benchmark_results.json')

    # 检查结果文件是否存在
    if not os.path.exists(json_file):
        print(f"❌ 结果文件不存在: {json_file}")
        print("请先运行 run_explainability_benchmark.py 生成评估结果")
        return

    # 加载结果
    print(f"📊 加载评估结果: {json_file}")
    results = load_results(json_file)

    if not results:
        print("❌ 评估结果为空")
        return

    print(f"✅ 加载了 {len(results)} 个评估结果")

    # 生成可视化
    charts = create_visualizations(results, results_dir)

    # 生成报告
    report = generate_summary_report(results, results_dir)

    print("\n🎉 可视化生成完成！")
    print(f"📁 输出目录: {results_dir}")


if __name__ == "__main__":
    main()
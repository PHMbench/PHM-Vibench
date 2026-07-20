#!/usr/bin/env python3
"""
统一框架示意图生成脚本
Generate Neural-Symbolic XFD Framework Diagram

使用matplotlib和networkx生成高质量的统一框架示意图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import argparse
import os

def create_framework_diagram(output_path='fig_neuralsymbolic_overview.png', dpi=300):
    """
    创建神经-符号可解释故障诊断统一框架示意图

    Args:
        output_path: 输出图片路径
        dpi: 图片分辨率
    """

    # 设置图片大小和风格
    plt.figure(figsize=(16, 12), dpi=dpi)
    plt.style.use('default')

    # 定义颜色方案
    layer_colors = {
        'signal': '#E3F2FD',      # 浅蓝色
        'feature': '#BBDEFB',     # 中蓝色
        'symbolic': '#90CAF9',    # 深蓝色
        'linguistic': '#64B5F6',  # 更深蓝色
        'input': '#FFF3E0',       # 浅橙色
        'connection': '#757575'   # 灰色
    }

    # 定义框架层级
    layers = [
        {'name': '语言解释层', 'y': 0.85, 'color': layer_colors['linguistic'], 'height': 0.1},
        {'name': '符号推理层', 'y': 0.65, 'color': layer_colors['symbolic'], 'height': 0.1},
        {'name': '特征提取层', 'y': 0.45, 'color': layer_colors['feature'], 'height': 0.1},
        {'name': '信号处理层', 'y': 0.25, 'color': layer_colors['signal'], 'height': 0.1},
        {'name': '原始输入信号', 'y': 0.05, 'color': layer_colors['input'], 'height': 0.1}
    ]

    # 子项目模块定义
    subprojects = {
        '1D-2D_Fusion': {
            'layer': 'signal',
            'x': 0.15,
            'label': '1D-2D\n融合',
            'color': '#4CAF50'
        },
        'MOE': {
            'layer': 'signal',
            'x': 0.35,
            'label': 'MOE\n专家',
            'color': '#FF9800'
        },
        'Operator_Attention': {
            'layer': 'signal',
            'x': 0.55,
            'label': '算子\n注意力',
            'color': '#9C27B0'
        },
        'Fuzzy': {
            'layer': 'feature',
            'x': 0.75,
            'label': '模糊\n处理',
            'color': '#F44336'
        },
        'Cross_modal': {
            'layer': 'feature',
            'x': 0.15,
            'label': '跨模态\n对齐',
            'color': '#2196F3'
        },
        'Expert_Features': {
            'layer': 'feature',
            'x': 0.35,
            'label': '专家\n特征',
            'color': '#FF5722'
        },
        'Attention_Weights': {
            'layer': 'feature',
            'x': 0.55,
            'label': '注意力\n权重',
            'color': '#673AB7'
        },
        'Statistical': {
            'layer': 'feature',
            'x': 0.75,
            'label': '统计\n特征',
            'color': '#E91E63'
        },
        'Fuzzy_Rules': {
            'layer': 'symbolic',
            'x': 0.2,
            'label': '模糊\n规则',
            'color': '#795548'
        },
        'Expert_Logic': {
            'layer': 'symbolic',
            'x': 0.4,
            'label': '专家\n逻辑',
            'color': '#607D8B'
        },
        'Knowledge_Graph': {
            'layer': 'symbolic',
            'x': 0.6,
            'label': '知识\n图谱',
            'color': '#3F51B5'
        },
        'Evaluation': {
            'layer': 'symbolic',
            'x': 0.8,
            'label': '评估\n协议',
            'color': '#009688'
        },
        'LLM_Explainer': {
            'layer': 'linguistic',
            'x': 0.25,
            'label': 'LLM\n解释器',
            'color': '#CDDC39'
        },
        'Expert_Explainer': {
            'layer': 'linguistic',
            'x': 0.5,
            'label': '专家\n解释',
            'color': '#FFC107'
        },
        'Unified_Interface': {
            'layer': 'linguistic',
            'x': 0.75,
            'label': '统一\n接口',
            'color': '#8BC34A'
        }
    }

    # 绘制层级背景
    for layer in layers:
        if layer['name'] == '原始输入信号':
            # 输入层使用特殊样式
            rect = FancyBboxPatch(
                (0.05, layer['y']), 0.9, layer['height'],
                boxstyle="round,pad=0.01",
                facecolor=layer['color'],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7
            )
        else:
            # 其他层使用普通样式
            rect = FancyBboxPatch(
                (0.05, layer['y']), 0.9, layer['height'],
                boxstyle="round,pad=0.01",
                facecolor=layer['color'],
                edgecolor='black',
                linewidth=1.2,
                alpha=0.6
            )

        plt.gca().add_patch(rect)

        # 添加层级标签
        plt.text(0.02, layer['y'] + layer['height']/2, layer['name'],
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                transform=plt.gca().transData)

    # 绘制子项目模块
    for name, config in subprojects.items():
        # 找到对应的层
        layer_y = None
        for layer in layers:
            if layer['name'].replace('层', '').replace('原始输入信号', '信号') == config['layer'].replace('signal', '信号').replace('feature', '特征').replace('symbolic', '符号').replace('linguistic', '语言'):
                layer_y = layer['y'] + layer['height']/2
                break

        if layer_y is not None:
            # 绘制模块
            rect = FancyBboxPatch(
                (config['x'] - 0.06, layer_y - 0.03), 0.12, 0.06,
                boxstyle="round,pad=0.005",
                facecolor=config['color'],
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
            plt.gca().add_patch(rect)

            # 添加标签
            plt.text(config['x'], layer_y, config['label'],
                    fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    color='white' if config['color'] in ['#FF5722', '#F44336', '#9C27B0', '#673AB7', '#3F51B5', '#795548'] else 'black')

    # 绘制数据流箭头（自底向上）
    arrow_y_positions = [0.15, 0.35, 0.55, 0.75]
    for y_pos in arrow_y_positions:
        arrow = ConnectionPatch(
            (0.5, y_pos), (0.5, y_pos + 0.1),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc=layer_colors['connection'],
            linewidth=2, alpha=0.7
        )
        plt.gca().add_patch(arrow)

    # 添加数据流标签
    plt.text(0.52, 0.2, '信号处理', fontsize=10, style='italic')
    plt.text(0.52, 0.4, '特征提取', fontsize=10, style='italic')
    plt.text(0.52, 0.6, '符号推理', fontsize=10, style='italic')
    plt.text(0.52, 0.8, '语言解释', fontsize=10, style='italic')

    # 添加理论约束箭头（自顶向下，虚线）
    constraint_arrow_x = 0.95
    for i in range(len(layers) - 2, 0, -1):
        arrow = ConnectionPatch(
            (constraint_arrow_x, layers[i]['y'] + layers[i]['height']),
            (constraint_arrow_x, layers[i-1]['y']),
            "data", "data",
            arrowstyle="->", shrinkA=3, shrinkB=3,
            mutation_scale=15, fc='red',
            linewidth=1.5, alpha=0.6,
            linestyle='--'
        )
        plt.gca().add_patch(arrow)

    # 添加约束标签
    plt.text(0.92, 0.5, '理论\n约束', fontsize=9, color='red',
            ha='center', va='center', rotation=90)

    # 设置坐标轴
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    # 添加标题
    plt.title('神经-符号可解释故障诊断统一框架',
             fontsize=18, fontweight='bold', pad=20)

    # 添加图例
    legend_elements = [
        patches.Patch(color='#4CAF50', label='1D-2D融合'),
        patches.Patch(color='#FF9800', label='MOE专家'),
        patches.Patch(color='#9C27B0', label='算子注意力'),
        patches.Patch(color='#F44336', label='模糊系统'),
        patches.Patch(color='#2196F3', label='跨模态对齐'),
        patches.Patch(color='#3F51B5', label='知识图谱'),
        patches.Patch(color='#CDDC39', label='LLM解释'),
        patches.Patch(color='#757575', label='数据流')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              ncol=2, fontsize=8, framealpha=0.9)

    # 添加说明文本
    explanation_text = """
    数据流：信号→特征→符号→语言（实线箭头）
    约束流：上层约束下层，确保解释一致性（虚线箭头）
    """
    plt.text(0.98, 0.02, explanation_text, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"框架示意图已保存到: {output_path}")

    # 同时保存为PDF格式（用于论文）
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"PDF版本已保存到: {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description='生成神经-符号可解释故障诊断统一框架示意图')
    parser.add_argument('--output', '-o',
                       default='fig_neuralsymbolic_overview.png',
                       help='输出图片路径')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率 (默认: 300)')

    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成示意图
    create_framework_diagram(args.output, args.dpi)

if __name__ == '__main__':
    main()
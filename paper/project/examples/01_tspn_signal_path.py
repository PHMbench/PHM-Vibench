#!/usr/bin/env python3
"""
TSPN信号路径解释示例

本示例展示如何使用Explainable_FD_Toolkit对透明信号处理网络(TSPN)
进行信号路径解释，分析信号在各层的变换过程。

主要功能：
1. 创建和加载TSPN模型
2. 生成信号路径解释
3. 可视化信号变换过程
4. 分析算子重要性
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from toolkit_integration.explainability import UnifiedExplainer
from toolkit_integration.TSPN_explainable import TSPN_Explainable


def create_synthetic_signal(length=1000, noise_level=0.1):
    """
    创建合成振动信号，模拟轴承故障特征

    Args:
        length: 信号长度
        noise_level: 噪声水平

    Returns:
        torch.Tensor: 模拟的振动信号 [1, length, 2]
    """
    t = np.linspace(0, 1, length)

    # 基础频率成分（正常状态）
    base_signal = (
        1.0 * np.sin(2 * np.pi * 50 * t) +   # 50 Hz 基频
        0.3 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz 谐波
        0.2 * np.sin(2 * np.pi * 150 * t)    # 150 Hz 谐波
    )

    # 故障特征（内圈故障）
    fault_signal = 0.5 * np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))

    # 合成信号
    signal = base_signal + fault_signal + noise_level * np.random.randn(length)

    # 创建双通道信号（X和Y方向）
    signal_x = signal
    signal_y = signal * 0.8 + 0.2 * np.random.randn(length)

    # 转换为张量并添加维度
    signal_tensor = torch.FloatTensor(np.stack([signal_x, signal_y], axis=-1))
    signal_tensor = signal_tensor.unsqueeze(0)  # 添加batch维度

    return signal_tensor


def demo_tspn_model_creation():
    """演示TSPN模型的创建和配置"""
    print("=" * 60)
    print("1. 创建TSPN模型")
    print("=" * 60)

    # 模拟TSPN模型配置
    class DemoTSPNConfig:
        def __init__(self):
            self.in_channels = 2
            self.out_channels = 64
            self.scale = 4
            self.skip_connection = True
            self.num_classes = 5  # 正常 + 4种故障类型
            self.device = 'cpu'
            self.layer1 = 'FFT'
            self.layer2 = 'HT'
            self.layer3 = 'WF'
            self.layer4 = 'I'

    config = DemoTSPNConfig()
    print(f"模型配置:")
    print(f"  输入通道: {config.in_channels}")
    print(f"  输出通道: {config.out_channels}")
    print(f"  缩放比例: {config.scale}")
    print(f"  信号处理层: {config.layer1} -> {config.layer2} -> {config.layer3} -> {config.layer4}")

    return config


def demo_signal_path_explanation():
    """演示信号路径解释功能"""
    print("\n" + "=" * 60)
    print("2. 信号路径解释")
    print("=" * 60)

    # 创建模拟信号
    signal_data = create_synthetic_signal(length=1000, noise_level=0.1)
    print(f"输入信号形状: {signal_data.shape}")
    print(f"信号统计: 均值={signal_data.mean():.4f}, 标准差={signal_data.std():.4f}")

    # 注意：这里我们使用一个简化的演示结构
    # 在实际使用中，您需要加载真实的TSPN模型
    print("\n注意：此示例使用演示数据。实际使用时，请加载预训练的TSPN模型。")

    # 模拟解释结果
    demo_explanation_data = {
        'signal_path': [
            {
                'layer_name': 'Layer 1 (FFT)',
                'operator_type': 'FFT',
                'input_stats': {'energy': 1.0, 'peak_freq': 50},
                'output_stats': {'energy': 1.2, 'dominant_freq': 150},
                'output_signal': torch.randn(1, 1000, 64)
            },
            {
                'layer_name': 'Layer 2 (HT)',
                'operator_type': 'Hilbert Transform',
                'input_stats': {'energy': 1.2, 'dominant_freq': 150},
                'output_stats': {'energy': 1.1, 'envelope_energy': 0.8},
                'output_signal': torch.randn(1, 1000, 64)
            },
            {
                'layer_name': 'Layer 3 (WF)',
                'operator_type': 'Wavelet Filter',
                'input_stats': {'energy': 1.1, 'envelope_energy': 0.8},
                'output_stats': {'energy': 0.9, 'filtered_energy': 0.6},
                'output_signal': torch.randn(1, 1000, 64)
            },
            {
                'layer_name': 'Layer 4 (I)',
                'operator_type': 'Identity',
                'input_stats': {'energy': 0.9, 'filtered_energy': 0.6},
                'output_stats': {'energy': 0.9, 'final_energy': 0.6},
                'output_signal': torch.randn(1, 1000, 64)
            }
        ],
        'transformation_summary': {
            'total_layers': 4,
            'overall_energy_change': -0.1,
            'dominant_frequency_shift': 100,
            'feature_extraction_gain': 0.3
        },
        'original_signal': signal_data
    }

    demo_meta_data = {
        'method': 'signal_path',
        'model_name': 'TSPN',
        'input_shape': list(signal_data.shape),
        'fault_type': 'inner_race'
    }

    # 创建解释对象
    from toolkit_integration.explainability.core.explanation import Explanation
    explanation = Explanation(demo_explanation_data, demo_meta_data)

    print(f"解释方法: {explanation.get_method_name()}")
    print(f"模型类型: {explanation.get_model_name()}")

    # 显示信号路径信息
    signal_path = explanation.get_data('signal_path')
    print(f"\n信号路径变换 (共{len(signal_path)}层):")
    for i, step in enumerate(signal_path):
        input_energy = step['input_stats'].get('energy', 0)
        output_energy = step['output_stats'].get('energy', 0)
        energy_change = output_energy - input_energy

        print(f"  第{i+1}层: {step['layer_name']} ({step['operator_type']})")
        print(f"    能量变化: {input_energy:.4f} -> {output_energy:.4f} (Δ{energy_change:+.4f})")

        if 'dominant_freq' in step['input_stats']:
            input_freq = step['input_stats']['dominant_freq']
            output_freq = step['output_stats'].get('dominant_freq', input_freq)
            print(f"    主频变化: {input_freq}Hz -> {output_freq}Hz")

    # 显示变换总结
    summary = explanation.get_data('transformation_summary')
    if summary:
        print(f"\n变换总结:")
        print(f"  总层数: {summary['total_layers']}")
        print(f"  整体能量变化: {summary['overall_energy_change']:+.4f}")
        print(f"  主频偏移: {summary['dominant_frequency_shift']}Hz")
        print(f"  特征提取增益: {summary['feature_extraction_gain']:.4f}")

    return explanation


def demo_visualization(explanation):
    """演示解释结果的可视化"""
    print("\n" + "=" * 60)
    print("3. 可视化信号路径")
    print("=" * 60)

    try:
        # 生成可视化
        fig = explanation.visualize(mode='path')

        # 设置标题
        plt.suptitle('TSPN信号路径解释 - 内圈故障诊断', fontsize=16, y=0.98)

        # 添加故障类型标注
        fig.text(0.02, 0.02, '故障类型: 内圈故障 | 采样频率: 1000Hz',
                fontsize=10, ha='left')

        plt.tight_layout()

        # 保存图片
        output_dir = Path('output/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'tspn_signal_path_explanation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"可视化已保存到: {save_path}")

        # 显示图片
        plt.show()

    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("这可能是因为缺少某些依赖库或图形环境")


def demo_operator_importance():
    """演示算子重要性分析"""
    print("\n" + "=" * 60)
    print("4. 算子重要性分析")
    print("=" * 60)

    # 模拟算子重要性分数
    importance_scores = {
        'FFT': {
            'energy_contribution': 0.35,
            'frequency_relevance': 0.42,
            'noise_reduction': 0.28,
            'combined_score': 0.35
        },
        'HT': {
            'energy_contribution': 0.28,
            'frequency_relevance': 0.25,
            'noise_reduction': 0.31,
            'combined_score': 0.28
        },
        'WF': {
            'energy_contribution': 0.22,
            'frequency_relevance': 0.18,
            'noise_reduction': 0.35,
            'combined_score': 0.25
        },
        'I': {
            'energy_contribution': 0.15,
            'frequency_relevance': 0.15,
            'noise_reduction': 0.06,
            'combined_score': 0.12
        }
    }

    print("各算子重要性分数:")
    for operator, scores in importance_scores.items():
        print(f"  {operator}:")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.3f}")
        print()

    # 生成重要性可视化
    try:
        operators = list(importance_scores.keys())
        scores = [importance_scores[op]['combined_score'] for op in operators]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(operators, scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])

        ax.set_title('TSPN算子重要性分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('重要性分数', fontsize=12)
        ax.set_xlabel('算子类型', fontsize=12)
        ax.set_ylim(0, 0.5)

        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存图片
        output_dir = Path('output/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'tspn_operator_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"算子重要性图已保存到: {save_path}")
        plt.show()

    except Exception as e:
        print(f"重要性可视化失败: {e}")


def demo_metrics_evaluation():
    """演示解释质量评估"""
    print("\n" + "=" * 60)
    print("5. 解释质量评估")
    print("=" * 60)

    # 模拟解释质量指标
    quality_metrics = {
        'completeness': 0.85,      # 完整性：解释覆盖了多少关键信息
        'understandability': 0.92, # 可理解性：用户理解程度
        'faithfulness': 0.78,      # 忠实性：解释与模型一致性
        'specificity': 0.81,       # 特异性：解释针对特定故障的能力
        'consistency': 0.88        # 一致性：相似输入的解释一致性
    }

    print("解释质量评估指标:")
    for metric, score in quality_metrics.items():
        print(f"  {metric:12s}: {score:.3f} ({'优秀' if score > 0.8 else '良好' if score > 0.6 else '一般'})")

    # 计算综合得分
    overall_score = sum(quality_metrics.values()) / len(quality_metrics)
    print(f"\n综合得分: {overall_score:.3f} ({'优秀' if overall_score > 0.8 else '良好' if overall_score > 0.6 else '一般'})")

    # 生成雷达图
    try:
        import matplotlib.pyplot as plt
        from math import pi

        metrics = list(quality_metrics.keys())
        scores = list(quality_metrics.values())

        # 角度计算
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # 闭合图形

        scores += scores[:1]  # 闭合图形

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # 绘制雷达图
        ax.plot(angles, scores, 'o-', linewidth=2, color='#2E86AB')
        ax.fill(angles, scores, alpha=0.25, color='#2E86AB')

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])

        # 设置标题和范围
        ax.set_title('TSPN解释质量评估雷达图', size=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        output_dir = Path('output/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'tspn_quality_radar.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"质量评估雷达图已保存到: {save_path}")
        plt.show()

    except Exception as e:
        print(f"雷达图生成失败: {e}")


def main():
    """主函数：运行完整的TSPN信号路径解释演示"""
    print("TSPN信号路径解释演示")
    print("=" * 80)

    # 1. 模型创建演示
    config = demo_tspn_model_creation()

    # 2. 信号路径解释
    explanation = demo_signal_path_explanation()

    # 3. 可视化演示
    demo_visualization(explanation)

    # 4. 算子重要性分析
    demo_operator_importance()

    # 5. 解释质量评估
    demo_metrics_evaluation()

    print("\n" + "=" * 80)
    print("演示完成！")
    print("\n关键要点:")
    print("1. TSPN的信号路径解释提供了从输入到输出的完整变换轨迹")
    print("2. 每个信号处理算子的贡献可以被量化和可视化")
    print("3. 解释质量可以通过多个维度进行评估")
    print("4. 可视化结果为工程师提供了直观的诊断依据")
    print("\n输出文件保存在 'output/figures/' 目录中")


if __name__ == "__main__":
    main()
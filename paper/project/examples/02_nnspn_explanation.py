#!/usr/bin/env python3
"""
NNSPN神经信号处理网络解释示例

本示例展示如何对神经信号处理网络(NNSPN)进行解释，
使用梯度和集成梯度方法分析模型的决策过程。

主要功能：
1. NNSPN模型解释
2. 梯度显著性分析
3. 集成梯度解释
4. 特征重要性评估
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


def create_complex_signal(length=2000, fault_type='outer_race'):
    """
    创建复杂的振动信号，模拟不同的故障类型

    Args:
        length: 信号长度
        fault_type: 故障类型 ('inner_race', 'outer_race', 'ball', 'normal')

    Returns:
        torch.Tensor: 模拟的振动信号 [1, length, 2]
    """
    t = np.linspace(0, 2, length)  # 2秒信号

    # 基础旋转频率 (30 Hz)
    rotation_freq = 30
    sample_freq = 1000

    # 基础信号
    base_signal = (
        1.0 * np.sin(2 * np.pi * rotation_freq * t) +          # 基频
        0.3 * np.sin(2 * np.pi * 2 * rotation_freq * t) +     # 2倍频
        0.15 * np.sin(2 * np.pi * 3 * rotation_freq * t)      # 3倍频
    )

    # 根据故障类型添加特定特征
    if fault_type == 'inner_race':
        # 内圈故障特征
        bpfi = rotation_freq * 4.2  # 内圈故障频率
        fault_signal = 0.6 * np.sin(2 * np.pi * bpfi * t) * (1 + 0.4 * np.sin(2 * np.pi * rotation_freq * t))
        envelope_freq = rotation_freq
    elif fault_type == 'outer_race':
        # 外圈故障特征
        bpfo = rotation_freq * 3.1  # 外圈故障频率
        fault_signal = 0.5 * np.sin(2 * np.pi * bpfo * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * rotation_freq * t))
        envelope_freq = 2 * rotation_freq
    elif fault_type == 'ball':
        # 滚动体故障特征
        bpf = rotation_freq * 2.8  # 滚动体故障频率
        fault_signal = 0.4 * np.sin(2 * np.pi * bpf * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * rotation_freq * t))
        envelope_freq = 3 * rotation_freq
    else:
        # 正常状态
        fault_signal = 0
        envelope_freq = 0

    # 添加噪声
    noise = 0.15 * np.random.randn(length)

    # 合成信号
    signal = base_signal + fault_signal + noise

    # 创建双通道信号
    signal_x = signal
    signal_y = signal * 0.9 + 0.1 * np.random.randn(length)

    # 转换为张量
    signal_tensor = torch.FloatTensor(np.stack([signal_x, signal_y], axis=-1))
    signal_tensor = signal_tensor.unsqueeze(0)  # 添加batch维度

    return signal_tensor


class DemoNNSPN(torch.nn.Module):
    """演示用的NNSPN模型"""
    def __init__(self, input_size=2000, num_classes=4):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # 信号编码器
        self.signal_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        # 特征提取器
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )

        # 分类器
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]

        # 信号编码
        encoded = self.signal_encoder(x)

        # 特征提取
        features = self.feature_extractor(encoded)

        # 分类
        logits = self.classifier(features)

        return logits


def demo_nnspn_model():
    """演示NNSPN模型的创建和结构"""
    print("=" * 60)
    print("1. NNSPN模型创建")
    print("=" * 60)

    # 创建模型
    model = DemoNNSPN(input_size=2000, num_classes=4)
    model.eval()

    print(f"模型类型: {type(model).__name__}")
    print(f"输入尺寸: {model.input_size}")
    print(f"类别数量: {model.num_classes}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 计算模型参数详细统计
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")

    return model


def demo_gradient_saliency(model, signal_data, target_class=1):
    """演示梯度显著性分析"""
    print("\n" + "=" * 60)
    print("2. 梯度显著性分析")
    print("=" * 60)

    # 创建解释器
    explainer = UnifiedExplainer(
        model=model,
        method='saliency',
        config={
            'abs_grad': True,  # 使用绝对梯度
            'normalize': True  # 归一化
        }
    )

    print(f"解释方法: {explainer.method}")
    print(f"目标类别: {target_class}")
    print(f"输入信号形状: {signal_data.shape}")

    # 生成解释
    try:
        explanation = explainer.explain(signal_data, target_class=target_class)
        print("✓ 梯度显著性解释生成成功")

        # 获取归因值
        attribution = explanation.get_attribution()
        if attribution is not None:
            print(f"归因值形状: {attribution.shape}")
            print(f"归因值统计: 均值={attribution.mean():.6f}, 最大值={attribution.max():.6f}")

            # 获取解释指标
            metrics = explanation.get_metrics()
            print("解释指标:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")

        return explanation

    except Exception as e:
        print(f"✗ 解释生成失败: {e}")
        return None


def demo_integrated_gradients(model, signal_data, target_class=1):
    """演示积分梯度分析"""
    print("\n" + "=" * 60)
    print("3. 积分梯度分析")
    print("=" * 60)

    # 创建解释器
    explainer = UnifiedExplainer(
        model=model,
        method='integrated_gradients',
        config={
            'n_steps': 25,           # 积分步数
            'baseline': 'zero',      # 基线设置
            'normalize': True,       # 归一化
            'internal_batch_size': 4  # 内部批大小
        }
    )

    print(f"解释方法: {explainer.method}")
    print(f"积分步数: {explainer.config.get('n_steps', 25)}")
    print(f"基线设置: {explainer.config.get('baseline', 'zero')}")

    # 生成解释
    try:
        explanation = explainer.explain(signal_data, target_class=target_class)
        print("✓ 积分梯度解释生成成功")

        # 获取归因值
        attribution = explanation.get_attribution()
        if attribution is not None:
            print(f"归因值形状: {attribution.shape}")
            print(f"归因值统计: 均值={attribution.mean():.6f}, 最大值={attribution.max():.6f}")

            # 计算额外的积分梯度指标
            if hasattr(attribution, 'sum'):
                total_attribution = attribution.sum()
                print(f"总归因值: {total_attribution:.6f}")

        return explanation

    except Exception as e:
        print(f"✗ 解释生成失败: {e}")
        return None


def demo_method_comparison(model, signal_data, target_class=1):
    """演示多种解释方法的比较"""
    print("\n" + "=" * 60)
    print("4. 解释方法比较")
    print("=" * 60)

    # 创建统一解释器
    explainer = UnifiedExplainer(model, method='auto')

    print("可用的解释方法:")
    available_methods = explainer.get_available_methods()
    for method, description in available_methods.items():
        print(f"  {method:20s}: {description}")

    # 比较不同方法
    methods_to_compare = ['saliency', 'integrated_gradients']

    try:
        comparisons = explainer.compare_methods(
            signal_data,
            target_class=target_class,
            methods=methods_to_compare
        )

        print("\n解释方法比较结果:")
        for method, explanation in comparisons.items():
            if explanation is not None:
                metrics = explanation.get_metrics()
                print(f"\n{method}:")
                print(f"  归因均值: {metrics.get('attribution_mean', 'N/A'):.6f}")
                print(f"  归因标准差: {metrics.get('attribution_std', 'N/A'):.6f}")
                print(f"  最大归因值: {metrics.get('attribution_max', 'N/A'):.6f}")
                print(f"  归因稀疏度: {metrics.get('attribution_sparsity', 'N/A'):.6f}")
            else:
                print(f"\n{method}: 解释生成失败")

        return comparisons

    except Exception as e:
        print(f"✗ 方法比较失败: {e}")
        return None


def visualize_explanations(explanations, signal_data):
    """可视化解释结果"""
    print("\n" + "=" * 60)
    print("5. 可视化解释结果")
    print("=" * 60)

    if explanations is None:
        print("没有有效的解释结果可以可视化")
        return

    # 创建输出目录
    output_dir = Path('output/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 获取信号数据
        signal_np = signal_data.detach().cpu().numpy().squeeze()  # [seq_len, 2]

        # 创建子图
        fig, axes = plt.subplots(len(explanations) + 1, 1, figsize=(15, 3 * (len(explanations) + 1)))

        if len(explanations) == 1:
            axes = [axes[0], axes[1]]  # 确保axes是列表

        # 绘制原始信号
        axes[0].plot(signal_np[:, 0], label='Channel X', alpha=0.8)
        axes[0].plot(signal_np[:, 1], label='Channel Y', alpha=0.8)
        axes[0].set_title('原始振动信号', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间点')
        axes[0].set_ylabel('幅值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 绘制解释结果
        for idx, (method, explanation) in enumerate(explanations.items(), 1):
            if explanation is not None:
                attribution = explanation.get_attribution()
                if attribution is not None:
                    # 确保归因值形状正确
                    attr_np = attribution.squeeze()
                    if attr_np.ndim == 2:
                        attr_np = attr_np.mean(axis=1)  # 多通道时取平均

                    axes[idx].plot(attr_np, label=f'{method} Attribution', color='red', alpha=0.8)
                    axes[idx].set_title(f'{method} 解释结果', fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel('时间点')
                    axes[idx].set_ylabel('归因值')
                    axes[idx].grid(True, alpha=0.3)

                    # 标记高归因值区域
                    threshold = np.percentile(np.abs(attr_np), 90)
                    high_attribution = np.where(np.abs(attr_np) > threshold)[0]
                    if len(high_attribution) > 0:
                        axes[idx].scatter(high_attribution, attr_np[high_attribution],
                                        color='red', s=20, alpha=0.6, label='高归因区域')
                        axes[idx].legend()

        plt.tight_layout()

        # 保存图片
        save_path = output_dir / 'nnspn_explanation_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"解释比较图已保存到: {save_path}")

        plt.show()

    except Exception as e:
        print(f"可视化失败: {e}")


def demo_feature_importance_analysis(model, signal_data, target_class=1):
    """演示特征重要性分析"""
    print("\n" + "=" * 60)
    print("6. 特征重要性分析")
    print("=" * 60)

    try:
        # 分析不同时间窗口的重要性
        window_size = 200
        step_size = 100
        seq_length = signal_data.shape[1]

        importance_scores = []
        window_positions = []

        for i in range(0, seq_length - window_size, step_size):
            # 提取时间窗口
            window_data = signal_data.clone()
            window_data[:, i:i+window_size, :] = 0  # 零化该窗口

            # 前向传播
            with torch.no_grad():
                original_output = model(signal_data)
                masked_output = model(window_data)

            # 计算重要性分数（输出变化）
            importance = torch.abs(original_output[0, target_class] - masked_output[0, target_class])
            importance_scores.append(importance.item())
            window_positions.append(i + window_size // 2)

        # 归一化重要性分数
        importance_scores = np.array(importance_scores)
        importance_scores = importance_scores / (importance_scores.max() + 1e-8)

        print("时间窗口重要性分析:")
        print(f"  窗口大小: {window_size}")
        print(f"  步长: {step_size}")
        print(f"  分析窗口数: {len(importance_scores)}")
        print(f"  最大重要性分数: {importance_scores.max():.4f}")
        print(f"  最重要窗口位置: {window_positions[np.argmax(importance_scores)]}")

        # 可视化
        try:
            plt.figure(figsize=(12, 6))
            plt.bar(window_positions, importance_scores, width=step_size*0.8, alpha=0.7, color='#2E86AB')
            plt.title('NNSPN时间窗口重要性分析', fontsize=14, fontweight='bold')
            plt.xlabel('时间位置')
            plt.ylabel('重要性分数')
            plt.grid(True, alpha=0.3)

            # 标记最重要的窗口
            max_idx = np.argmax(importance_scores)
            plt.bar(window_positions[max_idx], importance_scores[max_idx],
                   width=step_size*0.8, alpha=0.9, color='#A23B72')

            plt.tight_layout()

            # 保存图片
            output_dir = Path('output/figures')
            save_path = output_dir / 'nnspn_window_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间窗口重要性图已保存到: {save_path}")

            plt.show()

        except Exception as e:
            print(f"重要性可视化失败: {e}")

        return window_positions, importance_scores

    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        return None, None


def main():
    """主函数：运行完整的NNSPN解释演示"""
    print("NNSPN神经信号处理网络解释演示")
    print("=" * 80)

    # 1. 创建模型
    model = demo_nnspn_model()

    # 2. 创建测试信号
    fault_types = ['inner_race', 'outer_race', 'ball', 'normal']
    target_fault = fault_types[1]  # 外圈故障
    signal_data = create_complex_signal(length=2000, fault_type=target_fault)
    target_class = fault_types.index(target_fault)

    print(f"\n测试信号类型: {target_fault}")
    print(f"目标类别: {target_class}")

    # 3. 梯度显著性分析
    saliency_explanation = demo_gradient_saliency(model, signal_data, target_class)

    # 4. 积分梯度分析
    ig_explanation = demo_integrated_gradients(model, signal_data, target_class)

    # 5. 方法比较
    explanations = {}
    if saliency_explanation is not None:
        explanations['saliency'] = saliency_explanation
    if ig_explanation is not None:
        explanations['integrated_gradients'] = ig_explanation

    # 6. 可视化
    if explanations:
        visualize_explanations(explanations, signal_data)

    # 7. 特征重要性分析
    demo_feature_importance_analysis(model, signal_data, target_class)

    print("\n" + "=" * 80)
    print("演示完成！")
    print("\n关键要点:")
    print("1. NNSPN的梯度解释显示了模型决策的关键时间段")
    print("2. 积分梯度提供了更稳定和可信的归因结果")
    print("3. 不同解释方法可以相互验证和补充")
    print("4. 时间窗口重要性分析有助于理解模型的关注区域")
    print("\n输出文件保存在 'output/figures/' 目录中")


if __name__ == "__main__":
    main()
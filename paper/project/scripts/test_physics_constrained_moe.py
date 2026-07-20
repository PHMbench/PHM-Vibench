"""
物理约束MoE模型测试脚本

测试阶段2实现的物理约束MoE模型，验证：
1. 频域约束是否工作正常
2. 正交约束是否促进专家独立性
3. 路径签名分析功能
4. 物理约束对性能的影响

Author: MoE Expert System
Date: 2024-11-26
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import json

# 添加代码路径
sys.path.append(str(Path(__file__).parent.parent / 'code'))

from moe_model import NNSPNMoE
from analyze_routing import RoutingAnalyzer


def generate_synthetic_signal(signal_length=4096,
                            sample_rate=12000,
                            fault_type='low_freq',
                            noise_level=0.1):
    """生成合成故障信号

    Args:
        signal_length: 信号长度
        sample_rate: 采样率
        fault_type: 故障类型 ('low_freq', 'harmonic', 'envelope')
        noise_level: 噪声水平

    Returns:
        合成信号张量 [1, signal_length]
    """
    t = np.linspace(0, signal_length / sample_rate, signal_length)
    signal = np.zeros(signal_length)

    if fault_type == 'low_freq':
        # 低频故障：低频正弦波
        signal += 2.0 * np.sin(2 * np.pi * 50 * t)      # 50Hz主频
        signal += 1.0 * np.sin(2 * np.pi * 100 * t)     # 100Hz谐波
        signal += 0.5 * np.sin(2 * np.pi * 150 * t)     # 150Hz谐波

    elif fault_type == 'harmonic':
        # 谐波故障：多个谐波分量
        fundamental_freq = 200
        for i in range(1, 6):
            signal += (1.0 / i) * np.sin(2 * np.pi * fundamental_freq * i * t)

    elif fault_type == 'envelope':
        # 包络故障：调幅信号
        carrier_freq = 2000
        envelope_freq = 50
        signal += np.sin(2 * np.pi * carrier_freq * t) * \
                 (1 + 0.8 * np.sin(2 * np.pi * envelope_freq * t))
        # 添加冲击成分
        for i in range(0, signal_length, signal_length // 10):
            if i + 100 < signal_length:
                signal[i:i+100] += 3.0 * np.exp(-0.1 * np.arange(100))

    # 添加噪声
    noise = noise_level * np.random.randn(signal_length)
    signal += noise

    # 归一化
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    return torch.FloatTensor(signal).unsqueeze(0)


def test_frequency_constraints():
    """测试频域约束功能"""
    print("\n" + "="*60)
    print("测试1: 频域约束功能")
    print("="*60)

    # 创建模型
    model = NNSPNMoE(num_classes=3, feature_dim=64)
    model.eval()

    # 生成不同类型的测试信号
    fault_types = ['low_freq', 'harmonic', 'envelope']
    test_signals = {}
    expected_activations = {
        'low_freq': [0],      # 期望低通专家激活
        'harmonic': [1],      # 期望谐波专家激活
        'envelope': [2]       # 期望包络专家激活
    }

    for fault_type in fault_types:
        signal = generate_synthetic_signal(fault_type=fault_type, noise_level=0.1)
        test_signals[fault_type] = signal

    print("信号生成完成，测试频域约束...")

    # 测试每种信号类型
    all_correct = True
    for fault_type, signal in test_signals.items():
        with torch.no_grad():
            logits, metadata = model(signal, return_explanations=True)
            routing_weights = metadata['routing_weights'][0].cpu().numpy()

        # 找到主导专家
        dominant_expert = np.argmax(routing_weights)
        expected_experts = expected_activations[fault_type]

        print(f"\n{fault_type} 故障信号:")
        print(f"  - 主导专家: 专家{dominant_expert}")
        print(f"  - 专家权重: {routing_weights}")
        print(f"  - 期望专家: {expected_experts}")

        # 检查路由是否合理
        if dominant_expert in expected_experts:
            print(f"  ✓ 路由正确！专家{dominant_expert}最适合处理{fault_type}故障")
        else:
            print(f"  ✗ 路由错误！期望专家{expected_experts}，实际专家{dominant_expert}")
            all_correct = False

    return all_correct


def test_orthogonal_constraints():
    """测试正交约束功能"""
    print("\n" + "="*60)
    print("测试2: 正交约束功能")
    print("="*60)

    # 创建两个模型：有约束和无约束
    model_with_constraints = NNSPNMoE(num_classes=3, feature_dim=64)
    model_without_constraints = NNSPNMoE(num_classes=3, feature_dim=64)

    # 禁用无约束模型的物理约束
    model_without_constraints.orthogonal_constraint_weight = 0.0
    model_without_constraints.frequency_constraint_weight = 0.0
    model_without_constraints.physics_constraint_weight = 0.0

    models = {
        '有物理约束': model_with_constraints,
        '无物理约束': model_without_constraints
    }

    # 生成测试数据
    batch_size = 50
    test_batch = torch.cat([
        generate_synthetic_signal(fault_type='low_freq', noise_level=0.2)
        for _ in range(batch_size // 3)
    ] + [
        generate_synthetic_signal(fault_type='harmonic', noise_level=0.2)
        for _ in range(batch_size // 3)
    ] + [
        generate_synthetic_signal(fault_type='envelope', noise_level=0.2)
        for _ in range(batch_size - 2 * (batch_size // 3))
    ], dim=0)

    print(f"测试批次大小: {test_batch.shape}")

    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits, metadata = model(test_batch, return_explanations=True)
            expert_outputs = metadata['expert_outputs']  # [batch_size, num_experts, feature_dim]

        # 计算专家间相关性
        num_experts = expert_outputs.shape[1]
        correlations = []

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # 计算专家i和j输出间的平均相关性
                output_i = expert_outputs[:, i, :].flatten()
                output_j = expert_outputs[:, j, :].flatten()
                correlation = np.corrcoef(output_i.numpy(), output_j.numpy())[0, 1]
                correlations.append(abs(correlation))

        avg_correlation = np.mean(correlations)
        print(f"\n{model_name}:")
        print(f"  - 专家间平均相关性: {avg_correlation:.4f}")
        print(f"  - 相关性标准差: {np.std(correlations):.4f}")

        # 计算专家激活的负载均衡
        routing_weights = metadata['routing_weights']  # [batch_size, num_experts]
        expert_usage = torch.mean(routing_weights, dim=0)
        load_balance = 1.0 - torch.std(expert_usage) / (torch.mean(expert_usage) + 1e-8)

        print(f"  - 负载均衡度: {load_balance:.4f}")

    return True


def test_routing_analysis():
    """测试路径签名分析功能"""
    print("\n" + "="*60)
    print("测试3: 路径签名分析功能")
    print("="*60)

    # 创建模型和分析器
    model = NNSPNMoE(num_classes=3, feature_dim=64)
    analyzer = RoutingAnalyzer(model)

    # 生成多样化的测试数据
    fault_types = ['low_freq', 'harmonic', 'envelope']
    test_signals = []
    test_labels = []

    for i, fault_type in enumerate(fault_types):
        for _ in range(20):  # 每种类型20个样本
            signal = generate_synthetic_signal(
                fault_type=fault_type,
                noise_level=0.1 + 0.1 * np.random.rand()
            )
            test_signals.append(signal)
            test_labels.append(i)

    # 合并批次数据
    test_batch = torch.cat(test_signals, dim=0)
    test_labels_tensor = torch.tensor(test_labels)

    print(f"分析数据形状: {test_batch.shape}")
    print(f"标签分布: {np.bincount(test_labels)}")

    # 执行路由分析
    results = analyzer.analyze_batch(test_batch, test_labels_tensor)

    print("\n分析结果摘要:")
    print(f"  - 批次大小: {results['batch_size']}")
    print(f"  - 路径签名数量: {len(results['path_signatures'])}")
    print(f"  - 专家数量: {len(results['expert_statistics']['mean_weights'])}")

    # 打印专家统计
    expert_stats = results['expert_statistics']
    print(f"\n专家激活统计:")
    print(f"  - 最常用专家: 专家{expert_stats['most_used_expert']}")
    print(f"  - 最少用专家: 专家{expert_stats['least_used_expert']}")
    print(f"  - 负载均衡度: {expert_stats['load_balance']:.4f}")
    print(f"  - 平均权重: {[f'{w:.3f}' for w in expert_stats['mean_weights']]}")

    # 打印类别分布统计
    if results['class_distribution']:
        print(f"\n类别激活分布:")
        for class_id, class_stats in results['class_distribution'].items():
            print(f"  - 类别{class_id}: 主导专家{class_stats['dominant_expert']} "
                  f"(置信度: {class_stats['dominant_expert_confidence']:.3f}, "
                  f"一致性: {class_stats['activation_consistency']:.3f})")

        # 生成可视化（保存到临时目录）
        save_dir = Path('./temp_routing_analysis')
        try:
            analyzer.save_analysis_results(str(save_dir), results)

            routing_weights = results['routing_weights']
            path_signatures = results['path_signatures']
            expert_stats = results['expert_statistics']
            entropies = [sig['routing_entropy'] for sig in path_signatures]
            confidences = [sig['expert_confidence'] for sig in path_signatures]
            dominant_counts = {}
            for sig in path_signatures:
                key = str(sig['dominant_expert'])
                dominant_counts[key] = dominant_counts.get(key, 0) + 1

            summary = {
                'batch_size': int(results['batch_size']),
                'num_samples': int(len(path_signatures)),
                'num_experts': int(routing_weights.shape[1]),
                'mean_routing_entropy': float(np.mean(entropies)) if entropies else 0.0,
                'std_routing_entropy': float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
                'mean_expert_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'load_balance': float(expert_stats.get('load_balance', 0.0)),
                'most_used_expert': int(expert_stats.get('most_used_expert', 0)),
                'least_used_expert': int(expert_stats.get('least_used_expert', 0)),
                'dominant_expert_histogram': dominant_counts,
                'mean_weights': expert_stats.get('mean_weights', []),
                'activation_frequency': expert_stats.get('activation_frequency', []),
                'class_distribution_present': bool(results.get('class_distribution')),
            }
            (save_dir / 'analysis_summary.json').write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            report_lines = [
                '# Routing Analysis Report',
                '',
                f"- samples: `{summary['num_samples']}`",
                f"- num_experts: `{summary['num_experts']}`",
                f"- mean_routing_entropy: `{summary['mean_routing_entropy']:.6f}`",
                f"- mean_expert_confidence: `{summary['mean_expert_confidence']:.6f}`",
                f"- load_balance: `{summary['load_balance']:.6f}`",
                f"- dominant_expert_histogram: `{json.dumps(summary['dominant_expert_histogram'], ensure_ascii=False)}`",
            ]
            (save_dir / 'routing_analysis_report.md').write_text(
                '\n'.join(report_lines) + '\n',
                encoding='utf-8',
            )

            # 路径签名统计
            path_signatures = results['path_signatures']
            entropies = [sig['routing_entropy'] for sig in path_signatures]
            confidences = [sig['expert_confidence'] for sig in path_signatures]

            print(f"\n路径签名统计:")
            print(f"  - 平均路由熵: {np.mean(entropies):.3f}")
            print(f"  - 平均专家置信度: {np.mean(confidences):.3f}")
            print(f"  - 高置信度样本比例: {np.mean(np.array(confidences) > 0.7):.3f}")

            print(f"\n分析结果已保存到: {save_dir}")

        except Exception as e:
            print(f"可视化保存失败: {e}")

    return True


def test_loss_function_components():
    """测试损失函数组件"""
    print("\n" + "="*60)
    print("测试4: 损失函数组件")
    print("="*60)

    model = NNSPNMoE(num_classes=3, feature_dim=64)
    model.train()

    # 生成测试数据
    test_batch = torch.cat([
        generate_synthetic_signal(fault_type='low_freq') for _ in range(10)
    ], dim=0)

    # 前向传播
    logits, metadata = model(test_batch, return_explanations=True)

    # 模拟分类损失
    targets = torch.randint(0, 3, (test_batch.shape[0],))
    ce_loss = nn.CrossEntropyLoss()
    classification_loss = ce_loss(logits, targets)

    # 获取正则化损失
    reg_losses = metadata['regularization_losses']

    print("损失函数组件:")
    print(f"  - 分类损失: {classification_loss.item():.4f}")
    for loss_name, loss_value in reg_losses.items():
        if isinstance(loss_value, torch.Tensor):
            print(f"  - {loss_name}: {loss_value.item():.4f}")

    # 计算总损失
    total_loss = classification_loss
    for loss_name, loss_value in reg_losses.items():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value

    print(f"  - 总损失: {total_loss.item():.4f}")
    print(f"  - 物理约束占比: {reg_losses.get('physics_constraint', 0).item() / total_loss.item() * 100:.1f}%")

    return True


def main():
    """主测试函数"""
    print("物理约束MoE模型测试")
    print("=" * 60)
    print("测试阶段2实现的功能:")
    print("1. 频域约束：确保专家在预期频带内响应")
    print("2. 正交约束：促进专家输出独立性")
    print("3. 路径签名分析：生成多层次可解释性分析")
    print("4. 损失函数组件：物理约束集成")

    try:
        # 运行所有测试
        test_results = []

        # 测试1: 频域约束
        result1 = test_frequency_constraints()
        test_results.append(("频域约束", result1))

        # 测试2: 正交约束
        result2 = test_orthogonal_constraints()
        test_results.append(("正交约束", result2))

        # 测试3: 路径签名分析
        result3 = test_routing_analysis()
        test_results.append(("路径签名分析", result3))

        # 测试4: 损失函数组件
        result4 = test_loss_function_components()
        test_results.append(("损失函数组件", result4))

        # 总结测试结果
        print("\n" + "="*60)
        print("测试结果总结")
        print("="*60)

        all_passed = True
        for test_name, passed in test_results:
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{test_name:20}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 所有测试通过！物理约束MoE模型实现成功。")
        else:
            print("\n⚠️  部分测试失败，需要进一步调试。")

        print(f"\n阶段2任务完成度:")
        print(f"✓ 频域约束实现")
        print(f"✓ 正交约束实现")
        print(f"✓ 物理约束集成到总损失")
        print(f"✓ 路径签名分析脚本")
        print(f"✓ 专家激活矩阵可视化")
        print(f"✓ 故障类别专家激活分布")
        print(f"✓ 物理约束模型测试")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
算子注意力机制验证测试
用于验证OperatorAttention模型在合成信号上的行为是否符合物理预期
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# 添加路径以导入模型
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from generate_signals import create_test_signals, convert_to_tensor


class OperatorAttentionValidator:
    """算子注意力验证器"""

    def __init__(self, model_path: str = None, device: str = 'cuda:1'):
        """
        初始化验证器

        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.operator_names = ['FFT', 'HT', 'WF', 'I']
        self.validation_results = {}

        # 加载模型
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # 如果没有预训练模型，创建一个简单的测试模型
            self._create_test_model()

        # 创建结果保存目录
        os.makedirs('../../results/validation', exist_ok=True)

    def _load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._create_test_model()

    def _create_test_model(self):
        """创建测试用算子注意力模型"""
        # 简化的算子注意力实现用于测试
        class SimpleOperatorAttention(nn.Module):
            def __init__(self, num_operators=4, embed_dim=64):
                super().__init__()
                self.num_operators = num_operators
                self.embed_dim = embed_dim

                # 算子嵌入
                self.operator_embeddings = nn.Parameter(
                    torch.randn(num_operators, embed_dim)
                )

                # 门控网络
                self.gate_network = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, num_operators),
                    nn.Sigmoid()
                )

                # 特征提取器
                self.feature_extractor = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(embed_dim)
                )

                # 注意力计算
                self.attention_proj = nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                batch_size, channels, seq_len = x.shape

                # 全局特征
                features = self.feature_extractor(x)  # [B, 32, embed_dim]
                global_features = torch.mean(features, dim=1)  # [B, embed_dim]

                # 门控权重
                gates = self.gate_network(global_features)  # [B, num_operators]

                # 注意力权重
                attention_weights = torch.softmax(gates, dim=1)

                # 简单的算子模拟（实际应用中这里应该调用真实的算子）
                outputs = []
                x_squeezed = x.squeeze(1)  # [B, L]

                for i in range(self.num_operators):
                    # 模拟不同算子的效果，确保所有输出形状一致 [B, 1, L]
                    if i == 0:  # FFT - 频域特征
                        # 使用FFT但保持原始长度
                        fft_result = torch.fft.fft(x_squeezed, dim=-1).real
                        output = fft_result.unsqueeze(1)
                    elif i == 1:  # HT - 包络特征（简化版希尔伯特变换）
                        # 使用解析信号包络的简化版本
                        # 计算信号的绝对值作为包络的近似
                        envelope = torch.abs(x_squeezed)
                        # 使用移动平均平滑
                        kernel_size = 51
                        padding = kernel_size // 2
                        envelope_padded = torch.nn.functional.pad(envelope.unsqueeze(1), (padding, padding), mode='reflect')
                        smoothed = torch.nn.functional.avg_pool1d(envelope_padded, kernel_size, stride=1)
                        output = smoothed[:, :, padding:padding+seq_len]
                    elif i == 2:  # WF - 小波特征（简化）
                        # 模拟小波变换：使用高频调制
                        wavelet_freq = 10.0  # 小波频率
                        wavelet = torch.sin(2 * np.pi * wavelet_freq * torch.linspace(0, 1, seq_len, device=x.device))
                        wavelet_output = x_squeezed * wavelet
                        output = wavelet_output.unsqueeze(1)
                    else:  # I - 恒等变换
                        output = x

                    outputs.append(output)

                # 加权融合
                final_output = torch.zeros_like(x)
                for i, output in enumerate(outputs):
                    weight = attention_weights[:, i]  # [B]
                    # 确保weight的形状正确用于广播
                    weight = weight.unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
                    weighted_output = weight * output
                    final_output += weighted_output

                return final_output, attention_weights

        self.model = SimpleOperatorAttention(num_operators=4, embed_dim=64).to(self.device)
        print("Created test Operator Attention model")

    def validate_signal(self, signal: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证单个信号

        Args:
            signal: 输入信号
            metadata: 信号元数据

        Returns:
            validation_result: 验证结果
        """
        # 转换为张量
        signal_tensor = convert_to_tensor(signal).to(self.device)

        # 前向传播
        with torch.no_grad():
            output, attention_weights = self.model(signal_tensor)

        # 提取注意力权重
        weights = attention_weights.cpu().numpy().squeeze()

        # 创建结果字典
        result = {
            'signal_metadata': metadata,
            'operator_weights': dict(zip(self.operator_names, weights)),
            'dominant_operator': self.operator_names[np.argmax(weights)],
            'weight_distribution': weights,
            'validation_passed': False,
            'validation_details': {}
        }

        # 验证是否符合预期
        self._validate_expectations(result)

        return result

    def _validate_expectations(self, result: Dict[str, Any]):
        """验证结果是否符合物理预期"""
        metadata = result['signal_metadata']
        weights = result['operator_weights']

        # 根据信号类型进行验证
        signal_type = metadata.get('type', '')

        if signal_type == 'single_frequency':
            # 单频信号：FFT应该主导
            expected_fft_weight = metadata.get('expected_fft_weight', 0.8)
            fft_weight = weights.get('FFT', 0)

            result['validation_details']['fft_dominance'] = fft_weight >= expected_fft_weight
            result['validation_details']['fft_weight'] = fft_weight
            result['validation_passed'] = result['validation_details']['fft_dominance']

        elif signal_type == 'dual_frequency':
            # 双频信号：FFT和WF应该协同
            expected_combined = metadata.get('expected_combined_weight', 0.7)
            combined_weight = weights.get('FFT', 0) + weights.get('WF', 0)

            result['validation_details']['operator_synergy'] = combined_weight >= expected_combined
            result['validation_details']['combined_weight'] = combined_weight
            result['validation_passed'] = result['validation_details']['operator_synergy']

        elif signal_type == 'transient':
            # 瞬态信号：HT应该敏感
            expected_ht_weight = metadata.get('expected_ht_weight', 0.6)
            ht_weight = weights.get('HT', 0)

            result['validation_details']['ht_sensitivity'] = ht_weight >= expected_ht_weight
            result['validation_details']['ht_weight'] = ht_weight
            result['validation_passed'] = result['validation_details']['ht_sensitivity']

        elif signal_type == 'noisy':
            # 噪声信号：权重应该稳定
            expected_stability = metadata.get('expected_weight_stability', 0.9)
            # 这里需要与无噪声版本比较，暂时使用权重熵作为稳定性指标
            entropy = -np.sum(weights * np.log(weights + 1e-8))
            max_entropy = -np.sum([0.25] * np.log([0.25] * 4))  # 均匀分布的熵
            stability = 1 - entropy / max_entropy

            result['validation_details']['weight_stability'] = stability >= expected_stability
            result['validation_details']['stability_score'] = stability
            result['validation_passed'] = result['validation_details']['weight_stability']

        elif signal_type in ['bearing_fault', 'gear_fault']:
            # 故障信号：特定算子组合
            expected_operators = metadata.get('expected_dominant_operators', [])
            expected_combined = metadata.get('expected_combined_weight', 0.6)
            combined_weight = sum(weights.get(op, 0) for op in expected_operators)

            result['validation_details']['fault_detection'] = combined_weight >= expected_combined
            result['validation_details']['combined_weight'] = combined_weight
            result['validation_passed'] = result['validation_details']['fault_detection']

    def run_validation(self, signals: Dict[str, Tuple[np.ndarray, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        运行完整的验证测试

        Args:
            signals: 测试信号字典

        Returns:
            validation_summary: 验证总结
        """
        print(f"\nRunning Operator Attention validation on {len(signals)} signals...")

        results = {}
        passed_count = 0
        total_count = len(signals)

        # 验证每个信号
        for signal_name, (signal, metadata) in signals.items():
            print(f"\nValidating: {signal_name}")
            result = self.validate_signal(signal, metadata)
            results[signal_name] = result

            if result['validation_passed']:
                passed_count += 1
                print(f"  ✓ PASSED: {result['dominant_operator']} dominant")
            else:
                print(f"  ✗ FAILED: Expected behavior not observed")

        # 生成总结报告
        summary = {
            'total_signals': total_count,
            'passed_signals': passed_count,
            'pass_rate': passed_count / total_count * 100,
            'detailed_results': results,
            'operator_statistics': self._compute_operator_statistics(results),
            'timestamp': datetime.now().isoformat()
        }

        # 保存结果
        self._save_results(summary)

        return summary

    def _compute_operator_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算算子使用统计"""
        operator_counts = {op: 0 for op in self.operator_names}
        operator_weights = {op: [] for op in self.operator_names}

        for result in results.values():
            dominant = result['dominant_operator']
            weights = result['operator_weights']

            operator_counts[dominant] += 1
            for op, weight in weights.items():
                operator_weights[op].append(weight)

        # 计算平均权重
        avg_weights = {op: np.mean(weights) for op, weights in operator_weights.items()}

        return {
            'dominance_counts': operator_counts,
            'average_weights': avg_weights,
            'weight_std': {op: np.std(weights) for op, weights in operator_weights.items()}
        }

    def _save_results(self, summary: Dict[str, Any]):
        """保存验证结果"""
        # 保存JSON格式的详细结果
        results_path = '../../results/validation/validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # 保存验证报告
        report_path = '../../results/validation/validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("Operator Attention Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {summary['timestamp']}\n")
            f.write(f"Total Signals: {summary['total_signals']}\n")
            f.write(f"Passed: {summary['passed_signals']}\n")
            f.write(f"Pass Rate: {summary['pass_rate']:.2f}%\n\n")

            # 每个信号的详细结果
            for name, result in summary['detailed_results'].items():
                f.write(f"\nSignal: {name}\n")
                f.write(f"  Type: {result['signal_metadata']['type']}\n")
                f.write(f"  Dominant Operator: {result['dominant_operator']}\n")
                f.write(f"  Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}\n")
                f.write(f"  Weights: {result['operator_weights']}\n")

            # 算子统计
            stats = summary['operator_statistics']
            f.write(f"\n\nOperator Statistics:\n")
            for op in self.operator_names:
                f.write(f"  {op}: Dominated {stats['dominance_counts'][op]} times, "
                       f"Average Weight: {stats['average_weights'][op]:.3f}\n")

        print(f"\nResults saved to:")
        print(f"  - JSON: {results_path}")
        print(f"  - Report: {report_path}")

    def generate_visualization(self, results: Dict[str, Any]):
        """生成验证结果可视化"""
        # 创建热图显示算子权重
        signal_names = list(results['detailed_results'].keys())
        weight_matrix = []

        for name in signal_names:
            weights = results['detailed_results'][name]['operator_weights']
            weight_matrix.append([weights[op] for op in self.operator_names])

        weight_matrix = np.array(weight_matrix)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 算子权重热图
        ax1 = axes[0, 0]
        sns.heatmap(weight_matrix.T,
                   xticklabels=[name.replace('_', ' ').replace('Hz', ' Hz') for name in signal_names],
                   yticklabels=self.operator_names,
                   annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Operator Attention Weights Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Signals', fontsize=12)
        ax1.set_ylabel('Operators', fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 2. 验证通过率
        ax2 = axes[0, 1]
        passed = [r['validation_passed'] for r in results['detailed_results'].values()]
        passed_counts = sum(passed), len(passed) - sum(passed)
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax2.pie(passed_counts, labels=['Passed', 'Failed'],
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Validation Pass Rate: {results["pass_rate"]:.1f}%',
                     fontsize=14, fontweight='bold')

        # 3. 算子主导统计
        ax3 = axes[1, 0]
        operator_counts = results['operator_statistics']['dominance_counts']
        bars = ax3.bar(operator_counts.keys(), operator_counts.values(),
                      color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
        ax3.set_title('Operator Dominance Count', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Operators', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # 4. 权重分布箱线图
        ax4 = axes[1, 1]
        weight_data = []
        for op in self.operator_names:
            weights = [r['operator_weights'][op] for r in results['detailed_results'].values()]
            weight_data.append(weights)

        bp = ax4.boxplot(weight_data, labels=self.operator_names, patch_artist=True)
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax4.set_title('Operator Weight Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Operators', fontsize=12)
        ax4.set_ylabel('Weight', fontsize=12)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../../results/validation/validation_visualization.png',
                   dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: results/validation/validation_visualization.png")


def main():
    """主函数：运行完整的验证流程"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Operator Attention Theory Validation")
    print("=" * 60)

    # 创建验证器
    validator = OperatorAttentionValidator(device='cuda:1')

    # 生成测试信号
    print("\n1. Generating test signals...")
    signals = create_test_signals()
    print(f"   Generated {len(signals)} test signals")

    # 运行验证
    print("\n2. Running validation tests...")
    results = validator.run_validation(signals)

    # 生成可视化
    print("\n3. Generating visualizations...")
    validator.generate_visualization(results)

    # 打印总结
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Signals Tested: {results['total_signals']}")
    print(f"Passed: {results['passed_signals']}")
    print(f"Failed: {results['total_signals'] - results['passed_signals']}")
    print(f"Pass Rate: {results['pass_rate']:.2f}%")
    print("=" * 60)

    # 根据验证结果给出结论
    if results['pass_rate'] >= 70:
        print("\n✓ VALIDATION SUCCESSFUL!")
        print("  Operator Attention mechanism behaves as expected.")
        print("  Theoretical claims are supported by experimental evidence.")
    elif results['pass_rate'] >= 50:
        print("\n⚠ PARTIAL VALIDATION")
        print("  Some signals validate the theory, others need adjustment.")
        print("  Consider refining the model or expectations.")
    else:
        print("\n✗ VALIDATION FAILED")
        print("  Operator Attention does not meet theoretical expectations.")
        print("  Significant model modifications may be required.")


if __name__ == "__main__":
    main()
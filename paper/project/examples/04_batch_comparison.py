#!/usr/bin/env python3
"""
批量解释与比较示例

本示例展示如何对多个模型和解释方法进行批量比较，
包括性能评估和质量对比。

主要功能：
1. 批量解释生成
2. 多模型比较
3. 解释质量评估
4. 性能基准测试
"""

import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from toolkit_integration.explainability import UnifiedExplainer


class DemoBatchExplainer:
    """批量解释管理器"""

    def __init__(self):
        self.models = {}
        self.explainers = {}
        self.results = {}
        self.metrics_cache = {}

    def register_model(self, name: str, model: torch.nn.Module, model_type: str = 'generic'):
        """注册模型"""
        self.models[name] = {
            'model': model,
            'type': model_type,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        print(f"✓ 注册模型 '{name}' (参数量: {self.models[name]['parameters']:,})")

    def create_explainers(self, methods: List[str] = None):
        """为所有模型创建解释器"""
        if methods is None:
            methods = ['saliency', 'integrated_gradients', 'deeplift']

        for model_name, model_info in self.models.items():
            self.explainers[model_name] = {}
            for method in methods:
                try:
                    explainer = UnifiedExplainer(
                        model_info['model'],
                        method=method,
                        config={
                            'baseline': 'zero',
                            'n_steps': 20,
                            'normalize': True
                        }
                    )
                    self.explainers[model_name][method] = explainer
                    print(f"  ✓ {model_name}.{method} 解释器创建成功")
                except Exception as e:
                    print(f"  ✗ {model_name}.{method} 解释器创建失败: {e}")

    def batch_explain(self, test_data: torch.Tensor, target_classes: List[int] = None):
        """批量生成解释"""
        print("\n🚀 开始批量解释生成...")

        total_explanations = 0
        start_time = time.time()

        for model_name, explainers in self.explainers.items():
            self.results[model_name] = {}
            print(f"\n📊 处理模型: {model_name}")

            for method_name, explainer in explainers.items():
                try:
                    # 单个样本解释
                    start_method_time = time.time()
                    explanation = explainer.explain(test_data[0:1], target_class=target_classes[0] if target_classes else None)
                    method_time = time.time() - start_method_time

                    # 批量解释
                    start_batch_time = time.time()
                    batch_explanations = explainer.explain_batch(test_data, target_classes)
                    batch_time = time.time() - start_batch_time

                    self.results[model_name][method_name] = {
                        'single_explanation': explanation,
                        'batch_explanations': batch_explanations,
                        'single_time': method_time,
                        'batch_time': batch_time,
                        'metrics': self._compute_explanation_metrics(explanation, batch_explanations)
                    }

                    total_explanations += len(batch_explanations) + 1

                    print(f"  ✓ {method_name}: 单次{method_time:.3f}s, 批量{batch_time:.3f}s")

                except Exception as e:
                    print(f"  ✗ {method_name}: 解释生成失败 - {e}")
                    self.results[model_name][method_name] = None

        total_time = time.time() - start_time
        print(f"\n✅ 批量解释完成!")
        print(f"   总解释数: {total_explanations}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   平均每解释: {total_time/total_explanations:.4f}s")

    def _compute_explanation_metrics(self, single_explanation, batch_explanations):
        """计算解释质量指标"""
        metrics = {}

        if single_explanation:
            single_metrics = single_explanation.get_metrics()
            metrics.update({f"single_{k}": v for k, v in single_metrics.items()})

        if batch_explanations:
            batch_metrics = []
            for exp in batch_explanations:
                if exp:
                    batch_metrics.append(exp.get_metrics())

            if batch_metrics:
                # 计算批量指标的平均值
                avg_metrics = {}
                for key in batch_metrics[0].keys():
                    avg_metrics[f"avg_batch_{key}"] = np.mean([m[key] for m in batch_metrics])
                    avg_metrics[f"std_batch_{key}"] = np.std([m[key] for m in batch_metrics])
                metrics.update(avg_metrics)

        return metrics

    def compare_explanations(self):
        """比较不同解释方法的结果"""
        print("\n📈 解释质量比较分析")
        print("=" * 60)

        # 按模型组织比较结果
        for model_name, model_results in self.results.items():
            print(f"\n🔍 模型: {model_name}")
            print("-" * 40)

            valid_results = {k: v for k, v in model_results.items() if v is not None}
            if not valid_results:
                print("  没有有效的解释结果")
                continue

            # 性能比较
            print("  ⏱️  性能指标:")
            for method, result in valid_results.items():
                single_time = result.get('single_time', 0)
                batch_time = result.get('batch_time', 0)
                print(f"    {method:20s}: 单次{single_time:.3f}s, 批量{batch_time:.3f}s")

            # 质量指标比较
            print("  📊 质量指标 (单次解释):")
            for method, result in valid_results.items():
                metrics = result.get('metrics', {})
                attribution_mean = metrics.get('single_attribution_mean', 0)
                attribution_max = metrics.get('single_attribution_max', 0)
                attribution_sparsity = metrics.get('single_attribution_sparsity', 0)
                print(f"    {method:20s}: 均值{attribution_mean:.4f}, 最大值{attribution_max:.4f}, 稀疏度{attribution_sparsity:.4f}")

    def benchmark_performance(self, test_sizes: List[int] = None):
        """性能基准测试"""
        if test_sizes is None:
            test_sizes = [1, 4, 8, 16, 32]

        print("\n🏃 性能基准测试")
        print("=" * 60)

        benchmark_results = {}

        # 选择第一个有效模型和解释器进行测试
        if not self.explainers:
            print("没有可用的解释器进行基准测试")
            return benchmark_results

        model_name = list(self.explainers.keys())[0]
        method_name = list(self.explainers[model_name].keys())[0]
        explainer = self.explainers[model_name][method_name]

        print(f"使用 {model_name}.{method_name} 进行基准测试")

        for batch_size in test_sizes:
            # 创建测试数据
            test_data = torch.randn(batch_size, 1000, 2)
            target_classes = [0] * batch_size

            # 预热
            try:
                explainer.explain(test_data[0:1])
            except:
                pass

            # 性能测试
            times = []
            for _ in range(3):  # 重复3次取平均
                start_time = time.time()
                try:
                    explainer.explain_batch(test_data, target_classes)
                    times.append(time.time() - start_time)
                except:
                    times.append(float('inf'))

            avg_time = np.mean(times)
            throughput = batch_size / avg_time if avg_time != float('inf') else 0

            benchmark_results[batch_size] = {
                'avg_time': avg_time,
                'throughput': throughput,
                'success_rate': sum(1 for t in times if t != float('inf')) / len(times)
            }

            print(f"  批大小 {batch_size:2d}: 平均时间 {avg_time:.3f}s, 吞吐量 {throughput:.1f} 样本/秒")

        return benchmark_results

    def save_results(self, output_dir: Path = None):
        """保存比较结果"""
        if output_dir is None:
            output_dir = Path('output/batch_results')
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存原始结果
        results_file = output_dir / 'comparison_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        print(f"✓ 结果已保存到: {results_file}")

        # 保存性能基准
        benchmark_results = self.benchmark_performance()
        if benchmark_results:
            benchmark_file = output_dir / 'performance_benchmark.json'
            with open(benchmark_file, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
            print(f"✓ 基准测试已保存到: {benchmark_file}")

    def visualize_comparisons(self, output_dir: Path = None):
        """可视化比较结果"""
        if output_dir is None:
            output_dir = Path('output/batch_results')
        output_dir.mkdir(parents=True, exist_ok=True)

        # 性能比较图
        self._plot_performance_comparison(output_dir)

        # 质量指标比较图
        self._plot_quality_comparison(output_dir)

    def _plot_performance_comparison(self, output_dir: Path):
        """绘制性能比较图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 准备数据
            models = []
            methods = []
            single_times = []
            batch_times = []

            for model_name, model_results in self.results.items():
                for method_name, result in model_results.items():
                    if result:
                        models.append(model_name)
                        methods.append(method_name)
                        single_times.append(result.get('single_time', 0))
                        batch_times.append(result.get('batch_time', 0))

            # 单次解释时间
            labels = [f"{m}\n{n}" for m, n in zip(models, methods)]
            x_pos = np.arange(len(labels))

            ax1.bar(x_pos, single_times, alpha=0.7, color='#2E86AB')
            ax1.set_title('单次解释时间比较', fontsize=14, fontweight='bold')
            ax1.set_ylabel('时间 (秒)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)

            # 批量解释时间
            ax2.bar(x_pos, batch_times, alpha=0.7, color='#A23B72')
            ax2.set_title('批量解释时间比较', fontsize=14, fontweight='bold')
            ax2.set_ylabel('时间 (秒)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = output_dir / 'performance_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 性能比较图已保存到: {save_path}")
            plt.show()

        except Exception as e:
            print(f"性能比较图生成失败: {e}")

    def _plot_quality_comparison(self, output_dir: Path):
        """绘制质量指标比较图"""
        try:
            # 准备数据
            quality_data = {}
            for model_name, model_results in self.results.items():
                for method_name, result in model_results.items():
                    if result:
                        metrics = result.get('metrics', {})
                        key = f"{model_name}.{method_name}"
                        quality_data[key] = {
                            'attribution_mean': metrics.get('single_attribution_mean', 0),
                            'attribution_max': metrics.get('single_attribution_max', 0),
                            'attribution_sparsity': metrics.get('single_attribution_sparsity', 0)
                        }

            if not quality_data:
                print("没有质量数据可以可视化")
                return

            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            metrics_names = ['attribution_mean', 'attribution_max', 'attribution_sparsity']
            titles = ['归因均值', '最大归因值', '归因稀疏度']

            for ax, metric, title in zip(axes, metrics_names, titles):
                values = [quality_data[key][metric] for key in quality_data.keys()]
                labels = list(quality_data.keys())

                bars = ax.bar(labels, values, alpha=0.7, color=['#2E86AB', '#A23B72', '#F18F01'][:len(values)])
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_ylabel('指标值')
                ax.tick_params(axis='x', rotation=45)

                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')

                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = output_dir / 'quality_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 质量比较图已保存到: {save_path}")
            plt.show()

        except Exception as e:
            print(f"质量比较图生成失败: {e}")


def create_demo_models():
    """创建演示模型"""
    print("🏗️  创建演示模型")
    print("-" * 40)

    # 简单CNN模型
    class SimpleCNN(torch.nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(2, 32, 7, stride=2, padding=3)
            self.conv2 = torch.nn.Conv1d(32, 64, 5, stride=2, padding=2)
            self.conv3 = torch.nn.Conv1d(64, 128, 3, stride=2, padding=1)
            self.fc = torch.nn.Linear(128, num_classes)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.mean(x, dim=2)
            return self.fc(x)

    # 深度CNN模型
    class DeepCNN(torch.nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            layers = []
            in_channels = 2
            out_channels = 32

            for i in range(6):  # 6层
                layers.append(torch.nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.BatchNorm1d(out_channels))
                in_channels = out_channels
                out_channels *= 2 if i < 3 else 1

            layers.append(torch.nn.AdaptiveAvgPool1d(1))
            layers.append(torch.nn.Flatten())
            layers.append(torch.nn.Linear(in_channels, num_classes))

            self.network = torch.nn.Sequential(*layers)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            return self.network(x)

    # 创建模型
    models = {
        'SimpleCNN': SimpleCNN(),
        'DeepCNN': DeepCNN(),
    }

    # 注册到批量解释器
    batch_explainer = DemoBatchExplainer()
    for name, model in models.items():
        model.eval()
        batch_explainer.register_model(name, model, 'cnn')

    return batch_explainer


def main():
    """主函数：运行完整的批量解释演示"""
    print("批量解释与比较演示")
    print("=" * 80)

    # 1. 创建模型和批量解释器
    batch_explainer = create_demo_models()

    # 2. 创建解释器
    methods = ['saliency', 'integrated_gradients', 'deeplift']
    batch_explainer.create_explainers(methods)

    # 3. 创建测试数据
    print(f"\n📊 创建测试数据")
    test_data = torch.randn(8, 1000, 2)  # 8个样本
    target_classes = [0, 1, 2, 1, 0, 2, 1, 0]  # 目标类别
    print(f"   测试数据形状: {test_data.shape}")
    print(f"   目标类别: {target_classes}")

    # 4. 批量解释
    batch_explainer.batch_explain(test_data, target_classes)

    # 5. 比较分析
    batch_explainer.compare_explanations()

    # 6. 性能基准测试
    batch_explainer.benchmark_performance()

    # 7. 保存结果
    batch_explainer.save_results()

    # 8. 可视化
    batch_explainer.visualize_comparisons()

    print("\n" + "=" * 80)
    print("演示完成！")
    print("\n关键发现:")
    print("1. ⏱️  不同解释方法的计算效率差异显著")
    print("2. 📊 解释质量指标提供了客观的评估标准")
    print("3. 🏃 批量处理可以显著提高解释生成效率")
    print("4. 📈 可视化比较有助于选择最佳解释方法")
    print("5. 💾 自动化保存功能支持后续分析")
    print("\n输出文件保存在 'output/batch_results/' 目录中")


if __name__ == "__main__":
    main()
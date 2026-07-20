"""
路径签名分析脚本

生成样本×专家的激活矩阵，可视化路径签名热力图，
为每个故障类别统计专家激活分布。

Author: MoE Expert System
Date: 2024-11-26
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RoutingAnalyzer:
    """路径签名分析器

    分析MoE模型的专家路由行为，生成多层次的可解释性分析
    """

    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: NNSPN-MoE模型实例
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # 分析结果存储
        self.routing_data = {
            'expert_activations': [],
            'path_signatures': [],
            'class_distributions': {},
            'frequency_responses': []
        }

    def analyze_batch(self,
                     x_batch: torch.Tensor,
                     y_batch: Optional[torch.Tensor] = None,
                     return_explanations: bool = True) -> Dict:
        """分析单个批次的路由行为

        Args:
            x_batch: 输入信号 [batch_size, signal_length]
            y_batch: 标签 [batch_size] (可选)
            return_explanations: 是否返回详细解释

        Returns:
            批次分析结果
        """
        with torch.no_grad():
            x_batch = x_batch.to(self.device)

            # 前向传播获取路由信息
            logits, metadata = self.model(x_batch, return_explanations=True)

            # 提取关键信息
            routing_weights = metadata['routing_weights'].cpu().numpy()  # [batch_size, num_experts]
            expert_outputs = metadata['expert_outputs'].cpu().numpy()   # [batch_size, num_experts, feature_dim]

            batch_size = routing_weights.shape[0]

            # 路径签名分析
            path_signatures = []
            for i in range(batch_size):
                signature = self._compute_path_signature(
                    routing_weights[i],
                    expert_outputs[i],
                    y_batch[i].item() if y_batch is not None else None
                )
                path_signatures.append(signature)

            # 专家激活统计
            expert_stats = self._compute_expert_statistics(routing_weights)

            # 频率响应分析
            freq_responses = self.model.compute_frequency_response_matrix(x_batch)
            freq_responses = freq_responses.cpu().numpy()

            # 类别分布分析（如果提供标签）
            class_distribution = None
            if y_batch is not None:
                class_distribution = self._compute_class_distribution(
                    routing_weights, y_batch.cpu().numpy()
                )

            return {
                'batch_size': batch_size,
                'routing_weights': routing_weights,
                'expert_outputs': expert_outputs,
                'path_signatures': path_signatures,
                'expert_statistics': expert_stats,
                'frequency_responses': freq_responses,
                'class_distribution': class_distribution,
                'logits': logits.cpu().numpy(),
                'explanations': metadata.get('explanations', {})
            }

    def _compute_path_signature(self,
                               routing_weights: np.ndarray,
                               expert_outputs: np.ndarray,
                               label: Optional[int] = None) -> Dict:
        """计算单个样本的路径签名

        Args:
            routing_weights: 专家权重 [num_experts]
            expert_outputs: 专家输出 [num_experts, feature_dim]
            label: 样本标签（可选）

        Returns:
            路径签名字典
        """
        # 基础路由信息
        dominant_expert = np.argmax(routing_weights)
        expert_confidence = routing_weights[dominant_expert]
        active_experts = np.where(routing_weights > 0.1)[0]

        # 计算路由熵
        weights_normalized = routing_weights / (np.sum(routing_weights) + 1e-8)
        routing_entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-8))

        # 专家输出多样性
        expert_similarities = []
        for i in range(len(routing_weights)):
            for j in range(i + 1, len(routing_weights)):
                similarity = np.corrcoef(expert_outputs[i], expert_outputs[j])[0, 1]
                expert_similarities.append(similarity)

        diversity_score = 1.0 - np.mean(expert_similarities) if expert_similarities else 0.0

        return {
            'dominant_expert': int(dominant_expert),
            'expert_weights': routing_weights.tolist(),
            'expert_confidence': float(expert_confidence),
            'active_experts': active_experts.tolist(),
            'routing_entropy': float(routing_entropy),
            'diversity_score': float(diversity_score),
            'label': int(label) if label is not None else None,
            'expert_similarities': expert_similarities
        }

    def _compute_expert_statistics(self, routing_weights: np.ndarray) -> Dict:
        """计算专家激活统计信息

        Args:
            routing_weights: 专家权重 [batch_size, num_experts]

        Returns:
            专家统计信息
        """
        # 基础统计
        mean_weights = np.mean(routing_weights, axis=0)
        std_weights = np.std(routing_weights, axis=0)
        max_weights = np.max(routing_weights, axis=0)

        # 激活频率（权重 > 0.1）
        activation_frequency = np.mean(routing_weights > 0.1, axis=0)

        # 负载均衡度
        load_balance = 1.0 - np.std(mean_weights) / (np.mean(mean_weights) + 1e-8)

        return {
            'mean_weights': mean_weights.tolist(),
            'std_weights': std_weights.tolist(),
            'max_weights': max_weights.tolist(),
            'activation_frequency': activation_frequency.tolist(),
            'load_balance': float(load_balance),
            'most_used_expert': int(np.argmax(mean_weights)),
            'least_used_expert': int(np.argmin(mean_weights))
        }

    def _compute_class_distribution(self,
                                   routing_weights: np.ndarray,
                                   labels: np.ndarray) -> Dict:
        """计算每个类别的专家激活分布

        Args:
            routing_weights: 专家权重 [batch_size, num_experts]
            labels: 标签 [batch_size]

        Returns:
            类别分布统计
        """
        unique_labels = np.unique(labels)
        class_stats = {}

        for label in unique_labels:
            mask = labels == label
            class_weights = routing_weights[mask]

            # 计算该类别的专家激活统计
            class_mean = np.mean(class_weights, axis=0)
            class_std = np.std(class_weights, axis=0)
            class_dominant_expert = np.argmax(class_mean)

            # 专家激活一致性（类内相似度）
            activation_patterns = class_weights / (np.sum(class_weights, axis=1, keepdims=True) + 1e-8)
            consistency = 1.0 - np.mean(np.std(activation_patterns, axis=0))

            class_stats[int(label)] = {
                'num_samples': int(np.sum(mask)),
                'mean_activation': class_mean.tolist(),
                'std_activation': class_std.tolist(),
                'dominant_expert': int(class_dominant_expert),
                'dominant_expert_confidence': float(class_mean[class_dominant_expert]),
                'activation_consistency': float(consistency)
            }

        return class_stats

    def visualize_routing_heatmap(self,
                                 routing_weights: np.ndarray,
                                 labels: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """可视化路由权重热力图

        Args:
            routing_weights: 专家权重 [batch_size, num_experts]
            labels: 标签 [batch_size] (可选)
            save_path: 保存路径
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # 按样本排序的热力图
        sorted_indices = np.argsort(np.argmax(routing_weights, axis=1))
        sorted_weights = routing_weights[sorted_indices]

        sns.heatmap(sorted_weights.T,
                   ax=axes[0],
                   cmap='YlOrRd',
                   cbar_kws={'label': '专家权重'})
        axes[0].set_title('专家激活热力图（按主导专家排序）')
        axes[0].set_xlabel('样本索引')
        axes[0].set_ylabel('专家编号')

        # 按类别分组的平均激活
        if labels is not None:
            unique_labels = np.unique(labels)
            class_activations = []
            class_labels = []

            for label in unique_labels:
                mask = labels == label
                class_activation = np.mean(routing_weights[mask], axis=0)
                class_activations.append(class_activation)
                class_labels.append(f'类别{label}')

            class_activations = np.array(class_activations)

            sns.heatmap(class_activations.T,
                       ax=axes[1],
                       cmap='viridis',
                       xticklabels=class_labels,
                       cbar_kws={'label': '平均激活权重'})
            axes[1].set_title('各类别专家激活分布')
            axes[1].set_xlabel('故障类别')
            axes[1].set_ylabel('专家编号')
        else:
            axes[1].text(0.5, 0.5, '需要提供标签以显示类别分布',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('类别激活分布（无标签数据）')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def visualize_path_signature_network(self,
                                       path_signatures: List[Dict],
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """可视化路径签名网络图

        Args:
            path_signatures: 路径签名列表
            save_path: 保存路径
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        fig = plt.figure(figsize=figsize)

        # 创建网格子图
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 主导专家分布饼图
        ax1 = fig.add_subplot(gs[0, 0])
        dominant_experts = [sig['dominant_expert'] for sig in path_signatures]
        expert_counts = np.bincount(dominant_experts)
        ax1.pie(expert_counts, labels=[f'专家{i}' for i in range(len(expert_counts))],
               autopct='%1.1f%%')
        ax1.set_title('主导专家分布')

        # 2. 专家置信度分布
        ax2 = fig.add_subplot(gs[0, 1])
        confidences = [sig['expert_confidence'] for sig in path_signatures]
        ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('专家置信度')
        ax2.set_ylabel('频次')
        ax2.set_title('专家置信度分布')

        # 3. 路由熵分布
        ax3 = fig.add_subplot(gs[0, 2])
        entropies = [sig['routing_entropy'] for sig in path_signatures]
        ax3.hist(entropies, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('路由熵')
        ax3.set_ylabel('频次')
        ax3.set_title('路由熵分布')

        # 4. 多样性评分分布
        ax4 = fig.add_subplot(gs[1, 0])
        diversities = [sig['diversity_score'] for sig in path_signatures]
        ax4.hist(diversities, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('多样性评分')
        ax4.set_ylabel('频次')
        ax4.set_title('专家多样性分布')

        # 5. 激活专家数量分布
        ax5 = fig.add_subplot(gs[1, 1])
        active_counts = [len(sig['active_experts']) for sig in path_signatures]
        ax5.hist(active_counts, bins=range(1, len(path_signatures[0]['expert_weights']) + 2),
                alpha=0.7, color='red', edgecolor='black')
        ax5.set_xlabel('激活专家数量')
        ax5.set_ylabel('频次')
        ax5.set_title('激活专家数量分布')

        # 6. 专家权重矩阵
        ax6 = fig.add_subplot(gs[1, 2])
        weights_matrix = np.array([sig['expert_weights'] for sig in path_signatures[:100]])
        im = ax6.imshow(weights_matrix.T, cmap='YlOrRd', aspect='auto')
        ax6.set_xlabel('样本索引')
        ax6.set_ylabel('专家编号')
        ax6.set_title('专家权重矩阵（前100样本）')
        plt.colorbar(im, ax=ax6, label='权重')

        # 7. 熵 vs 置信度散点图
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.scatter(entropies, confidences, alpha=0.6, s=30)
        ax7.set_xlabel('路由熵')
        ax7.set_ylabel('专家置信度')
        ax7.set_title('路由熵 vs 专家置信度')

        # 添加趋势线
        z = np.polyfit(entropies, confidences, 1)
        p = np.poly1d(z)
        ax7.plot(sorted(entropies), p(sorted(entropies)), "r--", alpha=0.8)

        # 8. 综合统计文本
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        stats_text = f"""
路径签名统计分析:
• 总样本数: {len(path_signatures)}
• 平均路由熵: {np.mean(entropies):.3f}
• 平均置信度: {np.mean(confidences):.3f}
• 平均多样性: {np.mean(diversities):.3f}
• 平均激活专家数: {np.mean(active_counts):.1f}

主导专家:
• 专家0: {expert_counts[0]} 次 ({expert_counts[0]/len(path_signatures)*100:.1f}%)
• 专家1: {expert_counts[1]} 次 ({expert_counts[1]/len(path_signatures)*100:.1f}%)
• 专家2: {expert_counts[2]} 次 ({expert_counts[2]/len(path_signatures)*100:.1f}%)
"""

        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('路径签名综合分析', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def save_analysis_results(self,
                             save_dir: str,
                             routing_data: Dict,
                             filename_prefix: str = 'routing_analysis'):
        """保存分析结果

        Args:
            save_dir: 保存目录
            routing_data: 路由分析数据
            filename_prefix: 文件名前缀
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        def _serialize(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: _serialize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize(v) for v in value]
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            return value

        # 保存JSON数据
        json_path = save_path / f"{filename_prefix}_data.json"
        json_data = {key: _serialize(value) for key, value in routing_data.items()}

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        summary_alias_path = save_path / 'analysis_summary.json'
        with open(summary_alias_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"分析结果已保存到: {json_path}")

        # 生成和保存可视化
        if 'routing_weights' in routing_data:
            fig1 = self.visualize_routing_heatmap(
                routing_data['routing_weights'],
                routing_data.get('labels')
            )
            fig1_path = save_path / f"{filename_prefix}_heatmap.png"
            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print(f"路由热力图已保存到: {fig1_path}")

        if 'path_signatures' in routing_data:
            fig2 = self.visualize_path_signature_network(
                routing_data['path_signatures']
            )
            fig2_path = save_path / f"{filename_prefix}_signatures.png"
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"路径签名网络图已保存到: {fig2_path}")


def main():
    """示例使用"""
    # 这里需要导入实际的模型和数据加载器
    # from code.moe_model import NNSPNMoE
    # from data.datasets import load_thu_018_dataset

    print("路径签名分析工具")
    print("=" * 50)

    # 示例：如何使用分析器
    print("""
使用示例:
1. 初始化分析器:
   analyzer = RoutingAnalyzer(model)

2. 分析批次数据:
   results = analyzer.analyze_batch(x_batch, y_batch)

3. 可视化结果:
   fig = analyzer.visualize_routing_heatmap(results['routing_weights'], labels)

4. 保存结果:
   analyzer.save_analysis_results('./analysis_results/', results)
    """)


if __name__ == "__main__":
    main()

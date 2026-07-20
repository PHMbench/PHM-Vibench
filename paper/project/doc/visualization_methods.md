# 特征对齐效果可视化方案

## 概述

本文档详细描述了1D-2D融合可解释性故障诊断中的特征对齐效果可视化方案，包括多种可视化技术和分析方法，用于展示跨模态特征对齐的质量和效果。

## 1. 可视化方法分类

### 1.1 特征空间可视化
- t-SNE降维可视化
- PCA主成分分析可视化
- UMAP流形学习可视化
- 特征分布对比图

### 1.2 注意力权重可视化
- 融合注意力热力图
- 模态权重分布图
- 时频注意力对应图
- 类别特定注意力模式

### 1.3 梯度可解释性可视化
- Grad-CAM梯度可视化
- Guided Grad-CAM
- 特征重要性热力图
- 决策路径可视化

### 1.4 对齐质量评估可视化
- 对齐一致性评分图
- 跨模态相似性矩阵
- 局部邻域保持图
- 全局流形对齐图

## 2. 技术实现

### 2.1 t-SNE可视化实现

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

class FeatureAlignmentVisualizer:
    """特征对齐可视化器"""

    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = 'tab10'

    def plot_tsne_alignment(self, feat_1d, feat_2d, fused_feat, labels,
                           save_path=None, interactive=False):
        """
        t-SNE多模态特征对齐可视化

        Args:
            feat_1d: 1D特征矩阵 (N, d1)
            feat_2d: 2D特征矩阵 (N, d2)
            fused_feat: 融合特征矩阵 (N, df)
            labels: 标签数组 (N,)
            save_path: 保存路径
            interactive: 是否生成交互式图表
        """
        # 确保所有特征维度一致
        min_dim = min(feat_1d.shape[1], feat_2d.shape[1], fused_feat.shape[1])
        feat_1d_reduced = feat_1d[:, :min_dim]
        feat_2d_reduced = feat_2d[:, :min_dim]
        fused_feat_reduced = fused_feat[:, :min_dim]

        # 应用t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)//4))

        feat_1d_tsne = tsne.fit_transform(feat_1d_reduced)
        feat_2d_tsne = tsne.fit_transform(feat_2d_reduced)
        fused_tsne = tsne.fit_transform(fused_feat_reduced)

        # 编码标签
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        if interactive:
            # 生成交互式图表
            fig = go.Figure()

            # 1D特征
            fig.add_trace(go.Scatter(
                x=feat_1d_tsne[:, 0], y=feat_1d_tsne[:, 1],
                mode='markers',
                marker=dict(color=labels_encoded, colorscale='Viridis'),
                name='1D Features',
                text=[f'Class: {label}' for label in labels]
            ))

            # 2D特征
            fig.add_trace(go.Scatter(
                x=feat_2d_tsne[:, 0], y=feat_2d_tsne[:, 1],
                mode='markers',
                marker=dict(color=labels_encoded, colorscale='Plasma'),
                name='2D Features',
                text=[f'Class: {label}' for label in labels]
            ))

            # 融合特征
            fig.add_trace(go.Scatter(
                x=fused_tsne[:, 0], y=fused_tsne[:, 1],
                mode='markers',
                marker=dict(color=labels_encoded, colorscale='Rainbow'),
                name='Fused Features',
                text=[f'Class: {label}' for label in labels]
            ))

            fig.update_layout(
                title='t-SNE Feature Alignment Visualization',
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2'
            )

            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            return fig

        else:
            # 生成静态图表
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            scatter1 = axes[0].scatter(feat_1d_tsne[:, 0], feat_1d_tsne[:, 1],
                                     c=labels_encoded, cmap=self.color_palette, alpha=0.7)
            axes[0].set_title('1D Features (t-SNE)')
            axes[0].set_xlabel('t-SNE 1')
            axes[0].set_ylabel('t-SNE 2')

            scatter2 = axes[1].scatter(feat_2d_tsne[:, 0], feat_2d_tsne[:, 1],
                                     c=labels_encoded, cmap=self.color_palette, alpha=0.7)
            axes[1].set_title('2D Features (t-SNE)')
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')

            scatter3 = axes[2].scatter(fused_tsne[:, 0], fused_tsne[:, 1],
                                     c=labels_encoded, cmap=self.color_palette, alpha=0.7)
            axes[2].set_title('Fused Features (t-SNE)')
            axes[2].set_xlabel('t-SNE 1')
            axes[2].set_ylabel('t-SNE 2')

            # 添加图例
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=scatter1.cmap(scatter1.norm(i)),
                                        markersize=10, label=f'Class {i}')
                             for i in np.unique(labels_encoded)]
            fig.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

            return fig

    def plot_alignment_trajectory(self, feat_1d, feat_2d, fused_feat, labels,
                                sample_idx=None, save_path=None):
        """
        特征对齐轨迹可视化

        展示单个样本从1D到2D再到融合特征的空间变化轨迹
        """
        if sample_idx is None:
            sample_idx = np.random.choice(len(labels), 5, replace=False)

        # 对所有特征应用PCA
        from sklearn.decomposition import PCA

        # 合并所有特征以拟合PCA
        all_features = np.vstack([feat_1d, feat_2d, fused_feat])
        pca = PCA(n_components=2)
        pca.fit(all_features)

        # 转换特征
        feat_1d_pca = pca.transform(feat_1d)
        feat_2d_pca = pca.transform(feat_2d)
        fused_pca = pca.transform(fused_feat)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for idx, sample in enumerate(sample_idx):
            row = idx // 3
            col = idx % 3

            if row >= 2:  # 最多显示6个样本
                break

            # 获取样本在各空间的坐标
            point_1d = feat_1d_pca[sample]
            point_2d = feat_2d_pca[sample]
            point_fused = fused_pca[sample]

            # 绘制轨迹
            axes[row, col].plot([point_1d[0], point_2d[0], point_fused[0]],
                             [point_1d[1], point_2d[1], point_fused[1]],
                             'o-', linewidth=2, markersize=8)

            # 标注点
            axes[row, col].annotate('1D', point_1d, xytext=(5, 5),
                                   textcoords='offset points')
            axes[row, col].annotate('2D', point_2d, xytext=(5, 5),
                                   textcoords='offset points')
            axes[row, col].annotate('Fused', point_fused, xytext=(5, 5),
                                   textcoords='offset points')

            axes[row, col].set_title(f'Sample {sample} (Class: {labels[sample]})')
            axes[row, col].set_xlabel('PCA Component 1')
            axes[row, col].set_ylabel('PCA Component 2')
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig
```

### 2.2 Grad-CAM可视化实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image

class FusionGradCAM:
    """融合模型的Grad-CAM可视化"""

    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # 找到目标层
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_1d, input_2d, class_idx=None):
        """生成类激活映射"""
        # 前向传播
        self.model.eval()
        output = self.model(input_1d, input_2d)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # 计算梯度权重
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # 全局平均池化
        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        # 加权求和
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU激活
        cam = F.relu(cam)

        # 归一化
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def visualize_fusion_cam(self, input_1d, input_2d, labels,
                           save_path=None, top_k=3):
        """可视化融合注意力CAM"""
        self.model.eval()

        # 获取预测结果
        with torch.no_grad():
            outputs = self.model(input_1d, input_2d)
            probs = F.softmax(outputs, dim=1)

        fig, axes = plt.subplots(2, top_k + 1, figsize=(15, 8))

        # 显示原始输入
        # 1D信号可视化
        axes[0, 0].plot(input_1d[0].cpu().numpy())
        axes[0, 0].set_title('1D Input Signal')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Amplitude')

        # 2D频谱图可视化
        axes[1, 0].imshow(input_2d[0, 0].cpu().numpy(), cmap='viridis', aspect='auto')
        axes[1, 0].set_title('2D Spectrogram')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')

        # 获取top-k预测类别
        top_classes = torch.topk(probs[0], k=top_k)

        for i, (prob, class_idx) in enumerate(zip(top_classes.values, top_classes.indices)):
            # 生成CAM
            cam = self.generate_cam(input_1d, input_2d, class_idx.item())

            # 显示1D注意力
            if len(cam.shape) == 1:  # 1D注意力
                axes[0, i + 1].plot(input_1d[0].cpu().numpy(), alpha=0.7)
                axes[0, i + 1].plot(cam * max(input_1d[0].cpu().numpy()), 'r-', linewidth=2)
                axes[0, i + 1].set_title(f'1D CAM - Class {class_idx.item()} ({prob.item():.3f})')
            else:  # 2D注意力
                axes[0, i + 1].imshow(cam, cmap='jet', aspect='auto')
                axes[0, i + 1].set_title(f'1D CAM - Class {class_idx.item()} ({prob.item():.3f})')

            # 显示2D注意力
            if len(cam.shape) == 2:
                # 叠加显示
                spectrogram = input_2d[0, 0].cpu().numpy()
                cam_resized = cv2.resize(cam, (spectrogram.shape[1], spectrogram.shape[0]))

                axes[1, i + 1].imshow(spectrogram, cmap='viridis', alpha=0.7, aspect='auto')
                im = axes[1, i + 1].imshow(cam_resized, cmap='jet', alpha=0.5, aspect='auto')
                axes[1, i + 1].set_title(f'2D CAM - Class {class_idx.item()} ({prob.item():.3f})')
            else:
                axes[1, i + 1].text(0.5, 0.5, 'No 2D CAM available',
                                   ha='center', va='center', transform=axes[1, i + 1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig
```

### 2.3 注意力权重可视化

```python
class AttentionVisualizer:
    """注意力权重可视化器"""

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

    def plot_fusion_attention(self, attention_weights, labels,
                            save_path=None, top_samples=None):
        """
        绘制融合注意力权重热力图

        Args:
            attention_weights: 注意力权重矩阵 (N, num_layers, num_heads, seq_len, seq_len)
            labels: 标签数组 (N,)
            save_path: 保存路径
            top_samples: 显示的样本数量
        """
        if top_samples is None:
            top_samples = min(20, len(labels))

        # 平均注意力权重
        avg_attention = np.mean(attention_weights, axis=(1, 2))  # (N, seq_len, seq_len)

        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()

        for i in range(min(top_samples, len(avg_attention))):
            ax = axes[i]

            # 绘制注意力热力图
            im = ax.imshow(avg_attention[i], cmap='Blues', aspect='auto')

            # 设置标题
            ax.set_title(f'Sample {i} (Class: {labels[i]})')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 隐藏多余的子图
        for i in range(top_samples, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_modal_contribution(self, modal_weights, labels,
                               save_path=None):
        """
        绘制模态贡献度分析
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 模态权重分布
        axes[0, 0].hist(modal_weights[:, 0], bins=30, alpha=0.7, label='1D Modality')
        axes[0, 0].hist(modal_weights[:, 1], bins=30, alpha=0.7, label='2D Modality')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Modal Weight Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 类别特定的模态权重
        unique_labels = np.unique(labels)
        weight_by_class = {}
        for label in unique_labels:
            mask = labels == label
            weight_by_class[label] = {
                '1d_mean': np.mean(modal_weights[mask, 0]),
                '1d_std': np.std(modal_weights[mask, 0]),
                '2d_mean': np.mean(modal_weights[mask, 1]),
                '2d_std': np.std(modal_weights[mask, 1])
            }

        x = np.arange(len(unique_labels))
        width = 0.35

        axes[0, 1].bar(x - width/2, [weight_by_class[l]['1d_mean'] for l in unique_labels],
                      width, yerr=[weight_by_class[l]['1d_std'] for l in unique_labels],
                      label='1D Modality', alpha=0.8)
        axes[0, 1].bar(x + width/2, [weight_by_class[l]['2d_mean'] for l in unique_labels],
                      width, yerr=[weight_by_class[l]['2d_std'] for l in unique_labels],
                      label='2D Modality', alpha=0.8)
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Mean Weight')
        axes[0, 1].set_title('Modal Weight by Class')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(unique_labels)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 模态权重相关性
        axes[1, 0].scatter(modal_weights[:, 0], modal_weights[:, 1], alpha=0.6)
        axes[1, 0].set_xlabel('1D Modality Weight')
        axes[1, 0].set_ylabel('2D Modality Weight')
        axes[1, 0].set_title('Modal Weight Correlation')

        # 计算相关系数
        correlation = np.corrcoef(modal_weights[:, 0], modal_weights[:, 1])[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 模态主导性分析
        dominance_1d = np.sum(modal_weights[:, 0] > modal_weights[:, 1])
        dominance_2d = len(modal_weights) - dominance_1d

        axes[1, 1].pie([dominance_1d, dominance_2d],
                      labels=['1D Dominant', '2D Dominant'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Modal Dominance Analysis')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
```

### 2.4 对齐质量评估可视化

```python
class AlignmentQualityVisualizer:
    """对齐质量评估可视化器"""

    def __init__(self):
        pass

    def plot_alignment_metrics(self, alignment_scores, save_path=None):
        """
        绘制对齐质量指标
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = list(alignment_scores.keys())
        values = list(alignment_scores.values())

        # 1. 雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        values_radar = values + [values[0]]  # 闭合图形

        ax_radar = plt.subplot(2, 3, 1, projection='polar')
        ax_radar.plot(angles, values_radar, 'o-', linewidth=2)
        ax_radar.fill(angles, values_radar, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Alignment Quality Radar')

        # 2. 条形图
        ax_bar = axes[0, 1]
        bars = ax_bar.bar(metrics, values, alpha=0.8)
        ax_bar.set_title('Alignment Metrics')
        ax_bar.set_ylabel('Score')
        ax_bar.set_ylim(0, 1)
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')

        # 在柱状图上添加数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')

        # 3. 目标对比图
        target_values = [0.9, 0.8, 0.85] + [0.8] * (len(metrics) - 3)
        target_values = target_values[:len(metrics)]

        x = np.arange(len(metrics))
        width = 0.35

        ax_target = axes[0, 2]
        ax_target.bar(x - width/2, values, width, label='Actual', alpha=0.8)
        ax_target.bar(x + width/2, target_values, width, label='Target', alpha=0.8)
        ax_target.set_title('Actual vs Target Scores')
        ax_target.set_ylabel('Score')
        ax_target.set_xticks(x)
        ax_target.set_xticklabels(metrics)
        ax_target.legend()
        plt.setp(ax_target.get_xticklabels(), rotation=45, ha='right')

        # 4. 改进建议
        improvements = []
        for i, (metric, value) in enumerate(zip(metrics, values)):
            target = target_values[i]
            improvement = max(0, target - value)
            improvements.append(improvement)

        ax_improve = axes[1, 0]
        colors = ['red' if imp > 0.1 else 'orange' if imp > 0.05 else 'green' for imp in improvements]
        bars = ax_improve.bar(metrics, improvements, color=colors, alpha=0.8)
        ax_improve.set_title('Improvement Needed')
        ax_improve.set_ylabel('Gap to Target')
        plt.setp(ax_improve.get_xticklabels(), rotation=45, ha='right')

        # 5. 时间序列对比（如果有多轮训练的数据）
        ax_timeline = axes[1, 1]
        ax_timeline.text(0.5, 0.5, 'Training Timeline\n(Requires multi-epoch data)',
                       ha='center', va='center', transform=ax_timeline.transAxes)
        ax_timeline.set_title('Alignment Evolution')

        # 6. 综合评分
        overall_score = np.mean(values)
        ax_score = axes[1, 2]
        ax_score.bar(['Overall Alignment'], [overall_score], color='blue', alpha=0.8)
        ax_score.set_ylim(0, 1)
        ax_score.set_ylabel('Score')
        ax_score.set_title(f'Overall Score: {overall_score:.3f}')

        # 添加评分等级
        if overall_score >= 0.9:
            grade = 'Excellent'
            color = 'green'
        elif overall_score >= 0.8:
            grade = 'Good'
            color = 'blue'
        elif overall_score >= 0.7:
            grade = 'Fair'
            color = 'orange'
        else:
            grade = 'Poor'
            color = 'red'

        ax_score.text(0.5, overall_score + 0.05, grade, ha='center', va='bottom',
                     color=color, fontweight='bold', transform=ax_score.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cross_modal_similarity(self, similarity_matrix, labels, save_path=None):
        """
        绘制跨模态相似性矩阵
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 相似性矩阵热力图
        unique_labels = np.unique(labels)

        # 按类别排序
        sorted_indices = np.argsort(labels)
        sorted_similarity = similarity_matrix[sorted_indices][:, sorted_indices]

        im1 = axes[0, 0].imshow(sorted_similarity, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Cross-Modal Similarity Matrix')
        axes[0, 0].set_xlabel('Sample Index (Sorted by Class)')
        axes[0, 0].set_ylabel('Sample Index (Sorted by Class)')
        plt.colorbar(im1, ax=axes[0, 0])

        # 添加类别分界线
        class_counts = [np.sum(labels == label) for label in unique_labels]
        cumsum_counts = np.cumsum([0] + class_counts[:-1])
        for count in cumsum_counts[1:]:
            axes[0, 0].axhline(y=count, color='white', linestyle='--', linewidth=1)
            axes[0, 0].axvline(x=count, color='white', linestyle='--', linewidth=1)

        # 2. 类内相似性分析
        class_similarities = []
        for label in unique_labels:
            mask = labels == label
            class_similarity_matrix = similarity_matrix[mask][:, mask]
            # 排除对角线元素
            mask_triu = np.triu(np.ones_like(class_similarity_matrix, dtype=bool), k=1)
            class_similarities.append(class_similarity_matrix[mask_triu].mean())

        axes[0, 1].bar(unique_labels, class_similarities, alpha=0.8)
        axes[0, 1].set_title('Within-Class Similarity')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Average Similarity')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 类间相似性分析
        between_class_similarities = []
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j:
                    mask1 = labels == label1
                    mask2 = labels == label2
                    between_sim = similarity_matrix[mask1][:, mask2].mean()
                    between_class_similarities.append(between_sim)

        if between_class_similarities:
            axes[1, 0].hist(between_class_similarities, bins=20, alpha=0.8, edgecolor='black')
            axes[1, 0].set_title('Between-Class Similarity Distribution')
            axes[1, 0].set_xlabel('Similarity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 相似性统计
        all_similarities = similarity_matrix.flatten()
        similarity_stats = {
            'Mean': np.mean(all_similarities),
            'Std': np.std(all_similarities),
            'Min': np.min(all_similarities),
            'Max': np.max(all_similarities),
            'Median': np.median(all_similarities)
        }

        ax_stats = axes[1, 1]
        y_pos = np.arange(len(similarity_stats))
        ax_stats.barh(y_pos, list(similarity_stats.values()), alpha=0.8)
        ax_stats.set_yticks(y_pos)
        ax_stats.set_yticklabels(list(similarity_stats.keys()))
        ax_stats.set_xlabel('Value')
        ax_stats.set_title('Similarity Statistics')
        ax_stats.grid(True, alpha=0.3, axis='x')

        # 在柱状图上添加数值
        for i, (key, value) in enumerate(similarity_stats.items()):
            ax_stats.text(value + 0.01, i, f'{value:.3f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
```

## 3. 完整可视化流程

```python
class AlignmentVisualizationPipeline:
    """特征对齐可视化完整流程"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.visualizer = FeatureAlignmentVisualizer()
        self.attention_viz = AttentionVisualizer()
        self.alignment_viz = AlignmentQualityVisualizer()

    def run_complete_visualization(self, test_loader, save_dir):
        """
        运行完整的可视化流程
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 收集特征和预测
        all_features_1d = []
        all_features_2d = []
        all_fused_features = []
        all_labels = []
        all_attention_weights = []
        all_modal_weights = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collecting features"):
                signal_1d, signal_2d, labels = batch
                signal_1d = signal_1d.to(self.device)
                signal_2d = signal_2d.to(self.device)

                # 获取模型输出
                outputs, features, attention_info = self.model(
                    signal_1d, signal_2d,
                    return_features=True,
                    return_attention_info=True
                )

                # 收集特征
                if isinstance(features, tuple):
                    feat_1d, feat_2d, fused_feat = features
                else:
                    # 如果只返回融合特征，需要从模型中提取1D和2D特征
                    feat_1d, feat_2d = self._extract_modality_features(signal_1d, signal_2d)
                    fused_feat = features

                all_features_1d.append(feat_1d.cpu().numpy())
                all_features_2d.append(feat_2d.cpu().numpy())
                all_fused_features.append(fused_feat.cpu().numpy())
                all_labels.append(labels.numpy())

                # 收集注意力信息
                if 'attention_weights' in attention_info:
                    all_attention_weights.append(attention_info['attention_weights'].cpu().numpy())
                if 'modal_weights' in attention_info:
                    all_modal_weights.append(attention_info['modal_weights'].cpu().numpy())

        # 合并所有特征
        feat_1d = np.vstack(all_features_1d)
        feat_2d = np.vstack(all_features_2d)
        fused_feat = np.vstack(all_fused_features)
        labels = np.concatenate(all_labels)

        # 1. t-SNE对齐可视化
        print("生成t-SNE可视化...")
        self.visualizer.plot_tsne_alignment(
            feat_1d, feat_2d, fused_feat, labels,
            save_path=save_dir / 'tsne_alignment.png'
        )

        # 2. 对齐轨迹可视化
        print("生成对齐轨迹可视化...")
        self.visualizer.plot_alignment_trajectory(
            feat_1d, feat_2d, fused_feat, labels,
            save_path=save_dir / 'alignment_trajectory.png'
        )

        # 3. 注意力权重可视化
        if all_attention_weights:
            print("生成注意力权重可视化...")
            attention_weights = np.vstack(all_attention_weights)
            self.attention_viz.plot_fusion_attention(
                attention_weights, labels,
                save_path=save_dir / 'attention_weights.png'
            )

        # 4. 模态贡献可视化
        if all_modal_weights:
            print("生成模态贡献可视化...")
            modal_weights = np.vstack(all_modal_weights)
            self.attention_viz.plot_modal_contribution(
                modal_weights, labels,
                save_path=save_dir / 'modal_contribution.png'
            )

        # 5. 对齐质量评估
        print("计算对齐质量指标...")
        alignment_scores = self._calculate_alignment_scores(
            feat_1d, feat_2d, fused_feat, labels
        )

        self.alignment_viz.plot_alignment_metrics(
            alignment_scores,
            save_path=save_dir / 'alignment_quality.png'
        )

        # 6. 跨模态相似性分析
        print("分析跨模态相似性...")
        similarity_matrix = self._calculate_cross_modal_similarity(feat_1d, feat_2d)

        self.alignment_viz.plot_cross_modal_similarity(
            similarity_matrix, labels,
            save_path=save_dir / 'cross_modal_similarity.png'
        )

        # 保存对齐质量报告
        self._save_alignment_report(alignment_scores, save_dir)

        print(f"所有可视化结果已保存到: {save_dir}")

    def _extract_modality_features(self, signal_1d, signal_2d):
        """提取模态特征（需要根据具体模型实现）"""
        # 这里需要根据具体的模型架构来实现
        # 通常需要访问模型的中间层输出
        pass

    def _calculate_alignment_scores(self, feat_1d, feat_2d, fused_feat, labels):
        """计算对齐质量评分"""
        from sklearn.metrics import adjusted_rand_score, silhouette_score

        scores = {}

        # 1. 物理对齐评分
        scores['physical_alignment'] = self._calculate_physical_alignment(feat_1d, feat_2d)

        # 2. 语义对齐评分
        scores['semantic_alignment'] = self._calculate_semantic_alignment(feat_1d, feat_2d, labels)

        # 3. 几何对齐评分
        scores['geometric_alignment'] = self._calculate_geometric_alignment(feat_1d, feat_2d)

        # 4. 融合质量评分
        scores['fusion_quality'] = self._calculate_fusion_quality(feat_1d, feat_2d, fused_feat)

        return scores

    def _calculate_cross_modal_similarity(self, feat_1d, feat_2d):
        """计算跨模态相似性矩阵"""
        from sklearn.metrics.pairwise import cosine_similarity

        # 确保特征维度一致
        min_dim = min(feat_1d.shape[1], feat_2d.shape[1])
        feat_1d_reduced = feat_1d[:, :min_dim]
        feat_2d_reduced = feat_2d[:, :min_dim]

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(feat_1d_reduced, feat_2d_reduced)

        return similarity_matrix

    def _save_alignment_report(self, scores, save_dir):
        """保存对齐质量报告"""
        report = f"""
# 特征对齐质量报告

## 对齐指标评分
- 物理对齐: {scores['physical_alignment']:.4f}
- 语义对齐: {scores['semantic_alignment']:.4f}
- 几何对齐: {scores['geometric_alignment']:.4f}
- 融合质量: {scores['fusion_quality']:.4f}

## 综合评分
{np.mean(list(scores.values())):.4f}

## 评估说明
- 物理对齐: 衡量时频域特征的对应关系
- 语义对齐: 衡量跨模态特征的语义一致性
- 几何对齐: 衡量特征空间的几何结构保持
- 融合质量: 衡量融合特征的表示能力

## 改进建议
根据各项指标的评分，针对性地改进模型的特征对齐能力。
"""

        with open(save_dir / 'alignment_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
```

## 4. 使用示例

```python
# 使用示例
if __name__ == "__main__":
    # 加载训练好的模型
    model = ProgressiveFusionNetwork(...)
    model.load_state_dict(torch.load('best_model.pt'))

    # 创建可视化流程
    pipeline = AlignmentVisualizationPipeline(model)

    # 运行完整可视化
    pipeline.run_complete_visualization(test_loader, 'visualization_results')
```

---

*更新时间: 2025-11-26*
*版本: v1.0*
*状态: 完整可视化方案设计完成*
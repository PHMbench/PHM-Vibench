#!/usr/bin/env python3
"""
ContrastiveIDTask可视化分析工具
提供论文所需的各种可视化功能，包括特征空间分析、注意力可视化、性能对比等
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import argparse
import logging
from datetime import datetime
import warnings

# 科学计算和机器学习库
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 可选的高级可视化库
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP不可用，将使用t-SNE替代")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly不可用，将使用matplotlib替代交互式图表")

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class VisualizationTools:
    """ContrastiveIDTask可视化工具类"""
    
    def __init__(self, save_dir="./visualization_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / "visualization.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 颜色主题
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C',
            'accent': '#F39C12',
            'success': '#27AE60',
            'warning': '#F39C12',
            'info': '#8E44AD'
        }
        
        self.logger.info(f"可视化工具初始化完成，结果保存到: {self.save_dir}")
    
    def create_mock_features_and_labels(self, n_samples=1000, n_features=128, n_classes=10):
        """创建模拟特征和标签用于可视化测试"""
        np.random.seed(42)
        
        features = []
        labels = []
        
        for class_id in range(n_classes):
            # 为每个类别创建具有聚类特性的特征
            class_center = np.random.randn(n_features) * 2
            class_samples = n_samples // n_classes
            
            for i in range(class_samples):
                # 在类别中心周围生成样本
                sample = class_center + np.random.randn(n_features) * 0.5
                features.append(sample)
                labels.append(class_id)
        
        return np.array(features), np.array(labels)
    
    def plot_feature_space_2d(self, features: np.ndarray, labels: np.ndarray, 
                             method='tsne', title='Feature Space Visualization',
                             save_name='feature_space_2d.png') -> str:
        """
        2D特征空间可视化
        
        Args:
            features: 特征矩阵 [N, D]
            labels: 标签向量 [N]
            method: 降维方法 ('tsne', 'pca', 'umap')
            title: 图表标题
            save_name: 保存文件名
        """
        self.logger.info(f"生成2D特征空间可视化 - 方法: {method}")
        
        # 数据预处理
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 降维
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features_scaled)
            explained_var = reducer.explained_variance_ratio_
            method_info = f"PCA (解释方差: {explained_var[0]:.3f}, {explained_var[1]:.3f})"
            
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            features_2d = reducer.fit_transform(features_scaled)
            method_info = "t-SNE (perplexity=30)"
            
        elif method.lower() == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            features_2d = reducer.fit_transform(features_scaled)
            method_info = "UMAP (neighbors=15, min_dist=0.1)"
            
        else:
            self.logger.warning(f"方法 {method} 不可用，使用PCA替代")
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features_scaled)
            explained_var = reducer.explained_variance_ratio_
            method_info = f"PCA (解释方差: {explained_var[0]:.3f}, {explained_var[1]:.3f})"
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', 
                       alpha=0.7, s=50)
        
        plt.title(f'{title}\\n{method_info}', fontsize=16)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.02, 0.02, f'样本数: {len(features)}, 特征维度: {features.shape[1]}, 类别数: {len(unique_labels)}',
                   fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"2D特征空间可视化已保存: {save_path}")
        return str(save_path)
    
    def plot_feature_space_3d(self, features: np.ndarray, labels: np.ndarray,
                             method='pca', title='3D Feature Space',
                             save_name='feature_space_3d.png') -> str:
        """3D特征空间可视化"""
        self.logger.info(f"生成3D特征空间可视化 - 方法: {method}")
        
        # 数据预处理
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 降维到3D
        if method.lower() == 'pca':
            reducer = PCA(n_components=3, random_state=42)
            features_3d = reducer.fit_transform(features_scaled)
            explained_var = reducer.explained_variance_ratio_
            method_info = f"PCA (解释方差: {explained_var[0]:.3f}, {explained_var[1]:.3f}, {explained_var[2]:.3f})"
        elif method.lower() == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=3, random_state=42)
            features_3d = reducer.fit_transform(features_scaled)
            method_info = "UMAP 3D"
        else:
            reducer = PCA(n_components=3, random_state=42)
            features_3d = reducer.fit_transform(features_scaled)
            explained_var = reducer.explained_variance_ratio_
            method_info = f"PCA (解释方差: {explained_var[0]:.3f}, {explained_var[1]:.3f}, {explained_var[2]:.3f})"
        
        # 3D可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                      c=[colors[i]], label=f'Class {label}', alpha=0.6, s=30)
        
        ax.set_title(f'{title}\\n{method_info}', fontsize=16)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"3D特征空间可视化已保存: {save_path}")
        return str(save_path)
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, 
                              input_labels: List[str] = None,
                              title='Attention Weights Heatmap',
                              save_name='attention_heatmap.png') -> str:
        """
        注意力权重热图
        
        Args:
            attention_weights: 注意力权重矩阵 [seq_len, seq_len] 或 [heads, seq_len, seq_len]
            input_labels: 输入序列标签
            title: 图表标题
            save_name: 保存文件名
        """
        self.logger.info("生成注意力权重热图")
        
        # 处理多头注意力
        if attention_weights.ndim == 3:
            num_heads = attention_weights.shape[0]
            fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(16, 8))
            axes = axes.flatten() if num_heads > 1 else [axes]
            
            for head in range(num_heads):
                ax = axes[head]
                
                # 绘制热图
                im = ax.imshow(attention_weights[head], cmap='Blues', aspect='auto')
                ax.set_title(f'Head {head + 1}')
                
                # 设置坐标轴标签
                if input_labels:
                    ax.set_xticks(range(len(input_labels)))
                    ax.set_yticks(range(len(input_labels)))
                    ax.set_xticklabels(input_labels, rotation=45)
                    ax.set_yticklabels(input_labels)
                
                # 添加colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 隐藏多余的子图
            for i in range(num_heads, len(axes)):
                axes[i].set_visible(False)
                
        else:
            # 单个注意力矩阵
            plt.figure(figsize=(10, 8))
            
            # 绘制热图
            im = plt.imshow(attention_weights, cmap='Blues', aspect='auto')
            
            # 设置标签和标题
            if input_labels:
                plt.xticks(range(len(input_labels)), input_labels, rotation=45)
                plt.yticks(range(len(input_labels)), input_labels)
            
            plt.title(title, fontsize=16)
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # 添加数值标注（如果矩阵不太大）
            if attention_weights.shape[0] <= 20:
                for i in range(attention_weights.shape[0]):
                    for j in range(attention_weights.shape[1]):
                        plt.text(j, i, f'{attention_weights[i, j]:.3f}',
                               ha='center', va='center', 
                               color='red' if attention_weights[i, j] > 0.5 else 'black',
                               fontsize=8)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"注意力热图已保存: {save_path}")
        return str(save_path)
    
    def plot_training_curves(self, training_history: Dict[str, List[float]], 
                           title='Training Curves',
                           save_name='training_curves.png') -> str:
        """
        训练曲线可视化
        
        Args:
            training_history: 训练历史数据，例如：
                {
                    'train_loss': [0.5, 0.4, 0.3, ...],
                    'val_loss': [0.6, 0.5, 0.4, ...],
                    'train_acc': [0.7, 0.8, 0.85, ...],
                    'val_acc': [0.65, 0.75, 0.8, ...]
                }
        """
        self.logger.info("生成训练曲线可视化")
        
        # 确定子图布局
        metrics = list(training_history.keys())
        loss_metrics = [m for m in metrics if 'loss' in m.lower()]
        acc_metrics = [m for m in metrics if 'acc' in m.lower()]
        other_metrics = [m for m in metrics if m not in loss_metrics and m not in acc_metrics]
        
        n_plots = len(loss_metrics) > 0 + len(acc_metrics) > 0 + len(other_metrics) > 0
        
        if n_plots == 0:
            self.logger.warning("没有可绘制的训练曲线数据")
            return ""
        
        fig, axes = plt.subplots(1, min(n_plots, 3), figsize=(15, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 绘制损失曲线
        if loss_metrics:
            ax = axes[plot_idx]
            for metric in loss_metrics:
                epochs = range(1, len(training_history[metric]) + 1)
                ax.plot(epochs, training_history[metric], label=metric, linewidth=2)
            
            ax.set_title('Loss Curves', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 绘制准确率曲线
        if acc_metrics and plot_idx < len(axes):
            ax = axes[plot_idx]
            for metric in acc_metrics:
                epochs = range(1, len(training_history[metric]) + 1)
                ax.plot(epochs, training_history[metric], label=metric, linewidth=2)
            
            ax.set_title('Accuracy Curves', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 绘制其他指标
        if other_metrics and plot_idx < len(axes):
            ax = axes[plot_idx]
            for metric in other_metrics:
                epochs = range(1, len(training_history[metric]) + 1)
                ax.plot(epochs, training_history[metric], label=metric, linewidth=2)
            
            ax.set_title('Other Metrics', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线已保存: {save_path}")
        return str(save_path)
    
    def plot_cross_dataset_performance(self, performance_data: Dict[str, Dict[str, float]],
                                     title='Cross-Dataset Performance Comparison',
                                     save_name='cross_dataset_performance.png') -> str:
        """
        跨数据集性能对比
        
        Args:
            performance_data: 性能数据，格式如：
                {
                    'CWRU': {'accuracy': 0.85, 'f1': 0.83, 'precision': 0.84, 'recall': 0.85},
                    'XJTU': {'accuracy': 0.78, 'f1': 0.76, 'precision': 0.77, 'recall': 0.79},
                    ...
                }
        """
        self.logger.info("生成跨数据集性能对比图")
        
        # 转换为DataFrame
        df = pd.DataFrame(performance_data).T
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        metrics = list(df.columns)
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, metric in enumerate(metrics[:4]):  # 最多显示4个指标
            ax = axes[i // 2, i % 2]
            
            # 条形图
            bars = ax.bar(df.index, df[metric], color=colors[i % len(colors)], alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric.title()} Across Datasets', fontsize=14)
            ax.set_ylabel(metric.title())
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成详细的对比表格
        self.save_performance_table(df, save_name.replace('.png', '_table.csv'))
        
        self.logger.info(f"跨数据集性能对比图已保存: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str] = None,
                             title='Confusion Matrix',
                             save_name='confusion_matrix.png') -> str:
        """绘制混淆矩阵"""
        self.logger.info("生成混淆矩阵")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始计数混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 归一化混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax2)
        ax2.set_title('Normalized Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        # 设置类别标签
        if class_names:
            ax1.set_xticklabels(class_names)
            ax1.set_yticklabels(class_names)
            ax2.set_xticklabels(class_names)
            ax2.set_yticklabels(class_names)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"混淆矩阵已保存: {save_path}")
        return str(save_path)
    
    def plot_memory_usage_analysis(self, memory_data: Dict[str, List[float]],
                                  title='Memory Usage Analysis',
                                  save_name='memory_usage.png') -> str:
        """
        内存使用分析图
        
        Args:
            memory_data: 内存使用数据，例如：
                {
                    'timestamps': [0, 1, 2, 3, ...],
                    'ram_usage_mb': [100, 150, 200, ...],
                    'gpu_usage_mb': [0, 50, 100, ...]
                }
        """
        self.logger.info("生成内存使用分析图")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        timestamps = memory_data.get('timestamps', range(len(list(memory_data.values())[0])))
        
        # RAM使用图
        if 'ram_usage_mb' in memory_data:
            ax1 = axes[0]
            ax1.plot(timestamps, memory_data['ram_usage_mb'], 
                    color=self.colors['primary'], linewidth=2, label='RAM Usage')
            ax1.fill_between(timestamps, memory_data['ram_usage_mb'], 
                           alpha=0.3, color=self.colors['primary'])
            ax1.set_title('RAM Memory Usage Over Time', fontsize=14)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('RAM Usage (MB)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # GPU使用图
        if 'gpu_usage_mb' in memory_data:
            ax2 = axes[1]
            ax2.plot(timestamps, memory_data['gpu_usage_mb'], 
                    color=self.colors['secondary'], linewidth=2, label='GPU Memory')
            ax2.fill_between(timestamps, memory_data['gpu_usage_mb'], 
                           alpha=0.3, color=self.colors['secondary'])
            ax2.set_title('GPU Memory Usage Over Time', fontsize=14)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"内存使用分析图已保存: {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self, data: Dict[str, Any], 
                                   save_name='interactive_dashboard.html') -> str:
        """创建交互式仪表板（如果Plotly可用）"""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly不可用，跳过交互式仪表板生成")
            return ""
        
        self.logger.info("生成交互式仪表板")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Space', 'Training Curves', 'Performance Metrics', 'Memory Usage'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 添加示例图表（实际使用时应该传入真实数据）
        # 这里只是演示结构
        
        # 特征空间散点图
        if 'features_2d' in data and 'labels' in data:
            features_2d = data['features_2d']
            labels = data['labels']
            
            for label in np.unique(labels):
                mask = labels == label
                fig.add_trace(
                    go.Scatter(x=features_2d[mask, 0], y=features_2d[mask, 1],
                              mode='markers', name=f'Class {label}'),
                    row=1, col=1
                )
        
        # 更新布局
        fig.update_layout(
            title="ContrastiveIDTask Interactive Dashboard",
            showlegend=True,
            height=800
        )
        
        # 保存HTML
        save_path = self.save_dir / save_name
        fig.write_html(str(save_path))
        
        self.logger.info(f"交互式仪表板已保存: {save_path}")
        return str(save_path)
    
    def save_performance_table(self, df: pd.DataFrame, filename: str):
        """保存性能对比表格"""
        save_path = self.save_dir / filename
        df.to_csv(save_path)
        self.logger.info(f"性能表格已保存: {save_path}")
    
    def generate_comprehensive_report(self, experiment_results: Dict[str, Any]) -> str:
        """生成综合可视化报告"""
        self.logger.info("生成综合可视化报告")
        
        report_paths = []
        
        try:
            # 1. 特征空间可视化
            if 'features' in experiment_results and 'labels' in experiment_results:
                features = experiment_results['features']
                labels = experiment_results['labels']
                
                # 2D可视化
                for method in ['pca', 'tsne']:
                    if method == 'umap' and not UMAP_AVAILABLE:
                        continue
                    path = self.plot_feature_space_2d(
                        features, labels, method=method,
                        title=f'Feature Space - {method.upper()}',
                        save_name=f'feature_space_{method}.png'
                    )
                    report_paths.append(path)
                
                # 3D可视化
                path = self.plot_feature_space_3d(
                    features, labels, method='pca',
                    save_name='feature_space_3d.png'
                )
                report_paths.append(path)
            
            # 2. 训练曲线
            if 'training_history' in experiment_results:
                path = self.plot_training_curves(
                    experiment_results['training_history'],
                    save_name='training_curves.png'
                )
                report_paths.append(path)
            
            # 3. 跨数据集性能
            if 'cross_dataset_performance' in experiment_results:
                path = self.plot_cross_dataset_performance(
                    experiment_results['cross_dataset_performance'],
                    save_name='cross_dataset_performance.png'
                )
                report_paths.append(path)
            
            # 4. 混淆矩阵
            if 'y_true' in experiment_results and 'y_pred' in experiment_results:
                path = self.plot_confusion_matrix(
                    experiment_results['y_true'],
                    experiment_results['y_pred'],
                    class_names=experiment_results.get('class_names'),
                    save_name='confusion_matrix.png'
                )
                report_paths.append(path)
            
            # 5. 注意力权重
            if 'attention_weights' in experiment_results:
                path = self.plot_attention_heatmap(
                    experiment_results['attention_weights'],
                    input_labels=experiment_results.get('input_labels'),
                    save_name='attention_heatmap.png'
                )
                report_paths.append(path)
            
            # 6. 内存使用
            if 'memory_data' in experiment_results:
                path = self.plot_memory_usage_analysis(
                    experiment_results['memory_data'],
                    save_name='memory_usage.png'
                )
                report_paths.append(path)
            
            # 7. 交互式仪表板
            if PLOTLY_AVAILABLE:
                path = self.create_interactive_dashboard(
                    experiment_results,
                    save_name='dashboard.html'
                )
                if path:
                    report_paths.append(path)
            
            # 生成索引文件
            self._generate_index_html(report_paths)
            
            self.logger.info(f"综合报告生成完成，共生成 {len(report_paths)} 个可视化文件")
            return str(self.save_dir / "index.html")
            
        except Exception as e:
            self.logger.error(f"生成综合报告时出错: {e}")
            return ""
    
    def _generate_index_html(self, report_paths: List[str]):
        """生成索引HTML文件"""
        index_path = self.save_dir / "index.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ContrastiveIDTask Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86C1; }}
                h2 {{ color: #34495E; }}
                .image-container {{ margin: 20px 0; }}
                .image-container img {{ max-width: 100%; height: auto; }}
                .timestamp {{ color: #7F8C8D; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>ContrastiveIDTask 可视化分析报告</h1>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>可视化图表</h2>
        """
        
        for path in report_paths:
            if path.endswith('.png'):
                filename = Path(path).name
                title = filename.replace('_', ' ').replace('.png', '').title()
                html_content += f"""
                <div class="image-container">
                    <h3>{title}</h3>
                    <img src="{filename}" alt="{title}">
                </div>
                """
            elif path.endswith('.html') and 'dashboard' in path:
                html_content += f"""
                <div class="image-container">
                    <h3>Interactive Dashboard</h3>
                    <p><a href="dashboard.html" target="_blank">查看交互式仪表板</a></p>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def create_demo_visualizations(self):
        """创建演示可视化（用于测试）"""
        self.logger.info("创建演示可视化")
        
        # 创建模拟数据
        features, labels = self.create_mock_features_and_labels()
        
        # 模拟训练历史
        epochs = 20
        training_history = {
            'train_loss': [0.8 - 0.03*i + np.random.normal(0, 0.02) for i in range(epochs)],
            'val_loss': [0.85 - 0.025*i + np.random.normal(0, 0.03) for i in range(epochs)],
            'train_acc': [0.3 + 0.03*i + np.random.normal(0, 0.01) for i in range(epochs)],
            'val_acc': [0.25 + 0.035*i + np.random.normal(0, 0.02) for i in range(epochs)]
        }
        
        # 模拟跨数据集性能
        cross_dataset_performance = {
            'CWRU': {'accuracy': 0.85, 'f1': 0.83, 'precision': 0.84, 'recall': 0.85},
            'XJTU': {'accuracy': 0.78, 'f1': 0.76, 'precision': 0.77, 'recall': 0.79},
            'FEMTO': {'accuracy': 0.72, 'f1': 0.70, 'precision': 0.71, 'recall': 0.73},
            'THU': {'accuracy': 0.80, 'f1': 0.78, 'precision': 0.79, 'recall': 0.81}
        }
        
        # 模拟注意力权重
        attention_weights = np.random.random((8, 16, 16))  # 8个头，16x16注意力矩阵
        attention_weights = F.softmax(torch.tensor(attention_weights), dim=-1).numpy()
        
        # 模拟内存数据
        timestamps = list(range(0, 100, 2))
        memory_data = {
            'timestamps': timestamps,
            'ram_usage_mb': [100 + 5*i + np.random.normal(0, 10) for i in range(len(timestamps))],
            'gpu_usage_mb': [50 + 3*i + np.random.normal(0, 5) for i in range(len(timestamps))]
        }
        
        # 生成混淆矩阵数据
        n_samples = 1000
        y_true = np.random.randint(0, 10, n_samples)
        y_pred = y_true.copy()
        # 添加一些错误分类
        error_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
        y_pred[error_indices] = np.random.randint(0, 10, len(error_indices))
        
        # 组织数据
        experiment_results = {
            'features': features,
            'labels': labels,
            'training_history': training_history,
            'cross_dataset_performance': cross_dataset_performance,
            'attention_weights': attention_weights,
            'memory_data': memory_data,
            'y_true': y_true,
            'y_pred': y_pred,
            'class_names': [f'Class_{i}' for i in range(10)]
        }
        
        # 生成综合报告
        index_path = self.generate_comprehensive_report(experiment_results)
        
        self.logger.info("演示可视化创建完成")
        self.logger.info(f"查看报告: {index_path}")
        
        return index_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask可视化分析工具")
    parser.add_argument("--save-dir", default="./visualization_results",
                       help="结果保存目录")
    parser.add_argument("--demo", action="store_true",
                       help="运行演示模式")
    parser.add_argument("--data-file", type=str,
                       help="实验结果数据文件路径(JSON格式)")
    
    args = parser.parse_args()
    
    # 初始化可视化工具
    viz_tools = VisualizationTools(save_dir=args.save_dir)
    
    if args.demo:
        # 演示模式
        viz_tools.create_demo_visualizations()
    
    elif args.data_file:
        # 从文件加载数据并生成可视化
        try:
            with open(args.data_file, 'r') as f:
                experiment_results = json.load(f)
            
            viz_tools.generate_comprehensive_report(experiment_results)
            
        except Exception as e:
            viz_tools.logger.error(f"处理数据文件时出错: {e}")
    
    else:
        print("请指定 --demo 或 --data-file 参数")
        print("使用 --help 查看详细帮助")


if __name__ == "__main__":
    main()
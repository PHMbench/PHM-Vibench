#!/usr/bin/env python3
"""
ContrastiveIDTask基线方法对比脚本
实现多种基线方法与对比学习方法的公平对比实验
包括随机初始化、AutoEncoder预训练、MaskedReconstruction、SimCLR和MoCo的适配版本
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
from abc import ABC, abstractmethod

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class BaseMethod(ABC):
    """基线方法抽象基类"""
    
    def __init__(self, name: str, network: nn.Module, config: Dict[str, Any]):
        self.name = name
        self.network = network
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    @abstractmethod
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """训练一个epoch"""
        pass
    
    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型性能"""
        pass
    
    def get_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征和标签"""
        self.network.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                features = self.network(batch_data)
                
                features_list.append(features.cpu().numpy())
                labels_list.append(batch_labels.numpy())
        
        return np.vstack(features_list), np.hstack(labels_list)


class RandomInitMethod(BaseMethod):
    """随机初始化基线"""
    
    def __init__(self, network: nn.Module, config: Dict[str, Any]):
        super().__init__("Random_Init", network, config)
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        # 随机初始化不需要训练，直接返回
        return {'loss': 0.0, 'accuracy': 0.0}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        # 评估随机特征的质量
        features, labels = self.get_features(dataloader)
        
        # 使用简单的k-means聚类评估特征质量
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        ari_score = adjusted_rand_score(labels, cluster_labels)
        
        return {
            'ari_score': ari_score,
            'feature_std': np.std(features),
            'feature_mean': np.mean(np.abs(features))
        }


class AutoEncoderMethod(BaseMethod):
    """AutoEncoder预训练基线"""
    
    def __init__(self, network: nn.Module, config: Dict[str, Any]):
        super().__init__("AutoEncoder", network, config)
        
        # 创建解码器
        input_dim = config.get('input_dim', 2048)  # window_size * channels
        hidden_dim = config.get('hidden_dim', 128)
        
        self.encoder = network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(self.device)
            batch_size = batch_data.size(0)
            
            # 展平输入数据
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_size, -1)
            
            optimizer.zero_grad()
            
            # 编码-解码
            encoded = self.encoder(batch_data)
            decoded = self.decoder(encoded)
            
            # 重建损失
            loss = self.criterion(decoded, batch_data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': 0.0  # AutoEncoder没有分类准确率
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)
                
                if batch_data.dim() > 2:
                    batch_data = batch_data.view(batch_size, -1)
                
                encoded = self.encoder(batch_data)
                decoded = self.decoder(encoded)
                
                loss = self.criterion(decoded, batch_data)
                total_loss += loss.item()
                num_batches += 1
        
        # 评估特征质量
        features, labels = self.get_features(dataloader)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        ari_score = adjusted_rand_score(labels, cluster_labels)
        
        return {
            'reconstruction_loss': total_loss / num_batches,
            'ari_score': ari_score,
            'feature_std': np.std(features)
        }


class MaskedReconstructionMethod(BaseMethod):
    """掩码重建预训练基线（类似BERT）"""
    
    def __init__(self, network: nn.Module, config: Dict[str, Any]):
        super().__init__("Masked_Reconstruction", network, config)
        
        self.mask_ratio = config.get('mask_ratio', 0.15)
        self.prediction_head = nn.Linear(
            config.get('hidden_dim', 128), 
            config.get('input_dim', 2048)
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.network.train()
        self.prediction_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(self.device)
            batch_size = batch_data.size(0)
            
            # 展平输入数据
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_size, -1)
            
            # 创建掩码
            mask = torch.rand(batch_data.shape) < self.mask_ratio
            masked_data = batch_data.clone()
            masked_data[mask] = 0  # 或者用随机噪声
            
            optimizer.zero_grad()
            
            # 前向传播
            features = self.network(masked_data)
            predictions = self.prediction_head(features)
            
            # 只计算被掩码位置的损失
            loss = self.criterion(predictions[mask], batch_data[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': 0.0
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.network.eval()
        self.prediction_head.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)
                
                if batch_data.dim() > 2:
                    batch_data = batch_data.view(batch_size, -1)
                
                # 创建掩码
                mask = torch.rand(batch_data.shape) < self.mask_ratio
                masked_data = batch_data.clone()
                masked_data[mask] = 0
                
                features = self.network(masked_data)
                predictions = self.prediction_head(features)
                
                loss = self.criterion(predictions[mask], batch_data[mask])
                total_loss += loss.item()
                num_batches += 1
        
        # 评估特征质量
        features, labels = self.get_features(dataloader)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        ari_score = adjusted_rand_score(labels, cluster_labels)
        
        return {
            'reconstruction_loss': total_loss / num_batches,
            'ari_score': ari_score,
            'feature_std': np.std(features)
        }


class SimCLRMethod(BaseMethod):
    """SimCLR适配版本基线"""
    
    def __init__(self, network: nn.Module, config: Dict[str, Any]):
        super().__init__("SimCLR", network, config)
        
        self.temperature = config.get('temperature', 0.07)
        
        # 添加投影头
        hidden_dim = config.get('hidden_dim', 128)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        ).to(self.device)
        
    def create_augmented_pairs(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建增强样本对"""
        batch_size = batch_data.size(0)
        
        # 简单的数据增强：添加噪声
        noise1 = torch.randn_like(batch_data) * 0.1
        noise2 = torch.randn_like(batch_data) * 0.1
        
        aug1 = batch_data + noise1
        aug2 = batch_data + noise2
        
        return aug1, aug2
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent损失函数"""
        batch_size = z1.size(0)
        
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 拼接所有特征
        z = torch.cat([z1, z2], dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(z, z.t()) / self.temperature
        
        # 创建标签：正样本对的标签
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # 排除对角线（自己和自己）
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # 计算损失
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.network.train()
        self.projection_head.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(self.device)
            batch_size = batch_data.size(0)
            
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_size, -1)
            
            # 创建增强对
            aug1, aug2 = self.create_augmented_pairs(batch_data)
            
            optimizer.zero_grad()
            
            # 前向传播
            h1 = self.network(aug1)
            h2 = self.network(aug2)
            
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            
            # 计算损失
            loss = self.nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            # 计算准确率（正样本相似度排名）
            with torch.no_grad():
                z1_norm = F.normalize(z1, dim=1)
                z2_norm = F.normalize(z2, dim=1)
                similarity = torch.sum(z1_norm * z2_norm, dim=1)
                
                # 简单的准确率估计
                acc = (similarity > 0.5).float().mean()
                total_acc += acc.item()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        # 评估特征质量
        features, labels = self.get_features(dataloader)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        ari_score = adjusted_rand_score(labels, cluster_labels)
        
        return {
            'ari_score': ari_score,
            'feature_std': np.std(features),
            'feature_separation': self._compute_feature_separation(features, labels)
        }
    
    def _compute_feature_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """计算特征分离度"""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            return silhouette_score(features, labels)
        return 0.0


class MoCoMethod(BaseMethod):
    """MoCo适配版本基线"""
    
    def __init__(self, network: nn.Module, config: Dict[str, Any]):
        super().__init__("MoCo", network, config)
        
        self.temperature = config.get('temperature', 0.07)
        self.momentum = config.get('momentum', 0.999)
        self.queue_size = config.get('queue_size', 65536)
        
        # 创建momentum encoder
        self.momentum_network = self._copy_network(network)
        self._init_momentum_network()
        
        # 创建队列
        hidden_dim = config.get('hidden_dim', 128)
        self.register_buffer("queue", torch.randn(hidden_dim, self.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def _copy_network(self, network: nn.Module) -> nn.Module:
        """复制网络结构"""
        # 简单的深拷贝
        import copy
        return copy.deepcopy(network)
    
    def _init_momentum_network(self):
        """初始化momentum网络"""
        for param_q, param_k in zip(self.network.parameters(), self.momentum_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """动量更新"""
        for param_q, param_k in zip(self.network.parameters(), self.momentum_network.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新队列"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # 为了简化
        
        # 替换队列中的keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.network.train()
        self.momentum_network.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(self.device)
            batch_size = batch_data.size(0)
            
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_size, -1)
            
            # 创建augmentation
            aug1, aug2 = self._create_augmented_pairs(batch_data)
            
            optimizer.zero_grad()
            
            # 计算query
            q = self.network(aug1)
            q = F.normalize(q, dim=1)
            
            # 计算key
            with torch.no_grad():
                self._momentum_update()
                k = self.momentum_network(aug2)
                k = F.normalize(k, dim=1)
            
            # 计算logits
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
            
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            # 更新队列
            self._dequeue_and_enqueue(k)
            
            # 计算准确率
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean()
                total_acc += acc.item()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches
        }
    
    def _create_augmented_pairs(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建增强样本对"""
        noise1 = torch.randn_like(batch_data) * 0.1
        noise2 = torch.randn_like(batch_data) * 0.1
        
        aug1 = batch_data + noise1
        aug2 = batch_data + noise2
        
        return aug1, aug2
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        # 评估特征质量
        features, labels = self.get_features(dataloader)
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        from sklearn.metrics import silhouette_score
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        ari_score = adjusted_rand_score(labels, cluster_labels)
        silhouette = silhouette_score(features, labels) if len(np.unique(labels)) > 1 else 0.0
        
        return {
            'ari_score': ari_score,
            'silhouette_score': silhouette,
            'feature_std': np.std(features)
        }


class BaselineComparison:
    """基线方法对比实验类"""
    
    def __init__(self, save_dir="./baseline_comparison_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / "baseline_comparison.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 结果存储
        self.results = defaultdict(dict)
        
        self.logger.info(f"基线对比实验初始化完成，结果保存到: {self.save_dir}")
    
    def create_simple_network(self, input_dim: int, hidden_dim: int = 128) -> nn.Module:
        """创建简单网络"""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim)
        )
    
    def create_mock_data(self, num_samples: int = 1000, signal_length: int = 2048, 
                        num_channels: int = 2, num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建模拟数据"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        data_list = []
        label_list = []
        
        for class_id in range(num_classes):
            samples_per_class = num_samples // num_classes
            
            # 为每个类别创建特征模式
            base_freq = 0.1 + class_id * 0.05
            base_amp = 0.5 + class_id * 0.1
            
            for i in range(samples_per_class):
                t = np.linspace(0, 10, signal_length)
                signal = np.zeros((signal_length, num_channels))
                
                for ch in range(num_channels):
                    # 主频率
                    signal[:, ch] = base_amp * np.sin(2 * np.pi * base_freq * t)
                    # 谐波
                    signal[:, ch] += 0.3 * base_amp * np.sin(2 * np.pi * 2 * base_freq * t)
                    # 噪声
                    signal[:, ch] += np.random.normal(0, 0.1, signal_length)
                
                # 展平信号
                signal_flat = signal.flatten()
                data_list.append(signal_flat)
                label_list.append(class_id)
        
        data = torch.tensor(np.array(data_list), dtype=torch.float32)
        labels = torch.tensor(label_list, dtype=torch.long)
        
        # 随机打乱
        indices = torch.randperm(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return data, labels
    
    def create_contrastive_baseline(self, network: nn.Module, config: Dict[str, Any]) -> 'ContrastiveIDTask':
        """创建ContrastiveIDTask基线"""
        # 这里需要适配ContrastiveIDTask
        # 由于ContrastiveIDTask依赖于特定的架构，我们创建一个简化版本
        
        class SimpleContrastiveBaseline(BaseMethod):
            def __init__(self, network, config):
                super().__init__("ContrastiveID", network, config)
                self.temperature = config.get('temperature', 0.07)
            
            def train_epoch(self, dataloader, optimizer):
                self.network.train()
                total_loss = 0.0
                total_acc = 0.0
                num_batches = 0
                
                for batch_data, _ in dataloader:
                    batch_data = batch_data.to(self.device)
                    batch_size = batch_data.size(0)
                    
                    if batch_data.dim() > 2:
                        batch_data = batch_data.view(batch_size, -1)
                    
                    # 创建正样本对（添加不同噪声）
                    noise1 = torch.randn_like(batch_data) * 0.05
                    noise2 = torch.randn_like(batch_data) * 0.05
                    
                    anchor = batch_data + noise1
                    positive = batch_data + noise2
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    z_anchor = F.normalize(self.network(anchor), dim=1)
                    z_positive = F.normalize(self.network(positive), dim=1)
                    
                    # InfoNCE损失
                    similarity_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature
                    positive_samples = torch.diag(similarity_matrix)
                    logsumexp = torch.logsumexp(similarity_matrix, dim=1)
                    loss = (-positive_samples + logsumexp).mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    # 计算准确率
                    with torch.no_grad():
                        _, predicted = torch.max(similarity_matrix, dim=1)
                        correct = torch.arange(batch_size, device=self.device)
                        acc = (predicted == correct).float().mean()
                        total_acc += acc.item()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                return {
                    'loss': total_loss / num_batches,
                    'accuracy': total_acc / num_batches
                }
            
            def evaluate(self, dataloader):
                features, labels = self.get_features(dataloader)
                
                from sklearn.cluster import KMeans
                from sklearn.metrics import adjusted_rand_score
                
                n_clusters = len(np.unique(labels))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                ari_score = adjusted_rand_score(labels, cluster_labels)
                
                return {
                    'ari_score': ari_score,
                    'feature_std': np.std(features),
                    'feature_separation': self._compute_feature_separation(features, labels)
                }
            
            def _compute_feature_separation(self, features, labels):
                from sklearn.metrics import silhouette_score
                if len(np.unique(labels)) > 1:
                    return silhouette_score(features, labels)
                return 0.0
        
        return SimpleContrastiveBaseline(network, config)
    
    def run_method_comparison(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                            test_data: torch.Tensor, test_labels: torch.Tensor,
                            epochs: int = 20) -> Dict[str, Dict[str, float]]:
        """运行所有基线方法的对比实验"""
        self.logger.info("开始基线方法对比实验")
        
        # 数据参数
        input_dim = train_data.shape[1]
        hidden_dim = 128
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 配置参数
        config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'temperature': 0.07,
            'mask_ratio': 0.15,
            'momentum': 0.999,
            'queue_size': 1024  # 减小队列大小以适应测试
        }
        
        # 创建所有基线方法
        methods = []
        
        # 1. 随机初始化
        random_network = self.create_simple_network(input_dim, hidden_dim)
        methods.append(RandomInitMethod(random_network, config))
        
        # 2. AutoEncoder
        ae_network = self.create_simple_network(input_dim, hidden_dim)
        methods.append(AutoEncoderMethod(ae_network, config))
        
        # 3. 掩码重建
        mask_network = self.create_simple_network(input_dim, hidden_dim)
        methods.append(MaskedReconstructionMethod(mask_network, config))
        
        # 4. SimCLR
        simclr_network = self.create_simple_network(input_dim, hidden_dim)
        methods.append(SimCLRMethod(simclr_network, config))
        
        # 5. MoCo
        moco_network = self.create_simple_network(input_dim, hidden_dim)
        methods.append(MoCoMethod(moco_network, config))
        
        # 6. ContrastiveID (我们的方法)
        contrastive_network = self.create_simple_network(input_dim, hidden_dim)
        methods.append(self.create_contrastive_baseline(contrastive_network, config))
        
        # 运行每个方法
        results = {}
        
        for method in methods:
            self.logger.info(f"运行方法: {method.name}")
            method_results = self._run_single_method(method, train_loader, test_loader, epochs)
            results[method.name] = method_results
            
            # 保存中间结果
            self._save_intermediate_results(results)
        
        self.results['method_comparison'] = results
        return results
    
    def _run_single_method(self, method: BaseMethod, train_loader: DataLoader, 
                          test_loader: DataLoader, epochs: int) -> Dict[str, Any]:
        """运行单个方法的训练和评估"""
        start_time = time.time()
        
        # 创建优化器
        optimizer = torch.optim.Adam(list(method.network.parameters()) + 
                                   list(getattr(method, 'decoder', nn.ModuleList()).parameters()) +
                                   list(getattr(method, 'prediction_head', nn.ModuleList()).parameters()) +
                                   list(getattr(method, 'projection_head', nn.ModuleList()).parameters()),
                                   lr=1e-3, weight_decay=1e-4)
        
        # 训练历史
        train_history = {'loss': [], 'accuracy': []}
        
        # 训练循环
        for epoch in range(epochs):
            train_metrics = method.train_epoch(train_loader, optimizer)
            
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={train_metrics['loss']:.4f}, "
                               f"Acc={train_metrics['accuracy']:.4f}")
        
        # 最终评估
        eval_metrics = method.evaluate(test_loader)
        training_time = time.time() - start_time
        
        # 获取特征用于下游评估
        features, labels = method.get_features(test_loader)
        
        # 线性分类评估
        linear_eval_score = self._linear_evaluation(features, labels)
        
        return {
            'train_history': train_history,
            'eval_metrics': eval_metrics,
            'linear_evaluation': linear_eval_score,
            'training_time': training_time,
            'final_train_loss': train_history['loss'][-1] if train_history['loss'] else 0.0,
            'final_train_accuracy': train_history['accuracy'][-1] if train_history['accuracy'] else 0.0
        }
    
    def _linear_evaluation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """线性评估：使用学到的特征训练线性分类器"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        if len(features) < 10:  # 样本太少
            return 0.0
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # 训练线性分类器
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def _save_intermediate_results(self, results: Dict[str, Any]):
        """保存中间结果"""
        results_file = self.save_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def visualize_results(self, results: Dict[str, Dict[str, Any]]):
        """可视化对比结果"""
        self.logger.info("生成对比结果可视化")
        
        # 1. 性能对比条形图
        self._plot_performance_comparison(results)
        
        # 2. 训练曲线对比
        self._plot_training_curves_comparison(results)
        
        # 3. 特征质量对比
        self._plot_feature_quality_comparison(results)
        
        # 4. 综合雷达图
        self._plot_radar_comparison(results)
    
    def _plot_performance_comparison(self, results: Dict[str, Dict[str, Any]]):
        """性能对比条形图"""
        methods = list(results.keys())
        
        # 提取性能指标
        metrics = {
            'ARI Score': [],
            'Linear Evaluation': [],
            'Feature Std': []
        }
        
        for method in methods:
            result = results[method]
            
            # ARI分数
            ari = result['eval_metrics'].get('ari_score', 0.0)
            metrics['ARI Score'].append(ari)
            
            # 线性评估分数
            linear_score = result.get('linear_evaluation', 0.0)
            metrics['Linear Evaluation'].append(linear_score)
            
            # 特征标准差（归一化）
            feature_std = result['eval_metrics'].get('feature_std', 0.0)
            metrics['Feature Std'].append(feature_std)
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]
            bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric_name} Comparison', fontsize=14)
            ax.set_ylabel(metric_name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Baseline Methods Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / 'performance_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"性能对比图已保存: {save_path}")
    
    def _plot_training_curves_comparison(self, results: Dict[str, Dict[str, Any]]):
        """训练曲线对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (method_name, result) in enumerate(results.items()):
            if 'train_history' not in result:
                continue
                
            history = result['train_history']
            epochs = range(1, len(history['loss']) + 1)
            
            color = colors[i % len(colors)]
            
            # 损失曲线
            ax1.plot(epochs, history['loss'], label=method_name, 
                    color=color, linewidth=2, alpha=0.8)
            
            # 准确率曲线
            ax2.plot(epochs, history['accuracy'], label=method_name, 
                    color=color, linewidth=2, alpha=0.8)
        
        ax1.set_title('Training Loss Comparison', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Training Accuracy Comparison', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Training Curves Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / 'training_curves_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线对比图已保存: {save_path}")
    
    def _plot_feature_quality_comparison(self, results: Dict[str, Dict[str, Any]]):
        """特征质量对比散点图"""
        methods = list(results.keys())
        
        ari_scores = []
        linear_scores = []
        
        for method in methods:
            result = results[method]
            ari = result['eval_metrics'].get('ari_score', 0.0)
            linear = result.get('linear_evaluation', 0.0)
            
            ari_scores.append(ari)
            linear_scores.append(linear)
        
        plt.figure(figsize=(10, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, method in enumerate(methods):
            plt.scatter(ari_scores[i], linear_scores[i], 
                       s=200, c=colors[i % len(colors)], 
                       alpha=0.7, label=method)
            
            # 添加方法名标签
            plt.annotate(method, (ari_scores[i], linear_scores[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.xlabel('ARI Score (Clustering Quality)', fontsize=12)
        plt.ylabel('Linear Evaluation Score (Classification)', fontsize=12)
        plt.title('Feature Quality Comparison\\n(Higher is Better)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加最佳区域标识
        plt.axhline(y=np.mean(linear_scores), color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(ari_scores), color='red', linestyle='--', alpha=0.5)
        
        save_path = self.save_dir / 'feature_quality_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"特征质量对比图已保存: {save_path}")
    
    def _plot_radar_comparison(self, results: Dict[str, Dict[str, Any]]):
        """综合雷达图对比"""
        import math
        
        methods = list(results.keys())
        
        # 定义评估维度
        dimensions = ['ARI Score', 'Linear Eval', 'Feature Std', 'Training Speed']
        
        # 提取数据
        data = {}
        for method in methods:
            result = results[method]
            
            # 归一化各个指标
            ari = result['eval_metrics'].get('ari_score', 0.0)
            linear = result.get('linear_evaluation', 0.0)
            feature_std = min(result['eval_metrics'].get('feature_std', 0.0), 2.0) / 2.0  # 归一化到0-1
            
            # 训练速度（时间越短越好，所以用倒数）
            train_time = result.get('training_time', 1.0)
            speed_score = 1.0 / (1.0 + train_time / 60)  # 转换为分钟并归一化
            
            data[method] = [ari, linear, feature_std, speed_score]
        
        # 雷达图设置
        angles = [n / float(len(dimensions)) * 2 * math.pi for n in range(len(dimensions))]
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (method, values) in enumerate(data.items()):
            values += values[:1]  # 闭合数据
            color = colors[i % len(colors)]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Comprehensive Performance Comparison\\n(Radar Chart)', size=16, pad=20)
        
        save_path = self.save_dir / 'radar_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"雷达图对比已保存: {save_path}")
    
    def generate_report(self, results: Dict[str, Dict[str, Any]]):
        """生成对比报告"""
        self.logger.info("生成基线对比报告")
        
        # 保存详细结果
        results_file = self.save_dir / "baseline_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 生成Markdown报告
        report_file = self.save_dir / "baseline_comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# ContrastiveIDTask 基线方法对比报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 对比方法\n\n")
            f.write("本实验对比了以下基线方法：\n\n")
            f.write("1. **Random_Init**: 随机初始化基线\n")
            f.write("2. **AutoEncoder**: 自编码器预训练\n")
            f.write("3. **Masked_Reconstruction**: 掩码重建预训练\n")
            f.write("4. **SimCLR**: 适配的SimCLR对比学习\n")
            f.write("5. **MoCo**: 适配的MoCo对比学习\n")
            f.write("6. **ContrastiveID**: 我们的方法\n\n")
            
            f.write("## 评估指标\n\n")
            f.write("- **ARI Score**: 调整兰德指数，评估聚类质量\n")
            f.write("- **Linear Evaluation**: 线性分类器在学习特征上的准确率\n")
            f.write("- **Feature Std**: 特征标准差，表示特征多样性\n")
            f.write("- **Training Time**: 训练耗时\n\n")
            
            f.write("## 主要结果\n\n")
            
            # 性能排序
            methods_by_linear_eval = sorted(results.items(), 
                                          key=lambda x: x[1].get('linear_evaluation', 0.0), 
                                          reverse=True)
            
            f.write("### 线性评估性能排序\n\n")
            for i, (method, result) in enumerate(methods_by_linear_eval):
                linear_score = result.get('linear_evaluation', 0.0)
                ari_score = result['eval_metrics'].get('ari_score', 0.0)
                f.write(f"{i+1}. **{method}**: 线性评估={linear_score:.4f}, ARI={ari_score:.4f}\n")
            
            f.write(f"\n### 详细结果\n\n")
            for method, result in results.items():
                f.write(f"#### {method}\n\n")
                f.write(f"- 线性评估: {result.get('linear_evaluation', 0.0):.4f}\n")
                f.write(f"- ARI分数: {result['eval_metrics'].get('ari_score', 0.0):.4f}\n")
                f.write(f"- 训练时间: {result.get('training_time', 0.0):.2f}秒\n")
                f.write(f"- 最终训练损失: {result.get('final_train_loss', 0.0):.4f}\n\n")
            
            f.write("## 可视化图表\n\n")
            f.write("- `performance_comparison.png`: 性能指标对比\n")
            f.write("- `training_curves_comparison.png`: 训练曲线对比\n")
            f.write("- `feature_quality_comparison.png`: 特征质量对比\n")
            f.write("- `radar_comparison.png`: 综合性能雷达图\n")
        
        self.logger.info(f"对比报告已保存: {report_file}")
    
    def run_full_comparison(self, num_samples: int = 1000, epochs: int = 20):
        """运行完整的基线对比实验"""
        self.logger.info("开始完整基线对比实验")
        self.logger.info(f"样本数: {num_samples}, 训练轮数: {epochs}")
        
        # 创建数据
        self.logger.info("创建实验数据...")
        train_data, train_labels = self.create_mock_data(num_samples, num_classes=10)
        test_data, test_labels = self.create_mock_data(num_samples // 5, num_classes=10)
        
        # 运行对比实验
        results = self.run_method_comparison(train_data, train_labels, test_data, test_labels, epochs)
        
        # 生成可视化
        self.visualize_results(results)
        
        # 生成报告
        self.generate_report(results)
        
        self.logger.info("基线对比实验完成")
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask基线方法对比")
    parser.add_argument("--save-dir", default="./baseline_comparison_results",
                       help="结果保存目录")
    parser.add_argument("--samples", type=int, default=1000,
                       help="每个实验的样本数量")
    parser.add_argument("--epochs", type=int, default=20,
                       help="训练轮数")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式（减少样本和轮数）")
    
    args = parser.parse_args()
    
    # 快速模式参数
    if args.quick:
        args.samples = 500
        args.epochs = 10
    
    # 运行对比实验
    comparison = BaselineComparison(save_dir=args.save_dir)
    results = comparison.run_full_comparison(num_samples=args.samples, epochs=args.epochs)
    
    print(f"\\n实验完成！结果保存在: {args.save_dir}")
    print("\\n主要结果:")
    for method, result in results.items():
        linear_score = result.get('linear_evaluation', 0.0)
        ari_score = result['eval_metrics'].get('ari_score', 0.0)
        print(f"  {method}: Linear={linear_score:.4f}, ARI={ari_score:.4f}")


if __name__ == "__main__":
    main()
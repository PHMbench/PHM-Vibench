"""
M_04_ISFM_Flow - 主Flow集成模型
结合RectifiedFlow + 条件编码 + 维度适配
遵循PHM-Vibench工厂模式
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    from .layers.flow_model import RectifiedFlow
    from .layers.condition_encoder import ConditionalEncoder, AdaptiveConditionalEncoder
    from .layers.utils.flow_utils import DimensionAdapter, validate_tensor_shape
except ImportError:
    from layers.flow_model import RectifiedFlow
    from layers.condition_encoder import ConditionalEncoder, AdaptiveConditionalEncoder
    from layers.utils.flow_utils import DimensionAdapter, validate_tensor_shape


class Model(nn.Module):
    """
    M_04_ISFM_Flow主模型类
    
    功能:
    1. 处理(B,L,C) -> (B,L*C)维度适配
    2. 条件编码(基于metadata)
    3. RectifiedFlow生成建模
    4. 支持训练、采样、异常检测
    """
    
    def __init__(self, args_m, metadata=None):
        super().__init__()
        
        # 配置参数
        self.sequence_length = getattr(args_m, 'sequence_length', 1024)
        self.channels = getattr(args_m, 'channels', 1)
        self.latent_dim = self.sequence_length * self.channels  # 展开后的维度
        
        # Flow模型参数
        self.hidden_dim = getattr(args_m, 'hidden_dim', 256)
        self.time_dim = getattr(args_m, 'time_dim', 64)
        self.condition_dim = getattr(args_m, 'condition_dim', 64)
        
        # 条件编码器参数
        self.use_conditional = getattr(args_m, 'use_conditional', True)
        
        print(f"🚀 初始化M_04_ISFM_Flow:")
        print(f"   - 序列长度: {self.sequence_length}")
        print(f"   - 通道数: {self.channels}")  
        print(f"   - 潜在维度: {self.latent_dim}")
        print(f"   - 使用条件编码: {self.use_conditional}")
        
        # 创建条件编码器
        if self.use_conditional and metadata is not None:
            self.condition_encoder = AdaptiveConditionalEncoder.from_metadata(
                metadata.df,
                embed_dim=self.condition_dim
            )
        elif self.use_conditional:
            # 使用默认配置
            self.condition_encoder = ConditionalEncoder(
                embed_dim=self.condition_dim,
                num_domains=getattr(args_m, 'num_domains', 50),
                num_systems=getattr(args_m, 'num_systems', 50)
            )
        else:
            self.condition_encoder = None
            self.condition_dim = 0
        
        # 创建Flow模型
        self.flow_model = RectifiedFlow(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            time_dim=self.time_dim,
            condition_dim=self.condition_dim,
            sigma_min=getattr(args_m, 'sigma_min', 0.001),
            sigma_max=getattr(args_m, 'sigma_max', 1.0)
        )
        
        # 保存metadata引用
        self.metadata = metadata
        
        print(f"   ✅ Flow模型初始化完成")
        print(f"   ✅ 模型参数总数: {sum(p.numel() for p in self.parameters()):,}")
    
    def _encode_conditions(self, file_ids: List[str]) -> Optional[torch.Tensor]:
        """
        从file_id列表编码条件
        
        Args:
            file_ids: 文件ID列表
        
        Returns:
            condition_features: 条件特征 (batch_size, condition_dim)
        """
        if not self.use_conditional or self.condition_encoder is None:
            return None
        
        if self.metadata is None:
            raise ValueError("需要metadata来提取条件信息")
        
        # 从metadata提取信息
        metadata_batch = []
        for file_id in file_ids:
            if file_id in self.metadata:
                metadata_batch.append(dict(self.metadata[file_id]))
            else:
                # 使用默认值处理缺失的file_id
                metadata_batch.append({
                    'Domain_id': None,
                    'Dataset_id': None,
                    'Name': 'unknown'
                })
        
        return self.condition_encoder(metadata_batch)
    
    def forward(self, x: torch.Tensor, file_ids: Optional[List[str]] = None,
                return_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 (B, L, C)
            file_ids: 文件ID列表，用于条件编码
            return_loss: 是否计算并返回损失
        
        Returns:
            outputs: 包含模型输出和损失的字典
        """
        validate_tensor_shape(x, 3, "input x")
        
        batch_size, seq_len, channels = x.shape
        
        # 验证输入维度
        if seq_len != self.sequence_length or channels != self.channels:
            print(f"⚠️  维度不匹配: 期望({self.sequence_length}, {self.channels}), "
                  f"实际({seq_len}, {channels})")
        
        # 1. 维度适配: (B, L, C) -> (B, L*C)
        x_flat = DimensionAdapter.encode_3d_to_1d(x)
        
        # 2. 条件编码
        condition_features = None
        if file_ids is not None:
            condition_features = self._encode_conditions(file_ids)
        
        # 3. Flow模型前向传播
        flow_outputs = self.flow_model(x_flat, condition_features)
        
        # 4. 计算损失
        if return_loss:
            losses = self.flow_model.compute_loss(flow_outputs)
            flow_outputs.update(losses)
        
        # 添加额外信息
        flow_outputs.update({
            'x_original': x,
            'x_flat': x_flat,
            'condition_features': condition_features,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'channels': channels
        })
        
        return flow_outputs
    
    def sample(self, batch_size: int, file_ids: Optional[List[str]] = None,
               num_steps: int = 50, device: Optional[str] = None) -> torch.Tensor:
        """
        采样生成新数据
        
        Args:
            batch_size: 批量大小
            file_ids: 文件ID列表（用于条件生成）
            num_steps: 采样步数
            device: 计算设备
        
        Returns:
            samples: 生成样本 (batch_size, sequence_length, channels)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # 条件编码
        condition_features = None
        if file_ids is not None:
            condition_features = self._encode_conditions(file_ids)
        
        # Flow采样
        samples_flat = self.flow_model.sample(
            batch_size=batch_size,
            condition=condition_features,
            num_steps=num_steps,
            device=device
        )
        
        # 维度恢复: (B, L*C) -> (B, L, C)
        samples = DimensionAdapter.decode_1d_to_3d(
            samples_flat, self.sequence_length, self.channels
        )
        
        return samples
    
    def encode_to_noise(self, x: torch.Tensor, file_ids: Optional[List[str]] = None,
                       num_steps: int = 50) -> torch.Tensor:
        """
        将数据编码到噪声空间（用于异常检测）
        
        Args:
            x: 输入数据 (B, L, C)
            file_ids: 文件ID列表
            num_steps: 编码步数
        
        Returns:
            noise: 对应的噪声 (B, L, C)
        """
        validate_tensor_shape(x, 3, "input x")
        
        # 维度适配
        x_flat = DimensionAdapter.encode_3d_to_1d(x)
        
        # 条件编码
        condition_features = None
        if file_ids is not None:
            condition_features = self._encode_conditions(file_ids)
        
        # 编码到噪声
        noise_flat = self.flow_model.encode_to_noise(
            x_flat, condition_features, num_steps
        )
        
        # 维度恢复
        noise = DimensionAdapter.decode_1d_to_3d(
            noise_flat, self.sequence_length, self.channels
        )
        
        return noise
    
    def compute_anomaly_score(self, x: torch.Tensor, file_ids: Optional[List[str]] = None,
                             num_steps: int = 50) -> torch.Tensor:
        """
        计算异常分数
        
        Args:
            x: 输入数据 (B, L, C)
            file_ids: 文件ID列表
            num_steps: 计算步数
        
        Returns:
            scores: 异常分数 (B,)
        """
        # 编码到噪声空间
        noise = self.encode_to_noise(x, file_ids, num_steps)
        
        # 计算噪声的L2范数作为异常分数
        scores = torch.norm(noise.view(noise.size(0), -1), dim=1)
        
        return scores


# 测试代码
if __name__ == '__main__':
    print("🔬 测试M_04_ISFM_Flow集成模型")
    
    # Mock配置
    class MockArgs:
        def __init__(self):
            self.sequence_length = 1024
            self.channels = 1
            self.hidden_dim = 128
            self.time_dim = 32
            self.condition_dim = 64
            self.use_conditional = True
    
    # Mock metadata
    class MockMetadata:
        def __init__(self):
            import pandas as pd
            self.df = pd.DataFrame({
                'Domain_id': [1, 2, 1, 3],
                'Dataset_id': [5, 8, 5, 10],
                'Name': ['CWRU', 'XJTU', 'PU', 'FEMTO']
            })
        
        def __contains__(self, key):
            return key in ['file1', 'file2', 'file3']
        
        def __getitem__(self, key):
            if key == 'file1':
                return {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'CWRU'}
            elif key == 'file2':
                return {'Domain_id': 2, 'Dataset_id': 8, 'Name': 'XJTU'}
            else:
                return {'Domain_id': 3, 'Dataset_id': 10, 'Name': 'FEMTO'}
    
    args = MockArgs()
    metadata = MockMetadata()
    
    # 创建模型
    model = Model(args, metadata)
    
    # 测试数据
    batch_size = 4
    x = torch.randn(batch_size, args.sequence_length, args.channels)
    file_ids = ['file1', 'file2', 'file3', 'file1']
    
    print(f"\n📊 测试输入:")
    print(f"   - 数据形状: {x.shape}")
    print(f"   - 文件ID: {file_ids}")
    
    # 前向传播测试
    outputs = model(x, file_ids)
    print(f"\n✅ 前向传播成功:")
    print(f"   - v_pred形状: {outputs['v_pred'].shape}")
    print(f"   - 总损失: {outputs['total_loss'].item():.6f}")
    
    # 采样测试
    samples = model.sample(batch_size=2, file_ids=['file1', 'file2'], num_steps=20)
    print(f"\n✅ 采样成功:")
    print(f"   - 样本形状: {samples.shape}")
    
    # 异常检测测试
    anomaly_scores = model.compute_anomaly_score(x, file_ids, num_steps=20)
    print(f"\n✅ 异常检测成功:")
    print(f"   - 异常分数形状: {anomaly_scores.shape}")
    print(f"   - 分数范围: [{anomaly_scores.min().item():.3f}, {anomaly_scores.max().item():.3f}]")
    
    print(f"\n🎉 M_04_ISFM_Flow集成测试通过！")
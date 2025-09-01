"""
Flow工具函数 - 维度适配和基础工具
遵循简化原则，仅包含必要功能
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
import pandas as pd


class DimensionAdapter:
    """简单的维度适配器 - 直接展开方案"""
    
    @staticmethod
    def encode_3d_to_1d(x: torch.Tensor) -> torch.Tensor:
        """
        编码: (B, L, C) -> (B, L*C)
        最简单的展开方案
        """
        B, L, C = x.shape
        return x.view(B, L * C)
    
    @staticmethod
    def decode_1d_to_3d(x: torch.Tensor, seq_len: int, channels: int) -> torch.Tensor:
        """
        解码: (B, L*C) -> (B, L, C)
        """
        B = x.shape[0]
        return x.view(B, seq_len, channels)


class TimeEmbedding(nn.Module):
    """简单的正弦位置编码"""
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 (batch_size,) 范围 [0, 1]
        Returns:
            pos_emb: (batch_size, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 处理奇数维度
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device)], dim=-1)
        
        return emb


class MetadataExtractor:
    """从PHM-Vibench metadata提取条件信息"""
    
    @staticmethod
    def extract_condition_ids(metadata_dict) -> Tuple[int, int]:
        """
        从metadata字典提取domain_id和system_id
        
        Args:
            metadata_dict: 单个样本的metadata信息
            
        Returns:
            (domain_id, system_id)，未知值返回0
        """
        # 提取Domain_id
        domain_id = metadata_dict.get('Domain_id', -1)
        if pd.isna(domain_id) or domain_id is None:
            domain_id = 0  # 未知域使用0
        else:
            domain_id = max(0, int(domain_id))  # 确保非负
        
        # 提取Dataset_id作为system_id
        system_id = metadata_dict.get('Dataset_id', -1)
        if pd.isna(system_id) or system_id is None:
            system_id = 0  # 未知系统使用0
        else:
            system_id = max(0, int(system_id))  # 确保非负
        
        return domain_id, system_id
    
    @staticmethod
    def get_max_ids(metadata_df) -> Tuple[int, int]:
        """
        从metadata DataFrame获取最大的domain_id和system_id
        用于设置embedding大小
        """
        # 获取有效的domain和system值
        valid_domains = metadata_df['Domain_id'].dropna()
        valid_systems = metadata_df['Dataset_id'].dropna()
        
        max_domain = int(valid_domains.max()) if len(valid_domains) > 0 else 0
        max_system = int(valid_systems.max()) if len(valid_systems) > 0 else 0
        
        return max_domain, max_system


def simple_flow_loss(v_pred: torch.Tensor, v_true: torch.Tensor) -> torch.Tensor:
    """简单的流匹配损失"""
    return torch.nn.functional.mse_loss(v_pred, v_true)


def validate_tensor_shape(x: torch.Tensor, expected_dims: int, name: str = "tensor"):
    """验证张量形状"""
    if x.dim() != expected_dims:
        raise ValueError(f"{name} expected {expected_dims}D tensor, got {x.dim()}D")
    
    if torch.isnan(x).any():
        raise ValueError(f"{name} contains NaN values")
    
    if torch.isinf(x).any():
        raise ValueError(f"{name} contains infinite values")
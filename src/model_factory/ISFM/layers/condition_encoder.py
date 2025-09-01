"""
条件编码器 - 直接使用PHM-Vibench metadata
支持Domain_id和Dataset_id的层次化编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
try:
    from .utils.flow_utils import MetadataExtractor
except ImportError:
    from utils.flow_utils import MetadataExtractor


class ConditionalEncoder(nn.Module):
    """
    条件编码器 - 简化版本
    直接使用metadata中的Domain_id和Dataset_id
    """
    
    def __init__(self, embed_dim: int = 64, num_domains: int = 50, 
                 num_systems: int = 50, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_domains = num_domains
        self.num_systems = num_systems
        
        # 域嵌入 (padding_idx=0 表示未知域)
        self.domain_embedding = nn.Embedding(
            num_domains + 1,  # +1 for unknown
            embed_dim,
            padding_idx=0
        )
        
        # 系统嵌入 (padding_idx=0 表示未知系统)
        self.system_embedding = nn.Embedding(
            num_systems + 1,  # +1 for unknown
            embed_dim, 
            padding_idx=0
        )
        
        # 融合层 - 简单的注意力机制
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
    
    def forward(self, metadata_batch: List[Dict[str, Any]]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            metadata_batch: metadata字典列表，每个对应批次中的一个样本
        
        Returns:
            condition_features: 条件特征 (batch_size, embed_dim)
        """
        if not metadata_batch:
            raise ValueError("metadata_batch不能为空")
        
        batch_size = len(metadata_batch)
        device = next(self.parameters()).device
        
        # 提取domain_id和system_id
        domain_ids = []
        system_ids = []
        
        for metadata in metadata_batch:
            domain_id, system_id = MetadataExtractor.extract_condition_ids(metadata)
            
            # 确保ID在有效范围内，超出范围的设为0(未知)
            domain_id = min(domain_id, self.num_domains) if domain_id > 0 else 0
            system_id = min(system_id, self.num_systems) if system_id > 0 else 0
            
            domain_ids.append(domain_id)
            system_ids.append(system_id)
        
        # 转换为张量
        domain_ids = torch.tensor(domain_ids, device=device, dtype=torch.long)
        system_ids = torch.tensor(system_ids, device=device, dtype=torch.long)
        
        # 获取嵌入
        domain_emb = self.domain_embedding(domain_ids)  # (batch_size, embed_dim)
        system_emb = self.system_embedding(system_ids)  # (batch_size, embed_dim)
        
        # 拼接并融合
        combined = torch.cat([domain_emb, system_emb], dim=-1)  # (batch_size, embed_dim*2)
        condition_features = self.fusion(combined)  # (batch_size, embed_dim)
        
        return condition_features
    
    def get_domain_prototype(self, domain_id: int) -> torch.Tensor:
        """获取特定域的原型向量"""
        domain_id = min(max(domain_id, 0), self.num_domains)
        domain_tensor = torch.tensor([domain_id], device=next(self.parameters()).device)
        return self.domain_embedding(domain_tensor).squeeze(0)
    
    def get_system_prototype(self, system_id: int) -> torch.Tensor:
        """获取特定系统的原型向量"""
        system_id = min(max(system_id, 0), self.num_systems)
        system_tensor = torch.tensor([system_id], device=next(self.parameters()).device)
        return self.system_embedding(system_tensor).squeeze(0)


class AdaptiveConditionalEncoder(ConditionalEncoder):
    """
    自适应条件编码器
    根据metadata自动调整域和系统的数量
    """
    
    @classmethod
    def from_metadata(cls, metadata_df, embed_dim: int = 64, 
                     margin: int = 10, **kwargs):
        """
        从metadata DataFrame创建编码器
        
        Args:
            metadata_df: PHM-Vibench的metadata DataFrame
            embed_dim: 嵌入维度
            margin: 预留的扩展空间
        """
        max_domain, max_system = MetadataExtractor.get_max_ids(metadata_df)
        
        # 添加预留空间
        num_domains = max_domain + margin
        num_systems = max_system + margin
        
        print(f"创建自适应条件编码器:")
        print(f"  - 域数量: {num_domains} (最大ID: {max_domain})")
        print(f"  - 系统数量: {num_systems} (最大ID: {max_system})")
        
        return cls(
            embed_dim=embed_dim,
            num_domains=num_domains,
            num_systems=num_systems,
            **kwargs
        )


# 测试代码
if __name__ == '__main__':
    print("🔬 测试条件编码器")
    
    # 创建编码器
    encoder = ConditionalEncoder(embed_dim=64, num_domains=10, num_systems=20)
    
    # 模拟metadata批次
    metadata_batch = [
        {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'CWRU'},
        {'Domain_id': 2, 'Dataset_id': 8, 'Name': 'XJTU'},
        {'Domain_id': None, 'Dataset_id': 3, 'Name': 'PU'},  # 缺失Domain_id
        {'Domain_id': 1, 'Dataset_id': None, 'Name': 'FEMTO'},  # 缺失Dataset_id
    ]
    
    # 前向传播
    condition_features = encoder(metadata_batch)
    print(f"✅ 条件编码成功，输出形状: {condition_features.shape}")
    print(f"✅ 特征统计: 均值={condition_features.mean().item():.4f}, 标准差={condition_features.std().item():.4f}")
    
    # 测试原型获取
    domain_proto = encoder.get_domain_prototype(1)
    system_proto = encoder.get_system_prototype(5)
    print(f"✅ 域原型形状: {domain_proto.shape}")
    print(f"✅ 系统原型形状: {system_proto.shape}")
    
    # 测试不同条件的区分度
    same_condition = [
        {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'Test1'},
        {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'Test2'},
    ]
    
    diff_condition = [
        {'Domain_id': 1, 'Dataset_id': 5, 'Name': 'Test1'},
        {'Domain_id': 2, 'Dataset_id': 8, 'Name': 'Test2'},
    ]
    
    same_features = encoder(same_condition)
    diff_features = encoder(diff_condition)
    
    same_similarity = F.cosine_similarity(same_features[0], same_features[1], dim=0)
    diff_similarity = F.cosine_similarity(diff_features[0], diff_features[1], dim=0)
    
    print(f"✅ 相同条件相似度: {same_similarity.item():.4f}")
    print(f"✅ 不同条件相似度: {diff_similarity.item():.4f}")
    
    print("🎉 条件编码器测试通过！")
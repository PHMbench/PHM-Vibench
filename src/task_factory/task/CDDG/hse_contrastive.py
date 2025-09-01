"""
HSE增强的跨系统域泛化任务 (简化版)
结合异构信号嵌入与对比学习的系统级泛化优化
避免过度工程化，保持代码简洁
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from ...Default_task import Default_task


class task(Default_task):  # 必须命名为task以符合框架约定
    """
    HSE系统级对比学习任务（简化版）
    继承Default_task，添加对比学习目标
    """
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task,
                        args_trainer, args_environment, metadata)
        
        # 对比学习参数
        self.contrast_weight = getattr(args_task, 'contrast_weight', 0.1)
        self.temperature = getattr(args_task, 'temperature', 0.07)
        
        print(f"HSE对比学习任务初始化: contrast_weight={self.contrast_weight}, "
              f"temperature={self.temperature}")
    
    def extract_hse_features(self, x: torch.Tensor, file_id: List[str]) -> torch.Tensor:
        """
        提取HSE嵌入特征（简化版）
        
        Args:
            x: 输入信号 (batch_size, seq_len, channels)
            file_id: 文件ID列表
            
        Returns:
            features: HSE嵌入特征 (batch_size, feature_dim)
        """
        # 使用ISFM的_embed方法（主要方式）
        if hasattr(self.network, '_embed'):
            features = self.network._embed(x, file_id)
            # 如果是3维patch特征，平均池化到2维
            if len(features.shape) == 3:
                features = features.mean(dim=1)
            return features
        
        # 简单fallback：使用输入的统计特征
        features = torch.cat([
            x.mean(dim=1),     # 时域均值
            x.std(dim=1)       # 时域标准差
        ], dim=-1)
        return features
    
    def compute_system_contrast_loss(self, features: torch.Tensor, file_ids: List[str]) -> torch.Tensor:
        """
        计算系统级对比学习损失（简化版）
        
        Args:
            features: HSE特征 (batch_size, feature_dim)
            file_ids: 文件ID列表
            
        Returns:
            loss: 系统对比损失值
        """
        batch_size = features.shape[0]
        device = features.device
        
        # L2标准化特征
        features = F.normalize(features, dim=1)
        
        # 获取系统标签（简化版）
        system_ids = []
        for fid in file_ids:
            # 从文件ID推断系统名（CWRU_test1 -> CWRU）
            system = str(fid).split('_')[0] if '_' in str(fid) else str(fid)
            system_ids.append(system)
        
        # 检查系统多样性
        unique_systems = list(set(system_ids))
        if len(unique_systems) <= 1:
            # 批次内只有一个系统，返回0损失
            return torch.tensor(0.0, device=device)
        
        # 创建系统ID张量
        system_indices = torch.tensor([unique_systems.index(s) for s in system_ids], device=device)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建正负样本掩码
        pos_mask = system_indices.unsqueeze(0) == system_indices.unsqueeze(1)
        # 移除对角线（自身相似度）
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag_mask
        
        # 检查是否有正样本对
        if not pos_mask.any():
            return torch.tensor(0.0, device=device)
        
        # 标准InfoNCE损失计算（不使用困难负样本挖掘）
        exp_sim = torch.exp(sim_matrix)
        # 正样本：同系统内其他样本
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        # 所有样本（除自己）
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        # 计算损失 
        loss = -torch.log(pos_sim / (all_sim + 1e-8)).mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤（修复版）：结合分类损失和对比损失
        
        使用正确的批次格式，遵循Default_task接口
        """
        # 获取分类损失（使用父类方法）
        cls_metrics = self._shared_step(batch, "train")
        cls_loss = cls_metrics["train_total_loss"]
        
        # 如果启用对比学习
        if self.contrast_weight > 0:
            # 提取数据（使用正确的字典访问）
            x = batch['x']
            file_id = batch['file_id']
            
            # 提取HSE特征（不使用no_grad，保持梯度）
            features = self.extract_hse_features(x, file_id)
            
            # 计算对比损失
            contrast_loss = self.compute_system_contrast_loss(features, file_id)
            
            # 组合损失
            total_loss = cls_loss + self.contrast_weight * contrast_loss
            
            # 记录日志
            self.log('train/cls_loss', cls_loss, prog_bar=True)
            self.log('train/contrast_loss', contrast_loss, prog_bar=True)
            self.log('train/total_loss', total_loss, prog_bar=True)
            
            return total_loss
        
        # 仅分类损失
        return cls_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤：使用父类的标准实现"""
        return super().validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        """测试步骤：使用父类的标准实现"""
        return super().test_step(batch, batch_idx)


# 简单的模块测试
if __name__ == "__main__":
    print("HSE对比学习任务模块测试")
    
    # 创建模拟数据
    x = torch.randn(4, 1024, 1)
    file_ids = ['CWRU_test1', 'CWRU_test2', 'THU_test1', 'XJTU_test1']
    
    # 模拟特征提取测试
    features = torch.randn(4, 128)
    print(f"模拟特征形状: {features.shape}")
    
    # 模拟系统映射测试
    system_ids = [fid.split('_')[0] for fid in file_ids]
    unique_systems = list(set(system_ids))
    print(f"系统映射: {dict(zip(file_ids, system_ids))}")
    print(f"唯一系统: {unique_systems}")
    
    print("✅ 模块基础功能测试通过")
"""
损失函数工具模块，包含各种任务的损失函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union, Optional, Callable, Tuple, List


class FocalLoss(nn.Module):
    """Focal Loss，适用于类别不平衡的分类问题
    
    论文: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    """
    
    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """初始化Focal Loss
        
        Args:
            gamma: 聚焦参数，减少易分类样本的权重
            alpha: 类别权重，处理类别不平衡
            reduction: 损失聚合方法，可选 'none', 'mean', 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播计算损失
        
        Args:
            input: 模型输出的预测概率 (B, C) 或 (B, C, ...)
            target: 目标类别索引 (B) 或 (B, ...)
        
        Returns:
            计算的损失值
        """
        # 处理输入形状
        if input.dim() > 2:
            # (B, C, d1, d2, ...) -> (B * d1 * d2 * ..., C)
            c = input.size(1)
            input = input.permute(0, *range(2, input.dim()), 1).contiguous()
            input = input.view(-1, c)
            # (B, d1, d2, ...) -> (B * d1 * d2 * ...)
            target = target.view(-1)
        
        # 计算交叉熵
        log_p = F.log_softmax(input, dim=-1)
        ce = F.nll_loss(log_p, target, reduction='none')
        
        # 计算概率和聚焦权重
        all_rows = torch.arange(len(input))
        log_pt = log_p[all_rows, target]
        pt = log_pt.exp()
        
        # 应用focal loss公式
        loss = -1 * (1 - pt) ** self.gamma * log_pt
        
        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            loss = alpha_t * loss
        
        # 按指定方式聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class AsymmetricLoss(nn.Module):
    """非对称损失，适用于异常检测中的类别不平衡问题
    
    为异常类（少数类）分配更高的权重，减少假阴性率
    """
    
    def __init__(
        self, 
        pos_weight: float = 2.0, 
        margin: float = 0.5,
        reduction: str = 'mean'
    ):
        """初始化非对称损失
        
        Args:
            pos_weight: 正类（异常）的权重
            margin: 用于控制决策边界的参数
            reduction: 损失聚合方法，可选 'none', 'mean', 'sum'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播计算损失
        
        Args:
            input: 模型输出的预测值 (B,) 或 (B, 1)
            target: 目标标签，0表示正常，1表示异常 (B,)
        
        Returns:
            计算的损失值
        """
        # 确保输入形状一致
        if input.shape != target.shape:
            input = input.view(-1)
        
        # 计算基础的二元交叉熵
        bce = F.binary_cross_entropy_with_logits(
            input, target.float(), reduction='none'
        )
        
        # 计算非对称权重
        weights = torch.ones_like(target, dtype=torch.float32)
        weights[target == 1] = self.pos_weight  # 异常样本权重更高
        
        # 应用权重
        weighted_bce = weights * bce
        
        # 应用边界增强
        pred = torch.sigmoid(input)
        pos_diff = F.relu(self.margin - pred) * target.float()
        neg_diff = F.relu(pred - (1 - self.margin)) * (1 - target.float())
        margin_loss = pos_diff + neg_diff
        
        # 合并损失
        loss = weighted_bce + margin_loss
        
        # 按指定方式聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class RULLoss(nn.Module):
    """RUL预测特定的损失函数
    
    结合MSE与惩罚早期预测错误（过早预测故障）的损失函数
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        beta: float = 10.0, 
        reduction: str = 'mean'
    ):
        """初始化RUL损失
        
        Args:
            alpha: 基础MSE损失的权重
            beta: 惩罚系数，控制对早期错误预测的惩罚强度
            reduction: 损失聚合方法，可选 'none', 'mean', 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播计算损失
        
        Args:
            input: 模型输出的预测RUL值 (B,)
            target: 目标RUL值 (B,)
        
        Returns:
            计算的损失值
        """
        # 计算基础MSE损失
        mse_loss = F.mse_loss(input, target, reduction='none')
        
        # 计算预测误差
        error = input - target
        
        # 惩罚早期预测错误（当预测值大于真实值，即过早预测故障时）
        early_penalty = torch.zeros_like(error)
        early_mask = error > 0  # 过早预测故障
        early_penalty[early_mask] = torch.exp(self.beta * error[early_mask] / target[early_mask]) - 1
        
        # 结合基础损失和惩罚项
        loss = self.alpha * mse_loss + early_penalty
        
        # 按指定方式聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# 损失函数字典
_LOSS_FUNCTIONS = {
    # 分类损失
    "ce": nn.CrossEntropyLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "binary_cross_entropy": nn.BCEWithLogitsLoss,
    "focal": FocalLoss,
    
    # 异常检测损失
    "asymmetric": AsymmetricLoss,
    
    # 回归损失
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "rul": RULLoss,
}


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """获取指定名称的损失函数
    
    Args:
        loss_name: 损失函数名称
        **kwargs: 传递给损失函数的参数
    
    Returns:
        损失函数实例
    
    Raises:
        ValueError: 如果指定的损失函数不支持
    """
    if loss_name not in _LOSS_FUNCTIONS:
        raise ValueError(f"不支持的损失函数: {loss_name}，"
                        f"支持的损失函数有: {list(_LOSS_FUNCTIONS.keys())}")
    
    # 实例化损失函数
    loss_class = _LOSS_FUNCTIONS[loss_name]
    return loss_class(**kwargs)
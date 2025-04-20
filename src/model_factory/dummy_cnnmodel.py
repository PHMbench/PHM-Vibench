import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from src.model_factory import register_model
from src.model_factory.base_model import BaseModel

@register_model('DummyCNNModel')
class DummyCNNModel(BaseModel):
    """示例 CNN 模型，用于框架测试
    
    一个简单的 CNN 模型，用于信号处理或图像分类
    """
    
    def __init__(self, 
                input_channels=1, 
                hidden_channels=[16, 32, 64], 
                output_dim=1, 
                kernel_size=3,
                dropout_rate=0.2,
                **kwargs):
        """初始化模型
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏层通道数
            output_dim: 输出维度
            kernel_size: 卷积核大小
            dropout_rate: Dropout 概率
            **kwargs: 其他参数
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # 构建网络结构
        self.build_network()
        
    def build_network(self):
        """构建网络结构"""
        # 构建卷积层
        layers = []
        in_channels = self.input_channels
        
        for out_channels in self.hidden_channels:
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2  # 保持大小不变
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(self.dropout_rate))
            
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # 使用自适应池化，使得输入大小的变化不影响全连接层的输入维度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.output_dim)
        )
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据，形状为 [batch_size, feature_dim] 或 [batch_size, channels, seq_len]
            
        Returns:
            输出数据，形状为 [batch_size, output_dim]
        """
        # 确保输入形状正确 [batch_size, channels, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 特征提取
        x = self.features(x)
        
        # 全局池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 分类
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        """进行预测
        
        Args:
            x: 输入数据
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """添加模型特定的命令行参数
        
        Args:
            parent_parser: 父解析器
            
        Returns:
            更新后的解析器
        """
        parser = parent_parser.add_argument_group("DummyCNNModel")
        parser.add_argument("--input_channels", type=int, default=1)
        parser.add_argument("--hidden_channels", type=int, nargs='+', default=[16, 32, 64])
        parser.add_argument("--output_dim", type=int, default=1)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--dropout_rate", type=float, default=0.2)
        return parent_parser


if __name__ == '__main__':
    """测试入口点"""
    import argparse
    import numpy as np
    
    # 创建解析器
    parser = argparse.ArgumentParser(description='DummyCNNModel 测试')
    parser = DummyCNNModel.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # 创建模型实例
    model = DummyCNNModel(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        output_dim=args.output_dim,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout_rate
    )
    
    # 打印模型结构
    print(model)
    
    # 测试前向传播
    batch_size = 16
    seq_len = 128
    x = torch.randn(batch_size, args.input_channels, seq_len)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预期输出形状: [batch_size, output_dim] = [{batch_size}, {args.output_dim}]")
    
    # 测试单特征向量输入
    x_single = torch.randn(batch_size, seq_len)  # 无通道维度
    output_single = model(x_single)
    print(f"单特征输入形状: {x_single.shape}")
    print(f"单特征输出形状: {output_single.shape}")
    
    print("\n模型测试成功!")
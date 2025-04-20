import torch
import torch.nn as nn
from src.model_factory import register_model
from src.model_factory.base_model import BaseModel

@register_model('DummyCNNModel')  # 修改注册名称
class DummyCNNModel(BaseModel):  # 修改类名
    """示例CNN模型，用于测试框架
    
    简单的前馈CNN网络，可用于回归和分类任务
    """
    
    def __init__(self, 
                input_channels=1, 
                hidden_channels=None, 
                output_dim=1,
                dropout_rate=0.2,
                **kwargs):
        """初始化CNN模型
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏层通道数列表，如 [32, 64, 128]
            output_dim: 输出维度
            dropout_rate: dropout比率
            **kwargs: 其他参数
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
            
        # 构建网络层
        layers = []
        
        # 第一个卷积层
        layers.append(nn.Conv1d(input_channels, hidden_channels[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(hidden_channels[0]))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))
        layers.append(nn.Dropout(dropout_rate))
        
        # 后续卷积层
        for i in range(len(hidden_channels) - 1):
            layers.append(
                nn.Conv1d(hidden_channels[i], hidden_channels[i+1], kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm1d(hidden_channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout_rate))
        
        self.cnn_layers = nn.Sequential(*layers)
        
        # 计算全连接层输入维度
        # 假设输入形状为 (batch_size, input_channels, 10)
        # 经过n次MaxPool2d后，尺寸变为 10 / (2^n)
        fc_input_dim = hidden_channels[-1] * (10 // (2 ** len(hidden_channels)))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 形状为 (batch_size, features) 的输入张量
            
        Returns:
            形状为 (batch_size, output_dim) 的输出张量
        """
        # 确保输入形状正确 [batch, features] -> [batch, channels, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 通过CNN层
        x = self.cnn_layers(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 通过全连接层
        x = self.fc(x)
        
        return x


if __name__ == '__main__':
    """测试入口点"""
    # 创建模型实例
    model = DummyCNNModel(input_channels=1, hidden_channels=[16, 32, 64], output_dim=10)  # 修改类名
    
    # 打印模型概要
    print(model.summary())
    
    # 测试前向传播
    batch_size = 4
    feature_dim = 10
    
    # 创建随机输入
    x = torch.randn(batch_size, feature_dim)
    
    # 前向传播
    out = model(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    
    print("\n模型测试成功!")
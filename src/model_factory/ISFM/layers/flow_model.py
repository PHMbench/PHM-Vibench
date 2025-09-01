"""
RectifiedFlow核心模型 - 最简实现
仅包含Euler求解器和基础功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
try:
    from .utils.flow_utils import TimeEmbedding, DimensionAdapter, simple_flow_loss, validate_tensor_shape
except ImportError:
    from utils.flow_utils import TimeEmbedding, DimensionAdapter, simple_flow_loss, validate_tensor_shape


class VelocityNetwork(nn.Module):
    """速度预测网络 - 简单MLP实现"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256, 
                 time_dim: int = 64, condition_dim: int = 0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        
        # 计算输入维度
        input_dim = latent_dim + time_dim
        if condition_dim > 0:
            input_dim += condition_dim
        
        # 简单的3层MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Xavier初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x_t: torch.Tensor, t_emb: torch.Tensor, 
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测速度场
        
        Args:
            x_t: 插值点 (batch_size, latent_dim)
            t_emb: 时间嵌入 (batch_size, time_dim)
            condition: 条件 (batch_size, condition_dim)
        
        Returns:
            v: 速度场 (batch_size, latent_dim)
        """
        # 拼接输入
        inputs = [x_t, t_emb]
        if condition is not None:
            inputs.append(condition)
        
        x_input = torch.cat(inputs, dim=-1)
        return self.net(x_input)


class RectifiedFlow(nn.Module):
    """
    矫正流模型 - 最简实现
    仅包含Euler求解器和基础功能
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256,
                 time_dim: int = 64, condition_dim: int = 0,
                 sigma_min: float = 0.001, sigma_max: float = 1.0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # 组件
        self.time_embedding = TimeEmbedding(time_dim)
        self.velocity_net = VelocityNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            condition_dim=condition_dim
        )
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            x: 目标数据 (batch_size, latent_dim)
            condition: 条件 (batch_size, condition_dim)
        
        Returns:
            dict: 包含v_pred, v_true, x_t, t等
        """
        validate_tensor_shape(x, 2, "input x")
        
        batch_size, latent_dim = x.shape
        device = x.device
        
        # 1. 采样时间步 t ~ U(0,1)
        t = torch.rand(batch_size, device=device)
        
        # 2. 采样噪声
        noise = torch.randn_like(x)
        
        # 3. 线性插值: x_t = (1-t)*noise + t*x
        t_expanded = t.view(-1, 1)
        x_t = (1 - t_expanded) * noise + t_expanded * x
        
        # 4. 真实速度: v_true = x - noise
        v_true = x - noise
        
        # 5. 时间嵌入
        t_emb = self.time_embedding(t)
        
        # 6. 预测速度
        v_pred = self.velocity_net(x_t, t_emb, condition)
        
        return {
            'v_pred': v_pred,
            'v_true': v_true,
            'x_t': x_t,
            'noise': noise,
            't': t,
            't_emb': t_emb
        }
    
    def sample(self, batch_size: int, condition: Optional[torch.Tensor] = None,
               num_steps: int = 50, device: str = 'cpu') -> torch.Tensor:
        """
        采样生成新数据 - 仅Euler求解器
        
        Args:
            batch_size: 批量大小
            condition: 条件 (batch_size, condition_dim)
            num_steps: 采样步数
            device: 计算设备
        
        Returns:
            samples: 生成样本 (batch_size, latent_dim)
        """
        self.eval()
        
        # 从标准高斯噪声开始
        x = torch.randn(batch_size, self.latent_dim, device=device)
        
        # 时间步长
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((batch_size,), i * dt, device=device)
                t_emb = self.time_embedding(t)
                
                # 预测速度
                v = self.velocity_net(x, t_emb, condition)
                
                # Euler积分: x_{i+1} = x_i + dt * v_i
                x = x + dt * v
        
        return x
    
    def compute_loss(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            model_outputs: forward()的输出
        
        Returns:
            losses: 损失字典
        """
        v_pred = model_outputs['v_pred']
        v_true = model_outputs['v_true']
        
        # 基础流匹配损失
        flow_loss = simple_flow_loss(v_pred, v_true)
        
        # 简单的速度正则化
        velocity_reg = torch.mean(v_pred.pow(2)) * 0.001
        
        total_loss = flow_loss + velocity_reg
        
        return {
            'flow_loss': flow_loss,
            'velocity_reg': velocity_reg,
            'total_loss': total_loss
        }
    
    def encode_to_noise(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None,
                       num_steps: int = 50) -> torch.Tensor:
        """
        将数据编码到噪声空间 (反向过程)
        用于异常检测
        """
        self.eval()
        current = x.clone()
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((x.size(0),), 1 - i * dt, device=x.device)
                t_emb = self.time_embedding(t)
                
                v = self.velocity_net(current, t_emb, condition)
                current = current - dt * v  # 反向积分
        
        return current


# 测试代码
if __name__ == '__main__':
    print("🔬 测试基础RectifiedFlow模型")
    
    # 创建模型
    model = RectifiedFlow(latent_dim=512, hidden_dim=256, condition_dim=64)
    
    # 测试输入
    batch_size = 8
    x = torch.randn(batch_size, 512)
    condition = torch.randn(batch_size, 64)
    
    # 前向传播
    outputs = model(x, condition)
    print(f"✅ 前向传播成功，v_pred形状: {outputs['v_pred'].shape}")
    
    # 损失计算
    losses = model.compute_loss(outputs)
    print(f"✅ 损失计算成功，总损失: {losses['total_loss'].item():.6f}")
    
    # 采样测试
    samples = model.sample(batch_size=4, condition=condition[:4], num_steps=20, device='cpu')
    print(f"✅ 采样成功，样本形状: {samples.shape}")
    
    print("🎉 基础RectifiedFlow测试通过！")
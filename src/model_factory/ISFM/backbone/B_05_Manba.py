import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

class B_05_Manba(nn.Module):
    """
    解耦的Mamba核心模型，确保输入输出格式一致为[B, L, C]
    """
    def __init__(self, args):
        super(B_05_Manba, self).__init__()
        d_model = args.output_dim
        d_state=16 # by default
        expand=2
        d_conv=4
        dropout=0.1
        num_layers=2
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        
        # 多层Mamba残差块
        self.layers = nn.ModuleList([
            ResidualBlock(
                d_model=d_model,
                d_inner=self.d_inner,
                dt_rank=self.dt_rank,
                d_conv=d_conv,
                d_state=d_state,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, C] 格式的输入张量
        返回: [B, L, C] 格式的输出张量
        """
        # 输入形状直接通过Mamba块处理
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return x  # 输出保持与输入相同的形状 [B, L, C]


class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv, d_state, dropout):
        super(ResidualBlock, self).__init__()
        
        self.mixer = MambaBlock(d_model, d_inner, dt_rank, d_conv, d_state)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接
        output = self.mixer(self.norm(x)) + x
        return self.dropout(output)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv, d_state):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state

        # 投影到内部维度
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # 一维卷积
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner
        )

        # 输入特定投影
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        
        # delta投影
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # 状态空间参数
        A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: [B, L, d_model]
        返回: [B, L, d_model]
        """
        (b, l, _) = x.shape

        # 投影和分离
        x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 卷积处理
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # 状态空间模型处理
        y = self.ssm(x)
        y = y * F.silu(res)

        # 输出投影
        output = self.out_proj(y)  # [B, L, d_model]
        return output

    def ssm(self, x):
        """
        选择性状态空间处理
        x: [B, L, d_inner]
        返回: [B, L, d_inner]
        """
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # [d_in, n]
        D = self.D.float()  # [d_in]

        # 计算delta, B, C
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_in]
        
        # 选择性扫描
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        选择性扫描算法实现
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # 计算离散化参数
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        # 序列扫描
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        # 整合结果
        y = torch.stack(ys, dim=1)  # [B, L, d_in]
        y = y + u * D
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    


if __name__ == "__main__":
    def test_mamba_model():
        # 设置参数
        d_model = 128       # 模型维度
        batch_size = 8      # 批量大小
        seq_len = 64        # 序列长度
        num_layers = 2      # 层数
        
        # 创建模型实例
        model = B_05_Manba(
            d_model=d_model,
            d_state=16,
            expand=2,
            d_conv=4,
            dropout=0.1,
            num_layers=num_layers
        )
        
        # 创建随机输入张量 [B, L, C]
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 运行模型
        output = model(x)
        
        # 验证输出形状
        print(f"输入张量形状: {x.shape}")
        print(f"输出张量形状: {output.shape}")
        print(f"输入输出形状一致: {x.shape == output.shape}")
        
        return x, output
    test_mamba_model()
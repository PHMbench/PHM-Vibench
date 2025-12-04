# DPOT

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

ACTIVATION = {
    'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 
    'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.1), 
    'softplus': nn.Softplus(), 'ELU': nn.ELU(), 'silu': nn.SiLU()
}

class E_03_Patch(nn.Module):
    """
    将1D序列分块并进行线性嵌入的模块。

    - 可以直接使用显式参数初始化: E_03_Patch(seq_len=..., patch_len=..., in_chans=...)
    - 也可以从 VBench 模型配置命名空间初始化: E_03_Patch(args_m)
      （其中从 args_m.window_size / args_m.patch_size_L / args_m.input_dim 等读取信息）
    """

    def __init__(self, config_or_seq_len=1024, patch_len=16, in_chans=3,
                 embed_dim=768, out_dim=128, act='gelu'):
        super().__init__()

        # 兼容两种用法：
        # 1) E_03_Patch(args_m)  -> 从配置命名空间读取超参数
        # 2) E_03_Patch(seq_len=..., patch_len=..., in_chans=...)  -> 直接显式传参
        if hasattr(config_or_seq_len, "__dict__"):
            args_m = config_or_seq_len
            self.seq_len = getattr(args_m, "window_size", 1024)
            self.patch_len = getattr(args_m, "patch_size_L", 16)
            self.in_chans = getattr(args_m, "input_dim", 1)
            self.out_dim = getattr(args_m, "output_dim", 128)
            self.num_patches = self.seq_len // self.patch_len
            act_name = getattr(args_m, "activation", "gelu")
            embed_dim = getattr(args_m, "d_model", self.out_dim)
        else:
            self.seq_len = int(config_or_seq_len)
            self.patch_len = patch_len
            self.in_chans = in_chans
            self.out_dim = out_dim
            self.num_patches = self.seq_len // self.patch_len
            act_name = act

        self.act = ACTIVATION[act_name]

        # 使用1D卷积实现分块和嵌入
        self.proj = nn.Sequential(
            nn.Conv1d(self.in_chans, embed_dim, kernel_size=self.patch_len, stride=self.patch_len),
            self.act,
            nn.Conv1d(embed_dim, self.out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # 自动检测并转换维度格式
        # 支持两种常见格式:
        # 1. (B, C, L) - 标准Conv1d格式
        # 2. (B, L, C) - 数据加载器常见格式

        if len(x.shape) != 3:
            raise ValueError(f"E_03_Patch expects 3D input, got {len(x.shape)}D input with shape {x.shape}")

        B, dim1, dim2 = x.shape

        # 检测输入格式并转换
        if dim2 <= self.in_chans and dim1 > dim2:
            # 可能是 (B, L, C) 格式: 长度 > 通道数
            # 转换为 (B, C, L) 格式
            if dim2 != self.in_chans:
                raise ValueError(f"Channel dimension mismatch: expected {self.in_chans}, got {dim2}")
            x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
            # print(f"E_03_Patch: 自动转换维度 (B, L, C) -> (B, C, L): ({B}, {dim1}, {dim2}) -> {x.shape}")
        elif dim1 <= self.in_chans and dim2 > dim1:
            # 可能是 (B, C, L) 格式: 通道数 < 长度
            if dim1 != self.in_chans:
                raise ValueError(f"Channel dimension mismatch: expected {self.in_chans}, got {dim1}")
            # 已经是正确格式，无需转换
        else:
            raise ValueError(f"Cannot determine input format: {x.shape}. Expected (B, C, L) or (B, L, C)")

        B, C, L = x.shape

        # 验证维度匹配
        if C != self.in_chans:
            raise ValueError(f"Channel mismatch: model expects {self.in_chans}, got {C}")

        # 允许动态长度，只要满足 patch_len 要求即可；seq_len 主要用于 num_patches 估计
        # 如果 L 与 seq_len 不一致，这里不再强制报错，以兼容不同 window_size
        # proj 输出 x 形状: (B, out_dim, num_patches)
        x = self.proj(x)

        # 转换为 Transformer 期望的格式: (B, out_dim, num_patches) -> (B, num_patches, out_dim)
        x = x.transpose(1, 2)  # (B, D, L) -> (B, L, D)

        return x

if __name__ == "__main__":
    # 测试代码
    model = E_03_Patch(seq_len=4096, patch_len=16, in_chans=3, embed_dim=256, out_dim=128, act='gelu')
    x = torch.randn(2, 3, 1024)  # (B, C, L)
    out = model(x)
    print(out.shape)  # 预期输出形状: (2, 64, 128) - (B, num_patches, out_dim)
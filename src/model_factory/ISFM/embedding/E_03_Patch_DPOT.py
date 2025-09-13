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

class E_03_Patch_DPOT(nn.Module):
    """
    将1D序列分块并进行线性嵌入的模块。
    """
    def __init__(self, seq_len=1024, patch_len=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu'):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.in_chans = in_chans
        self.out_dim = out_dim
        self.num_patches = seq_len // patch_len
        self.act = ACTIVATION[act]

        # 使用1D卷积实现分块和嵌入
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_len, stride=patch_len),
            self.act,
            nn.Conv1d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # 输入 x 形状: (B, C, L)
        B, C, L = x.shape
        assert L == self.seq_len, f"Input sequence length ({L}) doesn't match model ({self.seq_len})."
        # 输出 x 形状: (B, out_dim, num_patches)
        x = self.proj(x)
        return x
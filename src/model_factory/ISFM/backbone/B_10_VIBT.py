import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, Attention

import torch.nn.functional as F


def modulate(x, scale, shift):

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class ViBTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        if c is None:
            c = x.mean(dim=1)  # 如果没有条件向量，则使用输入的均值作为条件
        # 进行 AdaLN 调制   
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaLN_modulation(c).chunk(6, dim=-1))
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), scale_msa, shift_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x

# --- 主要的 1D Backbone 模块 (仅重命名) ---
class B_10_VIBT(nn.Module): # vibration transformer backbone
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList([
            ViBTBlock(args.hidden_dim, args.num_heads, mlp_ratio = args.factor) for _ in range(args.num_layers)
        ])

    def forward(self, x, c):
        for block in self.blocks:
            x = block(x, c)
        return x

# --- Demo 代码 ---
if __name__ == '__main__':
    import argparse
    print("--- 1D Backbone Demo ---")
    # 模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--factor', type=float, default=4.0)
    args = parser.parse_args([])

    B = 4
    NUM_TOKENS = 2048 // 16 # L / patch_size

    # 创建 Backbone 模块实例
    backbone_1d = B_10_VIBT(args)

    # 创建模拟输入数据 (来自 1D Embedding 层的输出)
    x_tokens_in = torch.randn(B, NUM_TOKENS, args.hidden_dim)
    c_vector_in = None # torch.randn(B, args.hidden_dim)

    # 前向传播
    x_features = backbone_1d(x_tokens_in, c_vector_in)

    # 打印输出形状
    print(f"输入 Token 序列形状: {x_tokens_in.shape}")
    print(f"输出特征序列形状: {x_features.shape}")
    print("-" * 20)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import torch.nn.functional as F
import pandas as pd
# --------------------------------------------------------------------------
# SequencePatcher
# --------------------------------------------------------------------------
class SequencePatcher(nn.Module):
    def __init__(self, num_patches, patch_size):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

    def update_start_indices(self,x, L):
        """ 更新 start_indices 以适应新的序列长度 L """
        if L < self.patch_size:
            raise ValueError(f"序列长度 ({L}) 不能小于补丁大小 ({self.patch_size})")
        
        new_start_indices = torch.linspace(
            0, L - self.patch_size, steps=self.num_patches, device=x.device
        ).round().long()
        self.start_indices = new_start_indices
        self.L_original = L
        
        return new_start_indices
    def get_L_original(self):
        """ 获取原始序列长度 """
        if hasattr(self, 'L_original'):
            return self.L_original
        else:
            raise ValueError("L_original 尚未设置，请先调用 update_start_indices 方法。")
    def get_start_indices(self):
        """ 获取当前的 start_indices """
        if hasattr(self, 'start_indices'):
            return self.start_indices
        else:
            raise ValueError("start_indices 尚未设置，请先调用 update_start_indices 方法。")
    
    def patch(self, x):
        """ 输入 (B, C, L), 输出 (B, T, C, P) 和 start_indices (T,) """
        B, C, L = x.shape
        if L < self.patch_size:
            raise ValueError(f"序列长度 ({L}) 不能小于补丁大小 ({self.patch_size})")

        start_indices = self.update_start_indices(x, L)

        patch_indices = torch.arange(self.patch_size, device=x.device)
        absolute_indices = rearrange(start_indices, 't -> t 1') + rearrange(patch_indices, 'p -> 1 p')
        absolute_indices_for_gather = rearrange(absolute_indices, 't p -> 1 1 t p').expand(B, C, -1, -1)
        
        patches = torch.gather(x.unsqueeze(2).expand(-1, -1, self.num_patches, -1), 3, absolute_indices_for_gather)
        return rearrange(patches, 'b c t p -> b t c p')

    def unpatch(self, patches):
        """ 输入 (B, T, C, P), start_indices, L_original, 输出 (B, L, C) """
        B, T, C, P = patches.shape
        assert T == self.num_patches and P == self.patch_size
        
        L_original = self.get_L_original()
        start_indices = self.get_start_indices()

        output = torch.zeros(B, C, L_original, device=patches.device)
        overlap_count = torch.zeros(B, C, L_original, device=patches.device)
        
        patch_pos_indices = torch.arange(P, device=patches.device).unsqueeze(0)
        absolute_indices = start_indices.unsqueeze(1) + patch_pos_indices
        absolute_indices_expanded = rearrange(absolute_indices, 't p -> 1 1 t p').expand(B, C, -1, -1)
        patches_for_scatter = rearrange(patches, 'b t c p -> b c t p')
        
        output.scatter_add_(2, absolute_indices_expanded.flatten(2), patches_for_scatter.flatten(2))
        overlap_count.scatter_add_(2, absolute_indices_expanded.flatten(2), torch.ones_like(patches_for_scatter).flatten(2))
        
        reconstructed_ts = output / torch.clamp(overlap_count, min=1.0)
        return rearrange(reconstructed_ts, 'b c l -> b l c')

# --------------------------------------------------------------------------
# 模块一(B): 主要的 Embedding 模块
# --------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """ for flow loss """
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim),
                                 nn.SiLU(),
                                 nn.Linear(dim, dim))
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        # Latex : freqs = torch.exp(-\log(max\_period) \cdot \frac{\text{arange}(half)}{half})
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        embedding = torch.cat([torch.cos(t[:, None] * freqs),
                                torch.sin(t[:, None] * freqs)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding
    def forward(self, t):
        return self.mlp(self.timestep_embedding(t * 1000, self.mlp[0].in_features))

class LabelEmbedder(nn.Module):
    """ for conditional generation """
    """ Embeds class labels into a vector of specified dimension. """
    def __init__(self, num_classes, dim):
         super().__init__()
         self.embedding = nn.Embedding(num_classes + 1, dim)
    def forward(self, labels):
        return self.embedding(labels)

def get_1d_sincos_pos_embed(embed_dim, num_patches):
    pos = torch.arange(num_patches)
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.)
    omega = 1. / 10000**omega
    out = torch.einsum('m,d->md', pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()

class E_02_HSE_v2(nn.Module):
    def __init__(self, args):
        super().__init__()

        # num_patches, patch_size, num_channels, c_dim, dim, num_classes = args.num_patches, args.patch_size, args.num_channels, args.c_dim, args.hidden_dim, args.num_classes
        self.args = args

        self.use_cond =  None # TODO for conditional generalization args.num_classes is not
        self.use_interpolation = False # for generation
        # 核心修改: 使用统一的补丁配置
        self.patcher = SequencePatcher(args.num_patches, args.patch_size_L) # NOTE: 更新1

        # S1
        self.channel_embedders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(num_channels + 1, args.c_dim * 2), nn.GELU(), nn.Linear(args.c_dim * 2, args.c_dim))
            for name, num_channels in args.num_channels.items()
        }) # NOTE : 更新2
        # S2
        self.proj_patch = nn.Linear(args.patch_size_L * args.c_dim, args.hidden_dim)
        # S3 TODO ROPE
        self.pos_embed = nn.Parameter(get_1d_sincos_pos_embed(args.hidden_dim, args.num_patches).unsqueeze(0), requires_grad=False)

        if self.use_interpolation: # TODO
            self.t_embedder, self.r_embedder = TimestepEmbedder(args.hidden_dim), TimestepEmbedder(args.hidden_dim)
        if self.use_cond: # TODO y_embedder should be a module dict
            self.y_embedder = LabelEmbedder(args.num_classes, args.hidden_dim) 


    def update_channel_embedder(self, new_num_channels):
        """ 更新 channel_embedders 以适应新的 num_channels """
        for name, num_channels in new_num_channels.items():
            if name not in self.channel_embedders:
                self.channel_embedders[name] = nn.Sequential(
                    nn.Linear(num_channels + 1, self.args.c_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.args.c_dim * 2, self.args.c_dim)
                )
            else:
                # 更新现有的 channel embedder
                self.channel_embedders[name][0] = nn.Linear(num_channels + 1, self.channel_embedders[name][0].in_features)

    def get_grid_1d(self, x,sample_f):
        """生成1D坐标网格"""
        batchsize, seq_len, n_feats = x.shape
        grid = torch.arange(seq_len)  
        if isinstance(sample_f, (int, float)):
            grid = grid.reshape(1, seq_len, 1).repeat(batchsize, 1, 1) / sample_f
        elif isinstance(sample_f, pd.Series):
            sample_f_tensor = torch.tensor(sample_f.values, dtype=x.dtype).reshape(-1, 1, 1)
            grid = grid.reshape(1, seq_len, 1).repeat(batchsize, 1, 1) / sample_f_tensor
        return grid.to(x.device)
    
    def forward(self, x, system_id, sample_f, t = None, r = None, y=None):
    
        sample_T = self.get_grid_1d(x, sample_f)

        x_with_time = torch.cat([x, sample_T], dim=-1)
        
        patches = self.patcher.patch(rearrange(x_with_time, 'b l c -> b c l'))
        
        patches_for_mlp = rearrange(patches, 'b t c p -> b t p c')

        channel_embedded_patches = self.channel_embedders[str(system_id)](patches_for_mlp)

        flattened_patches = rearrange(channel_embedded_patches, 'b t p c_dim -> b t (p c_dim)')

        x_tokens = self.proj_patch(flattened_patches)

        x_tokens = x_tokens + self.pos_embed
        
        c = None

        if self.use_interpolation:
            t_emb, r_emb = self.t_embedder(t), self.r_embedder(r)

            c = t_emb + r_emb
        if self.use_cond and y is not None: 

            c = c + self.y_embedder(y)

        # 核心修改: 返回重建所需的 start_indices
        return x_tokens, c

if __name__ == "__main__":
    # --- 模块一: 最终版 Embedding 独立测试 ---
    print("--- 模块一: 统一补丁配置的 Embedding 独立测试 ---")

    # 1. 定义配置类
    class Config:
        def __init__(self):
            # 统一的补丁配置
            self.num_patches = 128
            self.patch_size_L = 16
            # 其他超参数
            self.c_dim = 16
            self.hidden_dim = 768
            self.num_channels = {'vibration': 3, 'temperature': 1}
            self.num_classes = 10 # 示例值

    # 2. 实例化配置
    args = Config()

    # 3. 实例化模块
    embedding_layer = E_02_HSE_v2(args)

    # --- 测试'vibration'信号 ---
    B, L_variable = 4, 3000
    x_vib = torch.randn(B, L_variable, 3)
    sample_f_in = 100.0 # 示例采样频率
    t_in, r_in = torch.rand(B), torch.rand(B)

    # 接收返回值
    x_tokens_vib, c_vib = embedding_layer(x_vib, 'vibration', sample_f_in, t_in, r_in)

    print(f"输入 vibration (3通道), L={L_variable}")
    print(f"  -> 输出 tokens shape: {x_tokens_vib.shape}")
    if c_vib is not None:
        print(f"  -> 输出 condition shape: {c_vib.shape}")
    assert x_tokens_vib.shape == (B, args.num_patches, args.hidden_dim)
    print("  'vibration'信号处理成功!")
    print("-" * 40)

    # --- 测试'temperature'信号 ---
    x_temp = torch.randn(B, L_variable, 1)
    # 接收返回值
    x_tokens_temp, c_temp = embedding_layer(x_temp, 'temperature', sample_f_in, t_in, r_in)
    print(f"输入 temperature (1通道), L={L_variable}")
    print(f"  -> 输出 tokens shape: {x_tokens_temp.shape}")
    if c_temp is not None:
        print(f"  -> 输出 condition shape: {c_temp.shape}")
    assert x_tokens_temp.shape == (B, args.num_patches, args.hidden_dim)
    print("  'temperature'信号处理成功!")
    print("-" * 40)
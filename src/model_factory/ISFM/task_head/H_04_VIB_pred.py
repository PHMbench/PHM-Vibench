
import torch
import torch.nn as nn
from einops import rearrange

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

class H_04_VIB_pred(nn.Module):
    def __init__(self, args, patcher):
        super().__init__()
        # 核心修改: 使用统一的补丁配置
        self.patcher = patcher
        self.num_channels = args.num_channels
        self.final_norm = RMSNorm(args.hidden_dim)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(args.hidden_dim, 2 * args.hidden_dim))
        self.proj_out = nn.Linear(args.hidden_dim, args.patch_size_L * args.c_dim)
        self.pred_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(args.c_dim, args.c_dim * 2), nn.GELU(), nn.Linear(args.c_dim * 2, num_channels + 1))
            for name, num_channels in self.num_channels.items()
        })
        
    def forward(self, x_tokens, system_id, c = None):

        # L_original = self.patcher.get_L_original()
        # start_indices = self.patcher.get_start_indices() 

        if c is None:
            c = x_tokens.mean(dim=1)

        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x_tokens), shift, scale)
        x = self.proj_out(x)
        
        patches_for_mlp = rearrange(x, 'b t (p c_dim) -> b t p c_dim', p=self.patcher.patch_size)
        reconstructed_channels = self.pred_heads[str(system_id)](patches_for_mlp)
        patches_to_unpatch = rearrange(reconstructed_channels, 'b t p c -> b t c p')
        
        # 核心修改: 使用传入的 start_indices 进行重建
        reconstructed_ts_with_time = self.patcher.unpatch(patches_to_unpatch)
        
        num_original_channels = self.num_channels[str(system_id)]
        reconstructed_ts = reconstructed_ts_with_time[:, :, :num_original_channels]
        
        return reconstructed_ts


# --- 模块三: 最终版 Task Head 独立测试 (使用 SequencePatcher) ---
if __name__ == '__main__':
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

        def unpatch(self, patches, start_indices=None, L_original=None):
            """ 
            输入 (B, T, C, P), 输出 (B, L, C).
            为了与 MfditTaskHead1d 兼容, 允许传入 start_indices 和 L_original.
            如果未传入, 则使用内部存储的状态.
            """
            B, T, C, P = patches.shape
            assert T == self.num_patches and P == self.patch_size
            
            # 优先使用传入的参数, 否则使用内部状态
            L_original_to_use = L_original if L_original is not None else self.get_L_original()
            start_indices_to_use = start_indices if start_indices is not None else self.get_start_indices()

            output = torch.zeros(B, C, L_original_to_use, device=patches.device)
            overlap_count = torch.zeros(B, C, L_original_to_use, device=patches.device)
            
            patch_pos_indices = torch.arange(P, device=patches.device).unsqueeze(0)
            absolute_indices = start_indices_to_use.unsqueeze(1) + patch_pos_indices
            absolute_indices_expanded = rearrange(absolute_indices, 't p -> 1 1 t p').expand(B, C, -1, -1)
            patches_for_scatter = rearrange(patches, 'b t c p -> b c t p')
            
            output.scatter_add_(2, absolute_indices_expanded.flatten(2), patches_for_scatter.flatten(2))
            overlap_count.scatter_add_(2, absolute_indices_expanded.flatten(2), torch.ones_like(patches_for_scatter).flatten(2))
            
            reconstructed_ts = output / torch.clamp(overlap_count, min=1.0)
            return rearrange(reconstructed_ts, 'b c l -> b l c')
    from argparse import Namespace
    print("--- 模块三: 最终版 Task Head 独立测试 (使用 SequencePatcher) ---")
    
    # --- 使用 args 管理参数 ---
    args = Namespace(
        dim=768,
        patch_size=16,
        c_dim=16,
        num_channels={'vibration': 3, 'temperature': 1},
        num_patches=128,
    )
    
    B, L_variable = 4, 3000


    # 实例化新的 Patcher
    sequence_patcher = SequencePatcher(num_patches=args.num_patches, patch_size=args.patch_size)
    sequence_patcher.update_start_indices(torch.randn(B, args.c_dim, L_variable), L_variable)

    # 实例化 Task Head, 传入新的 Patcher
    task_head = H_04_VIB_pred(args, sequence_patcher)

    # --- 测试重建 'vibration' (3通道) 信号 ---
    output_name = 'vibration'
    C_out = args.num_channels[output_name]
    print(f"请求重建 '{output_name}' 信号 ({C_out}个通道)")

    # 模拟输入
    features_in = torch.randn(B, args.num_patches, args.hidden_dim)
    c_vector_in = None
    # 模拟从Embedding层得到的重建索引
    start_indices_in = torch.linspace(0, L_variable - args.patch_size, steps=args.num_patches).round().long()

    # forward 方法中的 recon_heads 似乎是拼写错误，应为 pred_heads
    # 为了让测试通过，我们临时修正它
    task_head.recon_heads = task_head.pred_heads

    # 调用 forward 方法
    output_ts = task_head(features_in, output_name, c_vector_in)

    print(f"输入 features shape: {features_in.shape}")
    print(f"重建后 output shape: {output_ts.shape}")
    assert output_ts.shape == (B, L_variable, C_out)
    print(f"  '{output_name}' 信号重建成功!")
    print("-" * 40)


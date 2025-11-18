import torch.nn as nn

from .E_03_Patch import E_03_Patch


class E_03_Patch_DPOT(nn.Module):
    """
    Wrapper around E_03_Patch to integrate with the ISFM model factory.

    - 接受单一 args_m（VBench 模型配置命名空间）
    - 从 args_m 中读取 window_size/input_dim/patch_size_L/patch_size_C/num_patches/d_model/output_dim
    - 适配 data_factory 输出 (B, L, C) 与 E_03_Patch 期望的 (B, C, L)
    - 输出形状 (B, num_patches, out_dim)，与 Transformer (B, L, D) 约定一致
    """

    def __init__(self, args_m):
        super().__init__()

        seq_len = getattr(args_m, "window_size", 1024)
        patch_len = getattr(args_m, "patch_size_L", 16)
        in_chans = getattr(args_m, "input_dim", 1)

        embed_dim = getattr(args_m, "d_model", getattr(args_m, "output_dim", 128))
        out_dim = getattr(args_m, "output_dim", embed_dim)
        act = getattr(args_m, "activation", "gelu")

        self.out_dim = out_dim

        self._patch_embed = E_03_Patch(
            seq_len=seq_len,
            patch_len=patch_len,
            in_chans=in_chans,
            embed_dim=embed_dim,
            out_dim=out_dim,
            act=act,
        )

    def forward(self, x):
        # 输入 x: (B, L, C) -> 转换为 (B, C, L)
        x = x.permute(0, 2, 1)

        # E_03_Patch 输出: (B, out_dim, num_patches)
        x = self._patch_embed(x)

        # 转换为 (B, num_patches, out_dim)，供 backbone 使用
        x = x.permute(0, 2, 1)
        return x


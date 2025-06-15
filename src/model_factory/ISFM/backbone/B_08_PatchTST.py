# patchtst_backbone.py  ───────────────────────────────────────────
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# ────────────────────── 1. Tiny helpers ────────────────────────
class Transpose(nn.Module):
    """nn.Module wrapper around `tensor.transpose` so it can sit in nn.Sequential."""
    def __init__(self, dim0, dim1, contiguous=False):
        super().__init__()
        self.dim0, self.dim1, self.contig = dim0, dim1, contiguous
    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous() if self.contig else x


# ────────────────────── 2. Backbone module ─────────────────────
class PatchTSTBackbone(nn.Module):
    """
    Backbone of PatchTST (Patch-patch Transformer for multivariate TS)
    Output: encoded patch tokens  –  no heads, no de/normalisation.
    Paper: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, cfg,
                 patch_len: int = 16,
                 stride: int    = 8):
        """
        Args
        ----
        cfg        : argparse.Namespace or similar with keys
                     (d_model, n_heads, d_ff, e_layers, dropout, activation,
                      seq_len, enc_in)
        patch_len  : temporal length \(p\) of each patch
        stride     : stride \(s\) between patches
        """
        super().__init__()

        # ➀ Patch embedding
        #   x   ∈ ℝ^{B×C×T}  →  U ∈ ℝ^{B·C×N_p×d_model}
        #   where  N_p = \left\lfloor\frac{T-p}{s}\right\rfloor+1\;.
        self.patch_embed = PatchEmbedding(
            cfg.d_model, patch_len, stride, stride, cfg.dropout)

        # ➁ Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, cfg.factor, attention_dropout=cfg.dropout,
                            output_attention=False),
                        cfg.d_model, cfg.n_heads),
                    cfg.d_model, cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation
                )
                for _ in range(cfg.e_layers)
            ],
            # LayerNorm replaced by channel-wise BatchNorm1d (as in the paper)
            norm_layer=nn.Sequential(
                Transpose(1, 2),             # [B·C, d_model, N_p]
                nn.BatchNorm1d(cfg.d_model),
                Transpose(1, 2)
            )
        )

    # ───────────────── forward ─────────────────
    def forward(self, x):
        """
        Args
        ----
        x : Tensor[B, T, C]   (time, variables)
        Returns
        -------
        z : Tensor[B, C, N_p, d_model]
            encoded patch tokens per variable.
        """
        # Rearrange to [B, C, T] for patch extractor
        x = x.permute(0, 2, 1)

        # Patchify & embed  →  U  [B·C, N_p, d_model]
        U, n_vars = self.patch_embed(x)

        # Transformer encoder
        Z, _ = self.encoder(U)                        # [B·C, N_p, d_model]

        # Reshape back to per-variable tensor
        Z = Z.reshape(-1, n_vars, Z.size(1), Z.size(2))   # [B, C, N_p, d_model]

        # Optional: permute to [B, C, d_model, N_p] if downstream wants channels last
        return Z                                        # [B, C, N_p, d_model]
# Example usage:
if __name__ == "__main__":
# test_patchtst_backbone.py
    import torch
    from argparse import Namespace

    cfg = Namespace(
        seq_len     = 512,      # original series length T
        enc_in      = 4,        # number of variables C
        d_model     = 64,
        n_heads     = 8,
        d_ff        = 128,
        e_layers    = 2,
        dropout     = 0.1,
        activation  = 'gelu',
        factor      = 5         # used by FullAttention (chunk size)
    )

    model = PatchTSTBackbone(cfg, patch_len=16, stride=8)

    B = 6
    x = torch.randn(B, cfg.seq_len, cfg.enc_in)   # [B,T,C]
    z = model(x)

    print(f"input  shape : {x.shape}")
    print(f"output shape : {z.shape}")            # [B, C, N_p, d_model]

    # Expected patch count N_p
    N_p = (cfg.seq_len - 16) // 8 + 1
    assert z.shape == (B, cfg.enc_in, N_p, cfg.d_model)
    print("Shape check passed ✔")

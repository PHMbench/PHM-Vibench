# seqtst_backbone.py  ───────────────────────────────────────────
import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer


# ────────────────── 1. Helper (for channel-wise BatchNorm) ─────
class Transpose(nn.Module):
    """
    Just wraps tensor.transpose so it can sit inside nn.Sequential.
    """
    def __init__(self, dim0, dim1, contiguous=False):
        super().__init__()
        self.d0, self.d1, self.contig = dim0, dim1, contiguous
    def forward(self, x):
        x = x.transpose(self.d0, self.d1)
        return x.contiguous() if self.contig else x


# ────────────────── 2. Sequence-to-sequence backbone ───────────
class B_08_PatchTST(nn.Module):
    """
    A *minimal* Transformer encoder for multivariate TS.
    Keeps length:  (B,L,C) → (B,L,C).

    Paper inspiration: PatchTST (but without patch embedding).
    """

    def __init__(self, cfg):
        """
        Args
        ----
        cfg : argparse.Namespace   – must provide
              d_model (=C), n_heads, d_ff, e_layers,
              dropout, activation, factor
        """
        super().__init__()
        d_model = cfg.output_dim      # we assume d_model == C

        # ➀ Positional encoding (optional).   Here: none → fully learnable.
        #     You can plug a sin/cos PE or learnable PE if desired.

        # ➁ Transformer Encoder (stacked)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=cfg.factor,
                            attention_dropout=cfg.dropout,
                            output_attention=False),
                        d_model=d_model,
                        n_heads=cfg.num_heads),
                    d_model=d_model,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation
                )
                for _ in range(cfg.e_layers)
            ],
            # Channel-wise BatchNorm as in PatchTST
            norm_layer=nn.Sequential(
                Transpose(1, 2),           # (B,C,L)
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)            # back to (B,L,C)
            )
        )

    # ───────────────── forward ─────────────────
    def forward(self, x):
        """
        Args
        ----
        x : Tensor[B, L, C]

        Returns
        -------
        z : Tensor[B, L, C]   (same shape)
        """
        # Encoder expects (B,L,d_model) already; no embed needed.
        # z_l = \mathrm{Encoder}(x_l),    \forall l∈[1,L].
        z, _ = self.encoder(x)      # _ = attn list (ignored here)
        return z


# ──────────────── 3. Key formulae ─────────────
#
# Multi-head scaled dot-product attention
#   \mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)V
#
# Position-wise feed-forward
#   \mathrm{FFN}(h)=\sigma\!\bigl(h\,W_1+b_1\bigr)W_2+b_2
#
# The backbone output is simply
#   Z = \mathrm{Encoder}(X)\in\mathbb{R}^{B\times L\times C}.
                              # [B, C, N_p, d_model]
# Example usage:
if __name__ == "__main__":
# test_patchtst_backbone.py
    import torch
    from argparse import Namespace

    cfg = Namespace(
        d_model   = 4,          # must equal C
        n_heads   = 2,
        d_ff      = 32,
        e_layers  = 3,
        dropout   = 0.1,
        activation= 'gelu',
        factor    = 5
    )

    model = B_08_PatchTST(cfg)

    B = 6
    L = 640
    C = 4
    x = torch.randn(B, L, C)   # [B,T,C]
    z = model(x)

    print(f"input  shape : {x.shape}")
    print(f"output shape : {z.shape}")            # [B, C, N_p, d_model]

    assert z.shape == (B, L, C), "Output shape should match input shape [B, L, C]"
    print("Shape check passed ✔")

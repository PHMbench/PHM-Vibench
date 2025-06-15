# timesnet_backbone.py  ─────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ───────────────────── 1. Period discovery via FFT ─────────────
def FFT_for_Period(x: torch.Tensor, k: int = 2):
    """
    Estimate the dominant periods of a multivariate sequence by
    ranking the averaged amplitude spectrum.

    Args
    ----
    x : Tensor[B, T, C]   input sequence
    k : int               number of dominant periods to return

    Returns
    -------
    periods  : ndarray(k,)      integer period lengths
    weights  : Tensor[B, k]     batch-wise amplitude averages
    """
    # Discrete Fourier transform along the temporal dimension
    xf = torch.fft.rfft(x, dim=1)                 # X_f ∈ ℂ^{B × (T/2+1) × C}

    # Amplitude spectrum averaged across batch & channels
    #   \hat{A}(f) = \frac{1}{B·C} \sum_{b,c} |X_f(b,f,c)|
    amp = xf.abs().mean(0).mean(-1)               # ℝ^{T/2+1}

    amp[0] = 0                                    # Remove DC component
    _, idx = torch.topk(amp, k)                   # Indices of k largest peaks

    # Period p_i = T / f_i (integer division)
    periods = (x.size(1) // idx.cpu().numpy())    # ℕ^{k}

    # Batch-wise weights (mean amplitude at each peak)
    #   w_b,i = \frac{1}{C} \sum_c |X_f(b, f_i, c)|
    weights = xf.abs().mean(-1)[:, idx]           # ℝ^{B×k}
    return periods, weights


# ─────────────── 2. Core temporal–spatial convolution block ────
from layers.Conv_Blocks import Inception_Block_V1  # your existing impl.

class TimesBlock(nn.Module):
    """
    One TimesNet block:
      1. Rearrange sequence along each dominant period
      2. 2-D convolution (Inception style)
      3. Adaptive weighted sum across periods (softmax)
      4. Residual connection
    """
    def __init__(self, cfg):
        super().__init__()
        self.seq_len, self.pred_len, self.k = cfg.seq_len, cfg.pred_len, cfg.top_k

        # \mathrm{Conv}(C_{in}=d_{model}) → GELU → \mathrm{Conv}(C_{out}=d_{model})
        self.conv = nn.Sequential(
            Inception_Block_V1(cfg.d_model, cfg.d_ff,  num_kernels=cfg.num_kernels),
            nn.GELU(),
            Inception_Block_V1(cfg.d_ff,   cfg.d_model, num_kernels=cfg.num_kernels)
        )

    def forward(self, x):                         # x ∈ ℝ^{B×T×C}
        B, T, C = x.size()
        periods, weights = FFT_for_Period(x, self.k)      # periods: ℕ^{k}

        outputs = []
        for p in periods:                                  # Iterate over k periods
            L = T if T % p == 0 else (T // p + 1) * p      # Pad to multiple of p
            x_pad = F.pad(x, (0, 0, 0, L - T))             # x_pad ∈ ℝ^{B×L×C}

            # Reshape to 2-D grid: (batch, channel, rows=L/p, cols=p)
            x_2d = x_pad.view(B, L // p, p, C).permute(0, 3, 1, 2)

            y = self.conv(x_2d)                            # 2-D conv

            # Inverse reshape back to (B, L, C) and truncate padding
            y = y.permute(0, 2, 3, 1).reshape(B, L, C)[:, :T]
            outputs.append(y)

        # Stack along new period dimension → ℝ^{B×T×C×k}
        y_stack = torch.stack(outputs, dim=-1)

        # Softmax weights:  α_{b,i} =  \frac{e^{w_{b,i}}}{\sum_j e^{w_{b,j}}}
        α = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)  # ℝ^{B×1×1×k}

        # Weighted sum across periods + residual
        y = (y_stack * α).sum(-1) + x                        # ℝ^{B×T×C}
        return y


# ───────────────────── 3. Shallow TimesNet backbone ────────────
class TimesNetBackbone(nn.Module):
    """
    Minimal depth-stacked TimesNet.
    Output length == input length; additional heads can be added on top.
    """
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([TimesBlock(cfg) for _ in range(cfg.e_layers)])

    def forward(self, x):                                   # x ∈ ℝ^{B×T×C}
        for blk in self.blocks:
            x = blk(x)
        return x                                            # ℝ^{B×T×C}
# Example usage:
if __name__ == "__main__":
    # Example configuration
    class Config:
        seq_len = 128
        pred_len = 32
        d_model = 64
        d_ff = 128
        num_kernels = 4
        e_layers = 3
        top_k = 2

    cfg = Config()
    model = TimesNetBackbone(cfg)

    # Example input tensor (batch_size=2, seq_len=128, channels=64)
    x = torch.randn(2, cfg.seq_len, cfg.d_model)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)  # Should be (2, 128, 64)
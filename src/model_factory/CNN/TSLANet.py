"""Native TSLANet supervised classifier (Emadeldeen Eldele, MIT).

Ported from Classification/TSLANet_classification.py. ASB, ICB and their flags
belong to each instance, never a global argparse object. No timm/Lightning runner.
See ../NATIVE_MODELS.md for the intentionally limited classification scope.
"""

import torch
from torch import nn

from .._native_classification import check_signal, class_count, positive_int, probability


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        mask = x.new_empty((x.shape[0], 1, 1)).bernoulli_(keep)
        return x * mask / keep


class InteractiveConvolutionBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim * 3, 1)
        self.conv2 = nn.Conv1d(dim, dim * 3, 3, padding=1)
        self.conv3 = nn.Conv1d(dim * 3, dim, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        first, second = self.conv1(x), self.conv2(x)
        mixed = first * self.dropout(self.act(second)) + second * self.dropout(self.act(first))
        return self.conv3(mixed).transpose(1, 2)


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, dim, adaptive_filter):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight = nn.Parameter(torch.empty(dim, 2))
        self.complex_weight_high = nn.Parameter(torch.empty(dim, 2))
        nn.init.trunc_normal_(self.complex_weight, std=0.02)
        nn.init.trunc_normal_(self.complex_weight_high, std=0.02)
        self.threshold = nn.Parameter(torch.rand(1))

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=1, norm="ortho")
        result = spectrum * torch.view_as_complex(self.complex_weight)
        if self.adaptive_filter:
            energy = spectrum.abs().square().sum(-1)
            relative_energy = energy / (energy.median(1, keepdim=True).values + 1e-6)
            # Retain the original straight-through threshold estimator.
            mask = ((relative_energy > self.threshold).float() - self.threshold).detach() + self.threshold
            result = result + spectrum * mask.unsqueeze(-1) * torch.view_as_complex(self.complex_weight_high)
        return torch.fft.irfft(result, n=x.shape[1], dim=1, norm="ortho")


class TSLABlock(nn.Module):
    def __init__(self, dim, dropout, drop_path, adaptive_filter, apply_asb, apply_icb):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter)
        self.icb = InteractiveConvolutionBlock(dim, dropout)
        self.drop_path = DropPath(drop_path)
        self.apply_asb, self.apply_icb = apply_asb, apply_icb

    def forward(self, x):
        if self.apply_asb and self.apply_icb:
            residual = self.icb(self.norm2(self.asb(self.norm1(x))))
        elif self.apply_icb:
            residual = self.icb(self.norm2(x))
        else:
            residual = self.asb(self.norm1(x))
        return x + self.drop_path(residual)


class Model(nn.Module):
    """BLC classification; TSLANet's half-patch stride remains explicit in scope."""

    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.input_dim = positive_int(args, "input_dim")
        patch = positive_int(args, "patch_size")
        dim = positive_int(args, "d_model")
        depth = positive_int(args, "e_layers")
        dropout = probability(args, "dropout")
        for key in ("adaptive_filter", "apply_asb", "apply_icb"):
            if not isinstance(getattr(args, key), bool):
                raise TypeError(f"model.{key} must be a YAML boolean")
        if not args.apply_asb and not args.apply_icb:
            raise ValueError("TSLANet requires ASB or ICB; refusing an identity-only backbone")
        if patch < 2 or patch > self.seq_len:
            raise ValueError("model.patch_size must be between 2 and model.seq_len")
        stride = patch // 2
        if (self.seq_len - patch) % stride:
            raise ValueError("TSLANet patches must cover the full sequence; no silent tail truncation")
        patches = (self.seq_len - patch) // stride + 1
        self.patch_embedding = nn.Conv1d(self.input_dim, dim, patch, stride=stride)
        self.position_embedding = nn.Parameter(torch.empty(1, patches, dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)
        probabilities = torch.linspace(0, dropout, depth).tolist()
        self.blocks = nn.ModuleList([
            TSLABlock(dim, dropout, probabilities[i], args.adaptive_filter, args.apply_asb, args.apply_icb)
            for i in range(depth)
        ])
        self.head = nn.Linear(dim, class_count(args))
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_signal(x, self.seq_len, self.input_dim, task_id)
        x = self.patch_embedding(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x + self.position_embedding)
        for block in self.blocks:
            x = block(x)
        features = x.mean(1)
        logits = self.head(features)
        return (logits, features) if return_feature else logits

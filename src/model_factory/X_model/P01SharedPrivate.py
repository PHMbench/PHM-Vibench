"""Shared-private 1D/2D model for paper P01.

The second view is a deterministic log-magnitude STFT of the same raw signal;
it is not represented as an independent sensor modality.  The model keeps the
standard PHM-Vibench ``logits = model(x, data_id, task_id)`` contract and makes
its unweighted representation losses available through
``get_auxiliary_losses`` for the task wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UXFD.signal_processing_2d import STFTTimeFrequency
from .UXFD.signal_processing_2d.stft_tfr import STFTConfig


def _get_attr(obj: Any, dotted: str, default: Any) -> Any:
    current = obj
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _SignalEncoder1D(nn.Module):
    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv1d(hidden, dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_count(dim), dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.permute(0, 2, 1).contiguous())


class _TimeFrequencyEncoder2D(nn.Module):
    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(dim), dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _projection(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


def _cross_covariance_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[0] < 2:
        return (a.sum() + b.sum()) * 0.0
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    a_centered = a_norm - a_norm.mean(dim=0, keepdim=True)
    b_centered = b_norm - b_norm.mean(dim=0, keepdim=True)
    covariance = a_centered.transpose(0, 1) @ b_centered / float(a.shape[0] - 1)
    return covariance.square().mean()


def _variance_floor_penalty(z: torch.Tensor, floor: float) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.sum() * 0.0
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1.0e-4)
    return F.relu(floor - std).mean()


class Model(nn.Module):
    """Factorize paired raw and deterministic time-frequency views.

    Input has shape ``(batch, length, channels)`` and output has shape
    ``(batch, num_classes)``.  Four unweighted auxiliary terms are produced on
    every forward pass: ``alignment``, ``private_independence``,
    ``reconstruction``, and ``shared_variance``.
    """

    full_auxiliary_loss_names = (
        "alignment",
        "private_independence",
        "reconstruction",
        "shared_variance",
    )

    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        del metadata

        num_classes = getattr(args, "num_classes", None)
        if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
            raise ValueError("P01SharedPrivate requires integer model.num_classes")
        if int(num_classes) < 2:
            raise ValueError("P01SharedPrivate requires at least two classes")

        self.in_channels = int(getattr(args, "in_channels", 1))
        self.encoder_dim = int(getattr(args, "encoder_dim", 64))
        self.latent_dim = int(getattr(args, "latent_dim", 32))
        self.variance_floor = float(_get_attr(args, "objective.variance_floor", 0.1))
        self.private_branch_enabled = bool(
            _get_attr(args, "ablation.private_branch_enabled", True)
        )
        self.shared_only_head_hidden = int(
            _get_attr(args, "ablation.shared_only_head_hidden", 2 * self.latent_dim)
        )
        dropout = float(getattr(args, "dropout", 0.1))
        pairing_mode = str(_get_attr(args, "pairing.mode", "paired"))

        if (
            self.in_channels < 1
            or self.encoder_dim < 8
            or self.latent_dim < 2
            or self.shared_only_head_hidden < 2
        ):
            raise ValueError("P01SharedPrivate dimensions must be positive and non-degenerate")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if not 0.0 <= self.variance_floor <= 1.0:
            raise ValueError("model.objective.variance_floor must be in [0, 1]")
        if pairing_mode != "paired":
            raise ValueError(
                "P01SharedPrivate accepts only pairing.mode=paired; negative controls "
                "must use a frozen dataset-level permutation manifest"
            )

        n_fft = int(_get_attr(args, "time_frequency.n_fft", 128))
        hop_length = int(_get_attr(args, "time_frequency.hop_length", 32))
        if n_fft < 8 or hop_length < 1 or hop_length > n_fft:
            raise ValueError("Invalid STFT configuration")

        self.time_frequency = STFTTimeFrequency(
            STFTConfig(
                n_fft=n_fft,
                hop_length=hop_length,
                center=bool(_get_attr(args, "time_frequency.center", True)),
                normalized=bool(_get_attr(args, "time_frequency.normalized", False)),
                magnitude=True,
            )
        )
        self.encoder_1d = _SignalEncoder1D(self.in_channels, self.encoder_dim)
        self.encoder_2d = _TimeFrequencyEncoder2D(self.in_channels, self.encoder_dim)

        self.shared_1d = _projection(self.encoder_dim, self.latent_dim)
        self.shared_2d = _projection(self.encoder_dim, self.latent_dim)
        if self.private_branch_enabled:
            self.private_1d: nn.Module | None = _projection(
                self.encoder_dim, self.latent_dim
            )
            self.private_2d: nn.Module | None = _projection(
                self.encoder_dim, self.latent_dim
            )
            self.reconstructor_1d: nn.Module | None = nn.Sequential(
                nn.Linear(2 * self.latent_dim, self.encoder_dim),
                nn.GELU(),
                nn.Linear(self.encoder_dim, self.encoder_dim),
            )
            self.reconstructor_2d: nn.Module | None = nn.Sequential(
                nn.Linear(2 * self.latent_dim, self.encoder_dim),
                nn.GELU(),
                nn.Linear(self.encoder_dim, self.encoder_dim),
            )
            fused_dim = 3 * self.latent_dim
            fusion_hidden = 2 * self.latent_dim
            self.auxiliary_loss_names = self.full_auxiliary_loss_names
        else:
            self.private_1d = None
            self.private_2d = None
            self.reconstructor_1d = None
            self.reconstructor_2d = None
            fused_dim = self.latent_dim
            fusion_hidden = self.shared_only_head_hidden
            self.auxiliary_loss_names = ("alignment", "shared_variance")

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, int(num_classes))

        self._last_auxiliary_losses: Dict[str, torch.Tensor] | None = None
        self._last_representation_state: Dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, data_id: Any = None, task_id: Any = None) -> torch.Tensor:
        return self._forward_views(x, x, data_id=data_id, task_id=task_id)

    def forward_paired_views(
        self,
        x_1d: torch.Tensor,
        x_2d_source: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        """Use an explicitly supplied source window for the deterministic 2D view."""
        return self._forward_views(x_1d, x_2d_source, data_id=data_id, task_id=task_id)

    def _forward_views(
        self,
        x: torch.Tensor,
        x_2d_source: torch.Tensor,
        *,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        del data_id, task_id
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L,C), got {tuple(x.shape)}")
        if x_2d_source.ndim != 3:
            raise ValueError(
                f"Expected 2D-view source shape (B,L,C), got {tuple(x_2d_source.shape)}"
            )
        if x.shape != x_2d_source.shape:
            raise ValueError(
                f"Paired view sources must have matching shapes, got {tuple(x.shape)} and "
                f"{tuple(x_2d_source.shape)}"
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Configured in_channels={self.in_channels}, received C={x.shape[-1]}"
            )

        encoded_1d = self.encoder_1d(x)
        view_2d = torch.log1p(self.time_frequency(x_2d_source))
        view_2d = view_2d.permute(0, 3, 2, 1).contiguous()
        encoded_2d = self.encoder_2d(view_2d)

        shared_1d = self.shared_1d(encoded_1d)
        shared_2d = self.shared_2d(encoded_2d)
        shared_mean = 0.5 * (shared_1d + shared_2d)
        alignment = 1.0 - F.cosine_similarity(shared_1d, shared_2d, dim=1).mean()
        shared_variance = 0.5 * (
            _variance_floor_penalty(shared_1d, self.variance_floor)
            + _variance_floor_penalty(shared_2d, self.variance_floor)
        )
        self._last_representation_state = {
            "encoded_1d": encoded_1d,
            "encoded_2d": encoded_2d,
            "shared_1d": shared_1d,
            "shared_2d": shared_2d,
        }
        if self.private_branch_enabled:
            assert self.private_1d is not None and self.private_2d is not None
            assert self.reconstructor_1d is not None and self.reconstructor_2d is not None
            private_1d = self.private_1d(encoded_1d)
            private_2d = self.private_2d(encoded_2d)
            reconstructed_1d = self.reconstructor_1d(
                torch.cat([shared_1d, private_1d], dim=1)
            )
            reconstructed_2d = self.reconstructor_2d(
                torch.cat([shared_2d, private_2d], dim=1)
            )
            private_independence = (
                _cross_covariance_penalty(private_1d, private_2d)
                + _cross_covariance_penalty(shared_1d, private_1d)
                + _cross_covariance_penalty(shared_2d, private_2d)
            ) / 3.0
            reconstruction = 0.5 * (
                F.mse_loss(reconstructed_1d, encoded_1d.detach())
                + F.mse_loss(reconstructed_2d, encoded_2d.detach())
            )
            fused_input = torch.cat([shared_mean, private_1d, private_2d], dim=1)
            self._last_auxiliary_losses = {
                "alignment": alignment,
                "private_independence": private_independence,
                "reconstruction": reconstruction,
                "shared_variance": shared_variance,
            }
            self._last_representation_state.update(
                {
                    "private_1d": private_1d,
                    "private_2d": private_2d,
                    "reconstructed_1d": reconstructed_1d,
                    "reconstructed_2d": reconstructed_2d,
                }
            )
        else:
            fused_input = shared_mean
            self._last_auxiliary_losses = {
                "alignment": alignment,
                "shared_variance": shared_variance,
            }

        fused = self.fusion(fused_input)
        logits = self.classifier(fused)
        return logits

    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        if self._last_auxiliary_losses is None:
            raise RuntimeError("get_auxiliary_losses requires a preceding forward pass")
        return dict(self._last_auxiliary_losses)

    def get_representation_state(self, *, detach: bool = True) -> Dict[str, torch.Tensor]:
        if self._last_representation_state is None:
            raise RuntimeError("get_representation_state requires a preceding forward pass")
        if detach:
            return {name: value.detach() for name, value in self._last_representation_state.items()}
        return dict(self._last_representation_state)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

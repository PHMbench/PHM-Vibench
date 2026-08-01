"""Parameter-auditable paired-view baselines for paper P01."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .P01SharedPrivate import (
    _SignalEncoder1D,
    _TimeFrequencyEncoder2D,
    _get_attr,
    _projection,
)
from .UXFD.signal_processing_2d import STFTTimeFrequency
from .UXFD.signal_processing_2d.stft_tfr import STFTConfig


VARIANTS = {"one_d", "two_d", "concat", "generic_attention", "contrastive"}


class Model(nn.Module):
    """Build exactly one registered comparison architecture.

    All paired-view variants reuse the same encoder definitions as P01-M1.
    Widths are explicit configuration fields so trainable parameter counts can
    be matched without dormant parameters.
    """

    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        del metadata
        num_classes = getattr(args, "num_classes", None)
        if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
            raise ValueError("P01Baselines requires integer model.num_classes")
        if int(num_classes) < 2:
            raise ValueError("P01Baselines requires at least two classes")

        self.variant = str(getattr(args, "variant", "generic_attention"))
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown P01 baseline variant: {self.variant}")
        self.in_channels = int(getattr(args, "in_channels", 2))
        self.encoder_dim = int(getattr(args, "encoder_dim", 64))
        self.head_hidden = int(getattr(args, "head_hidden", 128))
        self.projection_dim = int(getattr(args, "projection_dim", 32))
        self.contrastive_temperature = float(
            getattr(args, "contrastive_temperature", 0.1)
        )
        dropout = float(getattr(args, "dropout", 0.1))
        if self.in_channels < 1 or self.encoder_dim < 8 or self.head_hidden < 2:
            raise ValueError("P01 baseline dimensions must be positive and non-degenerate")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("model.contrastive_temperature must be positive")

        uses_1d = self.variant != "two_d"
        uses_2d = self.variant != "one_d"
        if uses_1d:
            self.encoder_1d: nn.Module | None = _SignalEncoder1D(
                self.in_channels, self.encoder_dim
            )
        else:
            self.encoder_1d = None

        if uses_2d:
            n_fft = int(_get_attr(args, "time_frequency.n_fft", 128))
            hop_length = int(_get_attr(args, "time_frequency.hop_length", 32))
            if n_fft < 8 or hop_length < 1 or hop_length > n_fft:
                raise ValueError("Invalid STFT configuration")
            self.time_frequency: nn.Module | None = STFTTimeFrequency(
                STFTConfig(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    center=bool(_get_attr(args, "time_frequency.center", True)),
                    normalized=bool(_get_attr(args, "time_frequency.normalized", False)),
                    magnitude=True,
                )
            )
            self.encoder_2d: nn.Module | None = _TimeFrequencyEncoder2D(
                self.in_channels, self.encoder_dim
            )
        else:
            self.time_frequency = None
            self.encoder_2d = None

        if self.variant == "generic_attention":
            attention_heads = int(getattr(args, "attention_heads", 4))
            if self.encoder_dim % attention_heads != 0:
                raise ValueError("model.encoder_dim must be divisible by attention_heads")
            self.attention: nn.Module | None = nn.MultiheadAttention(
                self.encoder_dim,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            head_input = self.encoder_dim
        else:
            self.attention = None
            head_input = self.encoder_dim if self.variant in {"one_d", "two_d"} else 2 * self.encoder_dim

        if self.variant == "contrastive":
            self.project_1d: nn.Module | None = _projection(
                self.encoder_dim, self.projection_dim
            )
            self.project_2d: nn.Module | None = _projection(
                self.encoder_dim, self.projection_dim
            )
        else:
            self.project_1d = None
            self.project_2d = None

        self.head = nn.Sequential(
            nn.Linear(head_input, self.head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_hidden, int(num_classes)),
        )
        self._last_auxiliary_losses: Dict[str, torch.Tensor] = {}
        self._last_representation_state: Dict[str, torch.Tensor] | None = None

    def _validate(self, x_1d: torch.Tensor, x_2d_source: torch.Tensor) -> None:
        if x_1d.ndim != 3 or x_2d_source.ndim != 3:
            raise ValueError("P01 baselines expect both sources as (B,L,C)")
        if x_1d.shape != x_2d_source.shape:
            raise ValueError("P01 baseline paired sources must have matching shapes")
        if x_1d.shape[-1] != self.in_channels:
            raise ValueError(
                f"Configured in_channels={self.in_channels}, received C={x_1d.shape[-1]}"
            )

    def forward(self, x: torch.Tensor, data_id: Any = None, task_id: Any = None) -> torch.Tensor:
        return self._forward_views(x, x, data_id=data_id, task_id=task_id)

    def forward_paired_views(
        self,
        x_1d: torch.Tensor,
        x_2d_source: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        return self._forward_views(x_1d, x_2d_source, data_id=data_id, task_id=task_id)

    def _forward_views(
        self,
        x_1d_source: torch.Tensor,
        x_2d_source: torch.Tensor,
        *,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        del data_id, task_id
        self._validate(x_1d_source, x_2d_source)

        encoded_1d = (
            self.encoder_1d(x_1d_source) if self.encoder_1d is not None else None
        )
        if self.encoder_2d is not None and self.time_frequency is not None:
            view_2d = torch.log1p(self.time_frequency(x_2d_source))
            view_2d = view_2d.permute(0, 3, 2, 1).contiguous()
            encoded_2d = self.encoder_2d(view_2d)
        else:
            encoded_2d = None

        self._last_auxiliary_losses = {}
        if self.variant == "one_d":
            assert encoded_1d is not None
            fused = encoded_1d
        elif self.variant == "two_d":
            assert encoded_2d is not None
            fused = encoded_2d
        elif self.variant == "generic_attention":
            assert encoded_1d is not None and encoded_2d is not None and self.attention is not None
            tokens = torch.stack([encoded_1d, encoded_2d], dim=1)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            fused = attended.mean(dim=1)
        else:
            assert encoded_1d is not None and encoded_2d is not None
            fused = torch.cat([encoded_1d, encoded_2d], dim=1)
            if self.variant == "contrastive":
                assert self.project_1d is not None and self.project_2d is not None
                projected_1d = F.normalize(self.project_1d(encoded_1d), dim=1)
                projected_2d = F.normalize(self.project_2d(encoded_2d), dim=1)
                if projected_1d.shape[0] < 2:
                    contrastive_alignment = (projected_1d.sum() + projected_2d.sum()) * 0.0
                else:
                    similarity = (
                        projected_1d @ projected_2d.transpose(0, 1)
                    ) / self.contrastive_temperature
                    targets = torch.arange(similarity.shape[0], device=similarity.device)
                    contrastive_alignment = 0.5 * (
                        F.cross_entropy(similarity, targets)
                        + F.cross_entropy(similarity.transpose(0, 1), targets)
                    )
                self._last_auxiliary_losses = {
                    "contrastive_alignment": contrastive_alignment
                }

        logits = self.head(fused)
        state: Dict[str, torch.Tensor] = {"fused": fused}
        if encoded_1d is not None:
            state["encoded_1d"] = encoded_1d
        if encoded_2d is not None:
            state["encoded_2d"] = encoded_2d
        self._last_representation_state = state
        return logits

    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
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

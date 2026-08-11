"""Forward-only M1--M5 conditions for the P01 alignment study.

This module establishes the maintained representations and frozen rendering
operator only.  The physical, semantic, and geometric training objective is
deliberately deferred to the executable-semantic gate (G02).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


CONDITIONS = ("M1", "M2", "M3", "M4", "M5")
_MISSING = object()


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


def _required(obj: Any, dotted: str) -> Any:
    value = _get_attr(obj, dotted, _MISSING)
    if value is _MISSING:
        raise ValueError(f"model.{dotted} is required")
    return value


def _required_int(obj: Any, dotted: str) -> int:
    value = _required(obj, dotted)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"model.{dotted} must be an integer")
    return int(value)


def _required_float(obj: Any, dotted: str) -> float:
    value = _required(obj, dotted)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"model.{dotted} must be numeric")
    return float(value)


def _required_bool(obj: Any, dotted: str) -> bool:
    value = _required(obj, dotted)
    if not isinstance(value, bool):
        raise ValueError(f"model.{dotted} must be boolean")
    return value


def _required_str(obj: Any, dotted: str) -> str:
    value = _required(obj, dotted)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model.{dotted} must be a non-empty string")
    return value


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _SignalEncoder1D(nn.Module):
    """The narrow P01 1D encoder retained from the governed G040 branch."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(16, output_dim // 2)
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv1d(hidden, output_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_count(output_dim), output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.network(waveform.permute(0, 2, 1).contiguous())


class _TimeFrequencyEncoder2D(nn.Module):
    """The narrow P01 2D encoder retained from the governed G040 branch."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(16, output_dim // 2)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(output_dim), output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, time_frequency: torch.Tensor) -> torch.Tensor:
        return self.network(time_frequency)


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.GELU(),
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


@dataclass(frozen=True)
class RendererConfig:
    """Complete identity of the single deterministic rendering operator."""

    n_fft: int
    hop_length: int
    win_length: int
    window: str
    window_periodic: bool
    center: bool
    pad_mode: str
    normalized: bool
    onesided: bool
    representation: str
    scaling: str
    resize: str
    normalization: str


class DeterministicTimeFrequencyRenderer(nn.Module):
    """Render ``(B,L,C)`` waveforms as frozen log-magnitude Hann STFT views."""

    def __init__(self, config: RendererConfig) -> None:
        super().__init__()
        if config.n_fft < 8:
            raise ValueError("renderer.n_fft must be at least 8")
        if not 1 <= config.hop_length <= config.win_length <= config.n_fft:
            raise ValueError(
                "renderer requires 1 <= hop_length <= win_length <= n_fft"
            )
        fixed_choices = {
            "window": (config.window, "hann"),
            "pad_mode": (config.pad_mode, "reflect"),
            "representation": (config.representation, "magnitude"),
            "scaling": (config.scaling, "log1p"),
            "resize": (config.resize, "none"),
            "normalization": (config.normalization, "none"),
        }
        mismatches = [
            f"{name}={observed!r} (required {expected!r})"
            for name, (observed, expected) in fixed_choices.items()
            if observed != expected
        ]
        if mismatches:
            raise ValueError("Unsupported frozen renderer choice: " + "; ".join(mismatches))
        if not config.window_periodic:
            raise ValueError("renderer.window_periodic must be true")
        if not config.onesided:
            raise ValueError("renderer.onesided must be true")
        self.config = config

    def identity(self) -> Dict[str, Any]:
        return asdict(self.config)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(
                f"Renderer expects waveform shape (B,L,C), got {tuple(waveform.shape)}"
            )
        batch, length, channels = waveform.shape
        if length < self.config.n_fft:
            raise ValueError(
                f"Waveform length {length} is shorter than n_fft={self.config.n_fft}"
            )
        flattened = waveform.permute(0, 2, 1).contiguous().view(
            batch * channels, length
        )
        window = torch.hann_window(
            self.config.win_length,
            periodic=self.config.window_periodic,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        spectrum = torch.stft(
            flattened,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            normalized=self.config.normalized,
            onesided=self.config.onesided,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        rendered = torch.log1p(magnitude)
        return rendered.view(
            batch, channels, rendered.shape[-2], rendered.shape[-1]
        )


class Model(nn.Module):
    """Instantiate exactly one frozen P01 condition.

    M3 and M5 intentionally share the same concatenation forward architecture.
    G02 may distinguish M5 only through objective consumption; G01 must not make
    a performance or alignment claim.
    """

    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        del metadata
        num_classes = getattr(args, "num_classes", None)
        if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
            raise ValueError("P01Alignment requires integer model.num_classes")
        if int(num_classes) < 2:
            raise ValueError("P01Alignment requires at least two classes")

        self.condition = _required_str(args, "condition")
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"Unknown P01 condition {self.condition!r}; expected one of {CONDITIONS}"
            )
        self.in_channels = _required_int(args, "in_channels")
        self.encoder_dim = _required_int(args, "encoder_dim")
        self.latent_dim = _required_int(args, "latent_dim")
        self.head_hidden = _required_int(args, "head_hidden")
        attention_heads = _required_int(args, "attention_heads")
        dropout = _required_float(args, "dropout")
        if min(
            self.in_channels,
            self.encoder_dim,
            self.latent_dim,
            self.head_hidden,
        ) < 1:
            raise ValueError("P01Alignment dimensions must be positive")
        if self.encoder_dim < 8 or self.latent_dim < 2:
            raise ValueError("P01Alignment encoder/latent dimensions are degenerate")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("model.dropout must be in [0,1)")
        if attention_heads < 1 or self.latent_dim % attention_heads != 0:
            raise ValueError("model.latent_dim must be divisible by attention_heads")

        uses_1d = self.condition != "M2"
        uses_2d = self.condition != "M1"
        self.encoder_1d: nn.Module | None = (
            _SignalEncoder1D(self.in_channels, self.encoder_dim) if uses_1d else None
        )
        self.encoder_2d: nn.Module | None = (
            _TimeFrequencyEncoder2D(self.in_channels, self.encoder_dim)
            if uses_2d
            else None
        )
        self.project_1d: nn.Module | None = (
            _projection(self.encoder_dim, self.latent_dim) if uses_1d else None
        )
        self.project_2d: nn.Module | None = (
            _projection(self.encoder_dim, self.latent_dim) if uses_2d else None
        )

        if uses_2d:
            renderer = RendererConfig(
                n_fft=_required_int(args, "renderer.n_fft"),
                hop_length=_required_int(args, "renderer.hop_length"),
                win_length=_required_int(args, "renderer.win_length"),
                window=_required_str(args, "renderer.window"),
                window_periodic=_required_bool(args, "renderer.window_periodic"),
                center=_required_bool(args, "renderer.center"),
                pad_mode=_required_str(args, "renderer.pad_mode"),
                normalized=_required_bool(args, "renderer.normalized"),
                onesided=_required_bool(args, "renderer.onesided"),
                representation=_required_str(args, "renderer.representation"),
                scaling=_required_str(args, "renderer.scaling"),
                resize=_required_str(args, "renderer.resize"),
                normalization=_required_str(args, "renderer.normalization"),
            )
            self.renderer: DeterministicTimeFrequencyRenderer | None = (
                DeterministicTimeFrequencyRenderer(renderer)
            )
        else:
            self.renderer = None

        if self.condition == "M4":
            self.attention: nn.Module | None = nn.MultiheadAttention(
                self.latent_dim,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            fused_dim = self.latent_dim
        else:
            self.attention = None
            fused_dim = (
                2 * self.latent_dim
                if self.condition in {"M3", "M5"}
                else self.latent_dim
            )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, self.head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_hidden, int(num_classes)),
        )
        self._last_representation_state: Dict[str, torch.Tensor] | None = None

    def renderer_identity(self) -> Dict[str, Any] | None:
        return None if self.renderer is None else self.renderer.identity()

    def render_2d_view(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.renderer is None:
            raise RuntimeError(f"Condition {self.condition} has no 2D renderer")
        return self.renderer(waveform)

    def forward(
        self,
        waveform: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        return self.forward_paired_views(
            waveform,
            waveform,
            data_id=data_id,
            task_id=task_id,
        )

    def forward_paired_views(
        self,
        waveform: torch.Tensor,
        renderer_source: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        del data_id, task_id
        if waveform.ndim != 3 or renderer_source.ndim != 3:
            raise ValueError("P01Alignment expects paired sources shaped (B,L,C)")
        if waveform.shape != renderer_source.shape:
            raise ValueError(
                "P01Alignment paired sources must have identical shapes, got "
                f"{tuple(waveform.shape)} and {tuple(renderer_source.shape)}"
            )
        if waveform.shape[-1] != self.in_channels:
            raise ValueError(
                f"Configured in_channels={self.in_channels}, got {waveform.shape[-1]}"
            )

        state: Dict[str, torch.Tensor] = {}
        z_1: torch.Tensor | None = None
        z_2: torch.Tensor | None = None
        if self.encoder_1d is not None and self.project_1d is not None:
            encoded_1d = self.encoder_1d(waveform)
            z_1 = self.project_1d(encoded_1d)
            state.update({"encoded_1d": encoded_1d, "z_1": z_1})
        if (
            self.encoder_2d is not None
            and self.project_2d is not None
            and self.renderer is not None
        ):
            rendered_2d = self.renderer(renderer_source)
            encoded_2d = self.encoder_2d(rendered_2d)
            z_2 = self.project_2d(encoded_2d)
            state.update({"encoded_2d": encoded_2d, "z_2": z_2})

        if self.condition == "M1":
            if z_1 is None:
                raise RuntimeError("M1 1D branch was not constructed")
            fused = z_1
        elif self.condition == "M2":
            if z_2 is None:
                raise RuntimeError("M2 2D branch was not constructed")
            fused = z_2
        elif self.condition == "M4":
            if z_1 is None or z_2 is None or self.attention is None:
                raise RuntimeError("M4 paired attention path was not constructed")
            tokens = torch.stack((z_1, z_2), dim=1)
            attended, _ = self.attention(
                tokens, tokens, tokens, need_weights=False
            )
            fused = attended.mean(dim=1)
        else:
            if z_1 is None or z_2 is None:
                raise RuntimeError(f"{self.condition} paired path was not constructed")
            fused = torch.cat((z_1, z_2), dim=1)
        state["fused"] = fused
        self._last_representation_state = state
        return self.head(fused)

    def get_representation_state(
        self, *, detach: bool = True
    ) -> Dict[str, torch.Tensor]:
        if self._last_representation_state is None:
            raise RuntimeError("get_representation_state requires a preceding forward")
        if detach:
            return {
                name: value.detach()
                for name, value in self._last_representation_state.items()
            }
        return dict(self._last_representation_state)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

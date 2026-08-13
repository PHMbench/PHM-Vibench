"""Maintained M1--M5 conditions, C3, and the executable P01 objective.

M1--M4 expose the frozen reference forwards.  M5 keeps the same forward
parameterization as M3 and differs only because the maintained task consumes the
physical, semantic, and geometric losses defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CONDITIONS = ("M1", "M2", "M3", "M4", "M5", "C3")
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
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"model.{dotted} must be finite")
    return numeric


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


def _required_switch(obj: Any, dotted: str) -> int:
    value = _required(obj, dotted)
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError(f"model.{dotted} must be the integer 0 or 1")
    return int(value)


def _required_positive_float(obj: Any, dotted: str) -> float:
    value = _required_float(obj, dotted)
    if value <= 0.0:
        raise ValueError(f"model.{dotted} must be positive")
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


@dataclass(frozen=True)
class AlignmentConfig:
    """Frozen M5 objective coefficients and E0 audit threshold."""

    a_p: int
    a_s: int
    a_g: int
    lambda_p: float
    lambda_s: float
    lambda_g: float
    physical_energy_weight: float
    physical_spectral_weight: float
    physical_parseval_weight: float
    semantic_temperature: float
    eps: float
    gradient_min_norm: float


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

    def _flatten_waveform(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
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
        return flattened, batch, channels

    def _window(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(
            self.config.win_length,
            periodic=self.config.window_periodic,
            dtype=waveform.dtype,
            device=waveform.device,
        )

    def stft_magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the pre-log magnitude used by the frozen renderer."""

        flattened, batch, channels = self._flatten_waveform(waveform)
        spectrum = torch.stft(
            flattened,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self._window(waveform),
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            normalized=self.config.normalized,
            onesided=self.config.onesided,
            return_complex=True,
        )
        magnitude = spectrum.abs().view(
            batch, channels, spectrum.shape[-2], spectrum.shape[-1]
        )
        return magnitude

    def parseval_residual(
        self, waveform: torch.Tensor, *, eps: float
    ) -> torch.Tensor:
        """Audit frame-wise Parseval equality before magnitude ``log1p`` scaling.

        The published renderer is log-magnitude STFT, so Parseval equality is not
        claimed for its final pixels.  This component compares each Hann-windowed
        time frame with the corresponding pre-log, one-sided spectrum using the
        exact FFT normalization.  It is an input/transform diagnostic; the other
        physical components provide the trainable shared-space gradient.
        """

        if self.config.win_length != self.config.n_fft:
            raise ValueError(
                "Parseval audit requires renderer.win_length == renderer.n_fft"
            )
        flattened, _, _ = self._flatten_waveform(waveform)
        if self.config.center:
            pad = self.config.n_fft // 2
            flattened_for_frames = F.pad(
                flattened.unsqueeze(1),
                (pad, pad),
                mode=self.config.pad_mode,
            ).squeeze(1)
        else:
            flattened_for_frames = flattened
        frames = flattened_for_frames.unfold(
            -1, self.config.n_fft, self.config.hop_length
        )
        windowed = frames * self._window(waveform)
        time_energy = windowed.square().sum(dim=-1)

        magnitude = self.stft_magnitude(waveform).flatten(0, 1)
        frequency_weights = torch.ones(
            magnitude.shape[1], dtype=waveform.dtype, device=waveform.device
        )
        if self.config.n_fft % 2 == 0:
            frequency_weights[1:-1] = 2.0
        else:
            frequency_weights[1:] = 2.0
        spectral_energy = (
            magnitude.square() * frequency_weights.view(1, -1, 1)
        ).sum(dim=1)
        if not self.config.normalized:
            spectral_energy = spectral_energy / float(self.config.n_fft)
        if spectral_energy.shape != time_energy.shape:
            raise RuntimeError(
                "Renderer Parseval frame mismatch: "
                f"time={tuple(time_energy.shape)}, spectrum={tuple(spectral_energy.shape)}"
            )
        relative_error = (spectral_energy - time_energy) / time_energy.clamp_min(eps)
        return relative_error.square().mean()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        magnitude = self.stft_magnitude(waveform)
        return torch.log1p(magnitude)


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
        self.num_classes = int(num_classes)

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

        self.alignment_config: AlignmentConfig | None = None
        if self.condition == "M5":
            self.alignment_config = AlignmentConfig(
                a_p=_required_switch(args, "alignment.a_p"),
                a_s=_required_switch(args, "alignment.a_s"),
                a_g=_required_switch(args, "alignment.a_g"),
                lambda_p=_required_positive_float(args, "alignment.lambda_p"),
                lambda_s=_required_positive_float(args, "alignment.lambda_s"),
                lambda_g=_required_positive_float(args, "alignment.lambda_g"),
                physical_energy_weight=_required_positive_float(
                    args, "alignment.physical_energy_weight"
                ),
                physical_spectral_weight=_required_positive_float(
                    args, "alignment.physical_spectral_weight"
                ),
                physical_parseval_weight=_required_positive_float(
                    args, "alignment.physical_parseval_weight"
                ),
                semantic_temperature=_required_positive_float(
                    args, "alignment.semantic_temperature"
                ),
                eps=_required_positive_float(args, "alignment.eps"),
                gradient_min_norm=_required_positive_float(
                    args, "alignment.gradient_min_norm"
                ),
            )

        uses_1d = self.condition not in {"M2", "C3"}
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
        self.encoder_duplicate_2d: nn.Module | None = (
            _TimeFrequencyEncoder2D(self.in_channels, self.encoder_dim)
            if self.condition == "C3"
            else None
        )
        self.project_duplicate_2d: nn.Module | None = (
            _projection(self.encoder_dim, self.latent_dim)
            if self.condition == "C3"
            else None
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
                if self.condition in {"M3", "M5", "C3"}
                else self.latent_dim
            )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, self.head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_hidden, self.num_classes),
        )
        self._last_representation_state: Dict[str, torch.Tensor] | None = None

    @property
    def uses_alignment_objective(self) -> bool:
        return self.condition == "M5"

    def alignment_identity(self) -> Dict[str, Any] | None:
        return (
            None
            if self.alignment_config is None
            else asdict(self.alignment_config)
        )

    def renderer_identity(self) -> Dict[str, Any] | None:
        return None if self.renderer is None else self.renderer.identity()

    def duplicate_control_identity(self) -> Dict[str, Any] | None:
        """Describe and verify the executed C3 duplicate-rendering control."""
        if self.condition != "C3":
            return None
        modules = (
            self.encoder_2d,
            self.project_2d,
            self.encoder_duplicate_2d,
            self.project_duplicate_2d,
        )
        if any(module is None for module in modules):
            raise RuntimeError("C3 duplicate control is missing a required module")
        first_parameters = {
            parameter.data_ptr()
            for module in (self.encoder_2d, self.project_2d)
            for parameter in module.parameters()
        }
        second_parameters = {
            parameter.data_ptr()
            for module in (self.encoder_duplicate_2d, self.project_duplicate_2d)
            for parameter in module.parameters()
        }
        if first_parameters & second_parameters:
            raise RuntimeError("C3 duplicate branches unexpectedly share parameter storage")
        return {
            "representation": "frozen_log_magnitude_hann_stft",
            "renderer_execution": "single_call_shared_tensor_object",
            "branch_family": "time_frequency_2d",
            "branch_count": 2,
            "encoder_topology": "identical",
            "projection_topology": "identical",
            "parameter_storage": "independent_no_weight_sharing",
            "fusion": "concatenation_with_m5_shaped_head",
            "alignment_terms_consumed": "none",
        }

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

    def _forward_paired_views(
        self,
        waveform: torch.Tensor,
        renderer_source: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del data_id, task_id
        if waveform.ndim != 3 or renderer_source.ndim != 3:
            raise ValueError("P01Alignment expects paired sources shaped (B,L,C)")
        if waveform.shape != renderer_source.shape:
            raise ValueError(
                "P01Alignment paired sources must have identical shapes, got "
                f"{tuple(waveform.shape)} and {tuple(renderer_source.shape)}"
            )
        if self.condition == "C3" and not torch.equal(waveform, renderer_source):
            raise ValueError(
                "C3 requires both nominal views to receive the identical "
                "deterministic source tensor"
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
            projected_2d = self.project_2d(encoded_2d)
            if self.condition == "C3":
                if (
                    self.encoder_duplicate_2d is None
                    or self.project_duplicate_2d is None
                ):
                    raise RuntimeError("C3 duplicate 2D branch was not constructed")
                encoded_duplicate_2d = self.encoder_duplicate_2d(rendered_2d)
                z_1 = projected_2d
                z_2 = self.project_duplicate_2d(encoded_duplicate_2d)
                state.update(
                    {
                        "encoded_2d": encoded_2d,
                        "encoded_duplicate_2d": encoded_duplicate_2d,
                        "z_1": z_1,
                        "z_2": z_2,
                    }
                )
            else:
                z_2 = projected_2d
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
        elif self.condition in {"M3", "M5", "C3"}:
            if z_1 is None or z_2 is None:
                raise RuntimeError(f"{self.condition} paired path was not constructed")
            fused = torch.cat((z_1, z_2), dim=1)
        else:
            raise RuntimeError(f"Unsupported P01 condition {self.condition!r}")
        state["fused"] = fused
        logits = self.head(fused)
        return logits, state

    def forward_paired_views(
        self,
        waveform: torch.Tensor,
        renderer_source: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
    ) -> torch.Tensor:
        logits, state = self._forward_paired_views(
            waveform,
            renderer_source,
            data_id=data_id,
            task_id=task_id,
        )
        self._last_representation_state = state
        return logits

    def forward_with_alignment(
        self,
        waveform: torch.Tensor,
        labels: torch.Tensor,
        data_id: Any = None,
        task_id: Any = None,
        alignment_target_permutation: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the synchronized M5 forward and return raw alignment terms."""

        if not self.uses_alignment_objective:
            raise RuntimeError(
                f"Condition {self.condition} does not admit alignment-objective consumption"
            )
        logits, state = self._forward_paired_views(
            waveform,
            waveform,
            data_id=data_id,
            task_id=task_id,
        )
        if alignment_target_permutation is not None:
            state["alignment_target_permutation"] = alignment_target_permutation
        self._last_representation_state = state
        return logits, self.compute_alignment_losses(
            waveform,
            labels,
            state,
            target_permutation=alignment_target_permutation,
        )

    @staticmethod
    def _validated_target_permutation(
        permutation: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(permutation, torch.Tensor):
            raise TypeError("Alignment target permutation must be a tensor")
        if permutation.ndim != 1 or permutation.shape[0] != batch_size:
            raise ValueError(
                "Alignment target permutation must have shape (B,), got "
                f"{tuple(permutation.shape)} for B={batch_size}"
            )
        if permutation.dtype != torch.long:
            raise ValueError("Alignment target permutation must use torch.long")
        permutation = permutation.to(device=device)
        expected = torch.arange(batch_size, device=device)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError(
                "Alignment target permutation must contain every batch index once"
            )
        if bool(torch.eq(permutation, expected).any().item()):
            raise ValueError(
                "Alignment target permutation must be a derangement with no "
                "synchronized pair left unchanged"
            )
        return permutation

    @staticmethod
    def _energy_distribution(features: torch.Tensor, eps: float) -> torch.Tensor:
        energy = features.square().clamp_min(eps)
        return energy / energy.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _jensen_shannon(
        first: torch.Tensor, second: torch.Tensor, eps: float
    ) -> torch.Tensor:
        midpoint = 0.5 * (first + second)
        first_term = first * (
            first.clamp_min(eps).log() - midpoint.clamp_min(eps).log()
        )
        second_term = second * (
            second.clamp_min(eps).log() - midpoint.clamp_min(eps).log()
        )
        return 0.5 * (first_term.sum(dim=-1) + second_term.sum(dim=-1)).mean()

    def _physical_alignment(
        self,
        waveform: torch.Tensor,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        config = self.alignment_config
        if config is None or self.renderer is None:
            raise RuntimeError("Physical alignment requires the configured M5 renderer")
        if z_1.shape != z_2.shape or z_1.ndim != 2:
            raise ValueError(
                "Physical alignment requires z_1 and z_2 with identical (B,q) shapes"
            )

        energy_1 = self._energy_distribution(z_1, config.eps)
        energy_2 = self._energy_distribution(z_2, config.eps)
        energy_loss = self._jensen_shannon(energy_1, energy_2, config.eps)

        spectrum_1 = torch.fft.rfft(z_1, dim=-1, norm="ortho").abs().square()
        spectrum_2 = torch.fft.rfft(z_2, dim=-1, norm="ortho").abs().square()
        spectrum_1 = spectrum_1 / spectrum_1.sum(
            dim=-1, keepdim=True
        ).clamp_min(config.eps)
        spectrum_2 = spectrum_2 / spectrum_2.sum(
            dim=-1, keepdim=True
        ).clamp_min(config.eps)
        spectral_loss = F.mse_loss(spectrum_1, spectrum_2)

        parseval_loss = self.renderer.parseval_residual(
            waveform, eps=config.eps
        )
        total = (
            config.physical_energy_weight * energy_loss
            + config.physical_spectral_weight * spectral_loss
            + config.physical_parseval_weight * parseval_loss
        )
        return {
            "physical": total,
            "physical_energy": energy_loss,
            "physical_spectral": spectral_loss,
            "physical_parseval": parseval_loss,
        }

    def semantic_pair_masks(
        self, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return synchronized positives and different-class negatives only."""

        if labels.ndim != 1:
            raise ValueError(
                f"Semantic alignment labels must have shape (B,), got {tuple(labels.shape)}"
            )
        if labels.numel() < 2:
            raise ValueError("Semantic alignment requires at least two samples")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if labels.dtype not in integer_dtypes:
            raise ValueError("Semantic alignment labels must use an integer dtype")
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= self.num_classes:
            raise ValueError(
                f"Semantic alignment labels must be within [0,{self.num_classes})"
            )
        positive = torch.eye(
            labels.numel(), device=labels.device, dtype=torch.bool
        )
        different_class = labels[:, None] != labels[None, :]
        negative = different_class & ~positive
        if not bool(negative.any(dim=1).all().item()):
            raise ValueError(
                "Every semantic anchor requires at least one different-class negative"
            )
        return {
            "positive": positive,
            "negative": negative,
            "admissible": positive | negative,
        }

    def _semantic_alignment(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        config = self.alignment_config
        if config is None:
            raise RuntimeError("Semantic alignment requires M5 alignment config")
        if z_1.shape != z_2.shape or z_1.ndim != 2:
            raise ValueError(
                "Semantic alignment requires z_1 and z_2 with identical (B,q) shapes"
            )
        if labels.shape[0] != z_1.shape[0]:
            raise ValueError(
                "Semantic alignment requires one training label per paired sample"
            )
        masks = self.semantic_pair_masks(labels)
        similarity = F.normalize(z_1, dim=-1) @ F.normalize(z_2, dim=-1).T
        similarity = similarity / config.semantic_temperature
        masked = similarity.masked_fill(~masks["admissible"], -torch.inf)
        diagonal = similarity.diagonal()
        loss_1_to_2 = -diagonal + torch.logsumexp(masked, dim=1)
        loss_2_to_1 = -diagonal + torch.logsumexp(masked, dim=0)
        return 0.5 * (loss_1_to_2.mean() + loss_2_to_1.mean())

    def _normalized_pairwise_distances(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        config = self.alignment_config
        if config is None:
            raise RuntimeError("Geometric alignment requires M5 alignment config")
        if features.ndim != 2 or features.shape[0] < 3:
            raise ValueError(
                "Geometric alignment requires features shaped (B,q) with B >= 3"
            )
        distances = torch.cdist(features, features, p=2)
        off_diagonal = ~torch.eye(
            features.shape[0], device=features.device, dtype=torch.bool
        )
        scale = distances[off_diagonal].mean()
        if not bool(torch.isfinite(scale).item()) or float(scale.detach()) <= config.eps:
            raise ValueError(
                "Geometric alignment requires finite, non-collapsed off-diagonal distances"
            )
        return distances / scale, off_diagonal

    def _geometric_alignment(
        self, z_1: torch.Tensor, z_2: torch.Tensor
    ) -> torch.Tensor:
        if z_1.shape != z_2.shape:
            raise ValueError(
                "Geometric alignment requires z_1 and z_2 with identical shapes"
            )
        distances_1, mask_1 = self._normalized_pairwise_distances(z_1)
        distances_2, mask_2 = self._normalized_pairwise_distances(z_2)
        if not torch.equal(mask_1, mask_2):
            raise RuntimeError("Geometric alignment pair masks differ")
        return (distances_1[mask_1] - distances_2[mask_2]).square().mean()

    def compute_alignment_losses(
        self,
        waveform: torch.Tensor,
        labels: torch.Tensor,
        state: Mapping[str, torch.Tensor],
        *,
        target_permutation: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the three raw M5 losses without applying switches/lambdas."""

        if not self.uses_alignment_objective or self.alignment_config is None:
            raise RuntimeError("Alignment losses are defined only for condition M5")
        try:
            z_1 = state["z_1"]
            z_2 = state["z_2"]
        except KeyError as exc:
            raise RuntimeError("M5 alignment state is missing z_1 or z_2") from exc
        if target_permutation is not None:
            permutation = self._validated_target_permutation(
                target_permutation,
                batch_size=int(z_2.shape[0]),
                device=z_2.device,
            )
            z_2 = z_2.index_select(0, permutation)
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError("M5 alignment waveform contains NaN or Inf")
        if not bool(torch.isfinite(z_1).all().item()) or not bool(
            torch.isfinite(z_2).all().item()
        ):
            raise ValueError("M5 shared representation contains NaN or Inf")

        components = self._physical_alignment(waveform, z_1, z_2)
        components["semantic"] = self._semantic_alignment(z_1, z_2, labels)
        components["geometric"] = self._geometric_alignment(z_1, z_2)
        for name, value in components.items():
            if value.ndim != 0 or not bool(torch.isfinite(value).item()):
                raise FloatingPointError(
                    f"Alignment component {name!r} must be one finite scalar"
                )
        return components

    def compose_training_objective(
        self,
        classification_loss: torch.Tensor,
        alignment_losses: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply only ``a_k lambda_k`` and expose an exact reconstruction."""

        config = self.alignment_config
        if config is None or not self.uses_alignment_objective:
            raise RuntimeError("Training alignment objective is defined only for M5")
        if classification_loss.ndim != 0 or not bool(
            torch.isfinite(classification_loss).item()
        ):
            raise ValueError("classification_loss must be one finite scalar")
        required = {"physical", "semantic", "geometric"}
        missing = sorted(required - set(alignment_losses))
        if missing:
            raise KeyError(f"Missing alignment loss component(s): {missing}")

        weighted_physical = (
            config.a_p * config.lambda_p * alignment_losses["physical"]
        )
        weighted_semantic = (
            config.a_s * config.lambda_s * alignment_losses["semantic"]
        )
        weighted_geometric = (
            config.a_g * config.lambda_g * alignment_losses["geometric"]
        )
        total = (
            classification_loss
            + weighted_physical
            + weighted_semantic
            + weighted_geometric
        )
        result = {
            "classification": classification_loss,
            "physical": alignment_losses["physical"],
            "semantic": alignment_losses["semantic"],
            "geometric": alignment_losses["geometric"],
            "weighted_physical": weighted_physical,
            "weighted_semantic": weighted_semantic,
            "weighted_geometric": weighted_geometric,
            "total": total,
        }
        for name in (
            "physical_energy",
            "physical_spectral",
            "physical_parseval",
        ):
            if name in alignment_losses:
                result[name] = alignment_losses[name]
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("M5 training objective is not finite")
        return result

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

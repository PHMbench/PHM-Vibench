"""Role-constrained mixture of experts for falsifiable fault diagnosis studies.

The model deliberately separates architectural intent from scientific evidence.
Its four pathways are constrained to low-frequency, harmonic, impulsive-envelope,
and learned-residual representations, but a pathway is not called an identified
physical role until the external held-out matching and intervention protocol
passes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_NAMES: Tuple[str, ...] = (
    "low_frequency",
    "harmonic",
    "impulsive_envelope",
    "learned_residual",
)

EXPERT_REPRESENTATION_MODES: Tuple[str, ...] = (
    "role_constrained",
    "homogeneous_raw",
)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ExpertHead(nn.Module):
    """Parameter-matched encoder and classifier used by every role pathway."""

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        feature_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        expanded = hidden_channels * 2
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels,
                expanded,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(expanded), expanded),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(expanded, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.project(self.encoder(representation))
        return features, self.classifier(features)


class Model(nn.Module):
    """Four-role MoE with a learned router and a bounded physics-facing prior.

    Repository input convention is ``[batch, length, channels]``. ``forward``
    returns logits only so the standard classification task remains compatible.
    Diagnostic and deletion APIs are explicit and are intended for the P04
    held-out behavior-identification protocol.
    """

    role_names = ROLE_NAMES

    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        del metadata

        self.input_dim = int(getattr(args, "input_dim", 1))
        self.num_classes = getattr(args, "num_classes", None)
        self.feature_dim = int(getattr(args, "feature_dim", 64))
        self.expert_hidden_channels = int(
            getattr(args, "expert_hidden_channels", 32)
        )
        self.router_hidden_dim = int(getattr(args, "router_hidden_dim", 32))
        self.dropout_rate = float(getattr(args, "dropout", 0.1))
        self.routing_temperature = float(
            getattr(args, "routing_temperature", 1.0)
        )
        self.low_cutoff = float(getattr(args, "low_cutoff", 0.12))
        envelope_band = list(getattr(args, "envelope_band", [0.20, 0.80]))
        self.envelope_low = float(envelope_band[0]) if envelope_band else 0.20
        self.envelope_high = float(envelope_band[1]) if len(envelope_band) > 1 else 0.80
        self.filter_transition = float(
            getattr(args, "filter_transition", 0.03)
        )
        self.router_mode = str(getattr(args, "router_mode", "learned_prior"))
        self.expert_representation_mode = str(
            getattr(args, "expert_representation_mode", "role_constrained")
        )
        self.role_prior_assignment = str(
            getattr(args, "role_prior_assignment", "unspecified")
        )
        self.role_prior_max = float(getattr(args, "role_prior_max", 1.0))
        self.load_balance_weight = float(
            getattr(args, "load_balance_weight", 0.01)
        )
        self.entropy_floor_weight = float(
            getattr(args, "entropy_floor_weight", 0.01)
        )
        self.entropy_floor = float(getattr(args, "entropy_floor", 0.25))
        prior_initial = float(getattr(args, "role_prior_strength", 0.5))
        prior_permutation = tuple(
            int(index)
            for index in getattr(args, "role_prior_permutation", [0, 1, 2, 3])
        )
        self._validate_configuration(prior_initial, prior_permutation)
        self.register_buffer(
            "role_prior_permutation",
            torch.tensor(prior_permutation, dtype=torch.long),
            persistent=False,
        )

        if self.role_prior_max == 0.0:
            self.register_buffer("_fixed_prior_strength", torch.tensor(0.0))
            self.register_parameter("_prior_strength_logit", None)
        else:
            ratio = min(max(prior_initial / self.role_prior_max, 1e-4), 1.0 - 1e-4)
            self._prior_strength_logit = nn.Parameter(
                torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
            )
            self.register_buffer("_fixed_prior_strength", torch.tensor(0.0))

        self.experts = nn.ModuleList(
            [
                _ExpertHead(
                    input_dim=self.input_dim,
                    hidden_channels=self.expert_hidden_channels,
                    feature_dim=self.feature_dim,
                    num_classes=int(self.num_classes),
                    dropout=self.dropout_rate,
                )
                for _ in ROLE_NAMES
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(6, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.router_hidden_dim, len(ROLE_NAMES)),
        )
        self._last_diagnostics: Dict[str, torch.Tensor] = {}
        self._pending_auxiliary_losses: Dict[str, torch.Tensor] = {}

    def _validate_configuration(
        self,
        prior_initial: float,
        prior_permutation: Sequence[int],
    ) -> None:
        positive = {
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "expert_hidden_channels": self.expert_hidden_channels,
            "router_hidden_dim": self.router_hidden_dim,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive values required for: {', '.join(invalid)}")
        if not isinstance(self.num_classes, int) or self.num_classes <= 1:
            raise ValueError("model.num_classes must resolve to one integer greater than one")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.routing_temperature <= 0.0:
            raise ValueError("model.routing_temperature must be positive")
        if not 0.0 < self.low_cutoff < 1.0:
            raise ValueError("model.low_cutoff must be a normalized frequency in (0, 1)")
        if not 0.0 < self.envelope_low < self.envelope_high < 1.0:
            raise ValueError("model.envelope_band must satisfy 0 < low < high < 1")
        if not 0.0 < self.filter_transition < 0.5:
            raise ValueError("model.filter_transition must be in (0, 0.5)")
        if self.router_mode not in {
            "learned_prior",
            "learned_only",
            "prior_only",
            "uniform",
        }:
            raise ValueError(f"unsupported model.router_mode: {self.router_mode!r}")
        if self.expert_representation_mode not in EXPERT_REPRESENTATION_MODES:
            raise ValueError(
                "unsupported model.expert_representation_mode: "
                f"{self.expert_representation_mode!r}"
            )
        if self.role_prior_assignment not in {
            "unspecified",
            "aligned",
            "external_deranged",
        }:
            raise ValueError(
                "unsupported model.role_prior_assignment: "
                f"{self.role_prior_assignment!r}"
            )
        if self.role_prior_max < 0.0:
            raise ValueError("model.role_prior_max must be non-negative")
        if self.load_balance_weight < 0.0 or self.entropy_floor_weight < 0.0:
            raise ValueError("MoE auxiliary-loss weights must be non-negative")
        if not 0.0 <= self.entropy_floor <= 1.0:
            raise ValueError("model.entropy_floor must be in [0, 1]")
        if not 0.0 <= prior_initial <= self.role_prior_max:
            raise ValueError(
                "model.role_prior_strength must lie in [0, model.role_prior_max]"
            )
        if tuple(sorted(prior_permutation)) != tuple(range(len(ROLE_NAMES))):
            raise ValueError(
                "model.role_prior_permutation must be a permutation of [0, 1, 2, 3]"
            )
        if self.role_prior_assignment == "external_deranged" and any(
            index == assigned
            for index, assigned in enumerate(prior_permutation)
        ):
            raise ValueError(
                "model.role_prior_assignment=external_deranged requires a "
                "fixed-point-free role_prior_permutation"
            )

    @property
    def role_prior_strength(self) -> torch.Tensor:
        """Return the learned prior coefficient, bounded by ``role_prior_max``."""
        if self._prior_strength_logit is None:
            return self._fixed_prior_strength
        return self.role_prior_max * torch.sigmoid(self._prior_strength_logit)

    def _as_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "M_04_RoleConstrainedMoE expects [batch, length, channels], "
                f"received {tuple(x.shape)}"
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "M_04_RoleConstrainedMoE channel mismatch: expected "
                f"{self.input_dim}, received {x.shape[-1]}"
            )
        if x.shape[1] < 32:
            raise ValueError("M_04_RoleConstrainedMoE requires at least 32 samples")
        if not x.is_floating_point():
            x = x.float()
        x = torch.nan_to_num(x)
        return x.transpose(1, 2)

    @staticmethod
    def _standardize_window(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centered = x - x.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        return centered / scale, scale.squeeze(-1)

    @staticmethod
    def _normalized_frequency(length: int, reference: torch.Tensor) -> torch.Tensor:
        return (
            torch.fft.rfftfreq(length, d=1.0, device=reference.device)
            .to(reference.dtype)
            .mul(2.0)
        )

    def _smooth_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        frequencies = self._normalized_frequency(length, x)
        mask = torch.sigmoid(
            (self.low_cutoff - frequencies) / self.filter_transition
        )
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=length, dim=-1)

    def _smooth_bandpass(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        frequencies = self._normalized_frequency(length, x)
        lower = torch.sigmoid(
            (frequencies - self.envelope_low) / self.filter_transition
        )
        upper = torch.sigmoid(
            (self.envelope_high - frequencies) / self.filter_transition
        )
        return torch.fft.irfft(
            torch.fft.rfft(x, dim=-1) * lower * upper,
            n=length,
            dim=-1,
        )

    @staticmethod
    def _analytic_envelope(x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        multiplier = torch.zeros(length, dtype=x.dtype, device=x.device)
        multiplier[0] = 1.0
        if length % 2 == 0:
            multiplier[length // 2] = 1.0
            multiplier[1 : length // 2] = 2.0
        else:
            multiplier[1 : (length + 1) // 2] = 2.0
        analytic = torch.fft.ifft(torch.fft.fft(x, dim=-1) * multiplier, dim=-1)
        envelope = analytic.abs()
        return envelope - envelope.mean(dim=-1, keepdim=True)

    @staticmethod
    def _harmonic_representation(x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.fft.rfft(x, dim=-1).abs()
        if magnitude.shape[-1] > 1:
            magnitude = magnitude.clone()
            magnitude[..., 0] = 0.0
        log_magnitude = torch.log1p(magnitude)
        centered = log_magnitude - log_magnitude.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1)
        autocorrelation = torch.fft.irfft(
            spectrum * spectrum.conj(), n=centered.shape[-1], dim=-1
        ).real
        normalizer = autocorrelation[..., :1].abs().clamp_min(1e-8)
        return autocorrelation / normalizer

    def _role_representations(
        self, normalized: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        low_frequency = self._smooth_lowpass(normalized)
        harmonic = self._harmonic_representation(normalized)
        bandpassed = self._smooth_bandpass(normalized)
        impulsive_envelope = self._analytic_envelope(bandpassed)
        learned_residual = normalized - low_frequency
        return low_frequency, harmonic, impulsive_envelope, learned_residual

    def _expert_representations(
        self,
        normalized: torch.Tensor,
        role_representations: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Select expert inputs without changing any trainable parameter shape.

        The HOMO control keeps four independent, parameter-isomorphic heads and
        feeds every head the same per-channel, within-window standardized raw
        representation.  Router cues are still computed from the common
        analysis operators, so the control changes expert representations only.
        """
        if self.expert_representation_mode == "homogeneous_raw":
            return tuple(normalized for _ in ROLE_NAMES)
        return tuple(role_representations)

    def _router_inputs(
        self,
        raw: torch.Tensor,
        normalized: torch.Tensor,
        representations: Sequence[torch.Tensor],
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mono = normalized.mean(dim=1)
        length = mono.shape[-1]
        frequencies = self._normalized_frequency(length, mono)
        power = torch.fft.rfft(mono, dim=-1).abs().square()
        total_power = power.sum(dim=-1).clamp_min(1e-8)
        low_ratio = power[:, frequencies <= self.low_cutoff].sum(dim=-1) / total_power
        centroid = (power * frequencies).sum(dim=-1) / total_power

        harmonic = representations[1].mean(dim=1)
        upper_lag = max(2, min(harmonic.shape[-1], harmonic.shape[-1] // 3))
        harmonic_score = harmonic[:, 1:upper_lag].amax(dim=-1).clamp(0.0, 1.0)

        envelope = representations[2].mean(dim=1)
        envelope_centered = envelope - envelope.mean(dim=-1, keepdim=True)
        variance = envelope_centered.square().mean(dim=-1).clamp_min(1e-8)
        kurtosis = envelope_centered.pow(4).mean(dim=-1) / variance.square()
        impulse_score = torch.sigmoid((kurtosis - 3.0) / 2.0)

        raw_mono = raw.mean(dim=1)
        rms = raw_mono.square().mean(dim=-1).add(1e-8).sqrt()
        peak = raw_mono.abs().amax(dim=-1)
        crest = peak / rms.clamp_min(1e-8)
        rms_scale = scale.mean(dim=1).clamp_min(1e-8)
        bounded_log_rms = torch.tanh(torch.log(rms_scale))
        bounded_crest = torch.sigmoid((crest - 3.0) / 2.0)

        router_features = torch.stack(
            [
                bounded_log_rms,
                low_ratio,
                harmonic_score,
                impulse_score,
                centroid,
                bounded_crest,
            ],
            dim=-1,
        )
        residual_score = 1.0 - torch.stack(
            [low_ratio, harmonic_score, impulse_score], dim=-1
        ).amax(dim=-1)
        role_cues = torch.stack(
            [low_ratio, harmonic_score, impulse_score, residual_score.clamp(0.0, 1.0)],
            dim=-1,
        )
        return router_features, role_cues

    def _routing(
        self,
        router_features: torch.Tensor,
        role_cues: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in {"learned_prior", "learned_only", "prior_only", "uniform"}:
            raise ValueError(f"unsupported router mode: {mode!r}")
        learned_logits = self.router(router_features)
        assigned_cues = role_cues.index_select(-1, self.role_prior_permutation)
        prior_logits = (2.0 * assigned_cues - 1.0).clamp(-1.0, 1.0)
        if mode == "uniform":
            combined_logits = torch.zeros_like(learned_logits)
        elif mode == "learned_only":
            combined_logits = learned_logits
        elif mode == "prior_only":
            combined_logits = self.role_prior_strength * prior_logits
        else:
            combined_logits = learned_logits + self.role_prior_strength * prior_logits
        weights = F.softmax(combined_logits / self.routing_temperature, dim=-1)
        return weights, learned_logits, prior_logits, combined_logits

    @staticmethod
    def _apply_expert_mask(
        weights: torch.Tensor,
        expert_mask: Optional[torch.Tensor | Iterable[float]],
        renormalize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if expert_mask is None:
            mask = torch.ones_like(weights)
        else:
            mask = torch.as_tensor(
                expert_mask, dtype=weights.dtype, device=weights.device
            )
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(weights.shape[0], -1)
            if mask.shape != weights.shape:
                raise ValueError(
                    "expert_mask must have shape [experts] or [batch, experts]"
                )
            if torch.any(mask < 0.0) or torch.any(mask > 1.0):
                raise ValueError("expert_mask values must lie in [0, 1]")
        masked = weights * mask
        mass = masked.sum(dim=-1, keepdim=True)
        if torch.any(mass <= 1e-8):
            raise ValueError("expert_mask cannot delete every expert")
        if renormalize:
            masked = masked / mass
        return masked, mask

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
        *,
        expert_mask: Optional[torch.Tensor | Iterable[float]] = None,
        renormalize: bool = True,
        router_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = self._as_bcl(x)
        fft_dtype = (
            torch.float32
            if raw.dtype in {torch.float16, torch.bfloat16}
            else raw.dtype
        )
        raw_analysis = raw.to(dtype=fft_dtype)
        normalized, scale = self._standardize_window(raw_analysis)
        role_representations = self._role_representations(normalized)
        router_features, role_cues = self._router_inputs(
            raw_analysis, normalized, role_representations, scale
        )
        expert_representations = self._expert_representations(
            normalized, role_representations
        )
        mode = router_mode or self.router_mode
        weights, learned_logits, prior_logits, combined_logits = self._routing(
            router_features, role_cues, mode
        )
        masked_weights, applied_mask = self._apply_expert_mask(
            weights, expert_mask, renormalize
        )

        expert_features = []
        expert_logits = []
        representation_energy = []
        for expert, representation in zip(self.experts, expert_representations):
            features, logits = expert(representation)
            expert_features.append(features)
            expert_logits.append(logits)
            representation_energy.append(
                representation.square().mean(dim=(1, 2)).add(1e-8).sqrt()
            )
        feature_tensor = torch.stack(expert_features, dim=1)
        logit_tensor = torch.stack(expert_logits, dim=1)
        energy_tensor = torch.stack(representation_energy, dim=1)
        response_tensor = feature_tensor.square().mean(dim=-1).add(1e-8).sqrt()
        logits = torch.sum(logit_tensor * masked_weights.unsqueeze(-1), dim=1)
        entropy = -torch.sum(
            weights * weights.clamp_min(1e-8).log(), dim=-1
        ) / math.log(len(ROLE_NAMES))
        mean_usage = weights.mean(dim=0)
        load_balance_loss = len(ROLE_NAMES) * torch.mean(
            (mean_usage - 1.0 / len(ROLE_NAMES)).square()
        )
        entropy_floor_loss = F.relu(self.entropy_floor - entropy).mean()
        weighted_load_balance = self.load_balance_weight * load_balance_loss
        weighted_entropy_floor = self.entropy_floor_weight * entropy_floor_loss
        self._pending_auxiliary_losses = {
            "moe_load_balance": weighted_load_balance,
            "moe_entropy_floor": weighted_entropy_floor,
        }
        diagnostics = {
            "routing_weights": weights,
            "effective_routing_weights": masked_weights,
            "learned_router_logits": learned_logits,
            "role_prior_logits": prior_logits,
            "combined_router_logits": combined_logits,
            "router_features": router_features,
            "role_cues": role_cues,
            "routing_entropy": entropy,
            "expert_features": feature_tensor,
            "expert_logits": logit_tensor,
            "expert_response": response_tensor,
            "response_only_signature": response_tensor,
            "representation_energy": energy_tensor,
            "expert_mask": applied_mask,
            "role_prior_strength": self.role_prior_strength.expand(x.shape[0]),
            "load_balance_loss": load_balance_loss.expand(x.shape[0]),
            "entropy_floor_loss": entropy_floor_loss.expand(x.shape[0]),
        }
        self._last_diagnostics = {
            name: value.detach() for name, value in diagnostics.items()
        }
        return logits, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        file_id: Any = None,
        task_id: Any = None,
        return_diagnostics: bool = False,
        expert_mask: Optional[torch.Tensor | Iterable[float]] = None,
        renormalize: bool = True,
        router_mode: Optional[str] = None,
    ):
        del file_id
        if task_id not in {None, False, "classification"}:
            raise ValueError(
                "M_04_RoleConstrainedMoE supports classification only, "
                f"got {task_id!r}"
            )
        logits, diagnostics = self.forward_with_diagnostics(
            x,
            expert_mask=expert_mask,
            renormalize=renormalize,
            router_mode=router_mode,
        )
        if return_diagnostics:
            return logits, diagnostics
        return logits

    def delete_expert(
        self,
        x: torch.Tensor,
        expert: int | str,
        *,
        renormalize: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run a prespecified expert-deletion intervention."""
        if isinstance(expert, str):
            if expert not in ROLE_NAMES:
                raise ValueError(f"unknown expert role: {expert!r}")
            expert_index = ROLE_NAMES.index(expert)
        else:
            expert_index = int(expert)
        if not 0 <= expert_index < len(ROLE_NAMES):
            raise ValueError(f"expert index out of range: {expert_index}")
        mask = torch.ones(len(ROLE_NAMES), device=x.device)
        mask[expert_index] = 0.0
        return self.forward_with_diagnostics(
            x, expert_mask=mask, renormalize=renormalize
        )

    def deletion_effects(
        self, x: torch.Tensor, *, renormalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Return baseline and each single-expert deletion without rerouting."""
        baseline_logits, diagnostics = self.forward_with_diagnostics(x)
        weights = diagnostics["routing_weights"]
        expert_logits = diagnostics["expert_logits"]
        deleted_logits = []
        for expert_index in range(len(ROLE_NAMES)):
            mask = torch.ones_like(weights)
            mask[:, expert_index] = 0.0
            effective, _ = self._apply_expert_mask(weights, mask, renormalize)
            deleted_logits.append(
                torch.sum(expert_logits * effective.unsqueeze(-1), dim=1)
            )
        deleted = torch.stack(deleted_logits, dim=1)
        baseline_log_prob = F.log_softmax(baseline_logits, dim=-1)
        baseline_prob = baseline_log_prob.exp().unsqueeze(1)
        deleted_log_prob = F.log_softmax(deleted, dim=-1)
        deletion_kl = torch.sum(
            baseline_prob * (baseline_log_prob.unsqueeze(1) - deleted_log_prob),
            dim=-1,
        )
        return {
            "baseline_logits": baseline_logits,
            "deleted_logits": deleted,
            "deletion_kl": deletion_kl,
            "routing_weights": weights,
        }

    def behavioral_signature(self, x: torch.Tensor) -> torch.Tensor:
        """Return an observational four-statistic signature for each expert.

        Rows follow ``ROLE_NAMES``. Columns are mean feature response, routing
        weight, absolute expert logit magnitude, and representation energy.
        Held-out mechanism-cell aggregation and role matching are performed by
        ``role_identification.build_mechanism_signature``.
        """
        _, diagnostics = self.forward_with_diagnostics(x)
        return torch.stack(
            [
                diagnostics["expert_response"].mean(dim=0),
                diagnostics["routing_weights"].mean(dim=0),
                diagnostics["expert_logits"].abs().mean(dim=(0, 2)),
                diagnostics["representation_energy"].mean(dim=0),
            ],
            dim=-1,
        )

    def response_only_signature(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-routing per-observation expert responses as ``[B, 4]``.

        This is the primary response-only evaluator input.  Mechanism/cell
        aggregation, blinding, standardization across experts, and assignment
        remain external protocol operations.
        """
        _, diagnostics = self.forward_with_diagnostics(x)
        return diagnostics["response_only_signature"]

    def get_last_diagnostics(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_diagnostics)

    def consume_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """Return each pending weighted router loss once for task integration."""
        losses = self._pending_auxiliary_losses
        self._pending_auxiliary_losses = {}
        return losses


__all__ = ["Model", "ROLE_NAMES", "EXPERT_REPRESENTATION_MODES"]

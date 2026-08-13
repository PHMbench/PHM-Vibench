"""Role-constrained mixture of experts for falsifiable fault diagnosis studies.

The model deliberately separates architectural intent from scientific evidence.
Its four pathways are constrained to low-frequency, harmonic, impulsive-envelope,
and learned-residual representations, but a pathway is not called an identified
physical role until the external held-out matching and intervention protocol
passes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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

DECISIVE_REQUIRED_ARGUMENTS: Tuple[str, ...] = (
    "input_dim",
    "num_classes",
    "feature_dim",
    "expert_hidden_channels",
    "router_hidden_dim",
    "dropout",
    "routing_temperature",
    "low_order_cutoff",
    "envelope_order_band",
    "filter_transition_order",
    "harmonic_order_max",
    "harmonic_order_bandwidth",
    "load_reference_hp",
    "speed_reference_rpm",
    "speed_scale_rpm",
    "router_mode",
    "expert_representation_mode",
    "role_prior_assignment",
    "role_prior_max",
    "load_balance_weight",
    "physical_loss_weight",
    "entropy_floor_weight",
    "entropy_floor",
    "compatibility_alpha",
    "role_prior_strength",
    "role_prior_permutation",
    "semantic_alignment",
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
        structure_id: int,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "structure_id", torch.tensor(int(structure_id), dtype=torch.long)
        )
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
        self.metadata = metadata
        requested_arm = getattr(args, "scientific_arm", None)
        self.scientific_arm = (
            str(requested_arm).upper() if requested_arm is not None else None
        )
        if requested_arm is not None and self.scientific_arm not in {"P0", "P1", "P2"}:
            raise ValueError("model.scientific_arm must be P0, P1, or P2")
        self.is_decisive_protocol = self.scientific_arm in {"P0", "P1", "P2"}
        self.requires_physical_metadata = self.is_decisive_protocol
        if self.is_decisive_protocol:
            missing = [
                name for name in DECISIVE_REQUIRED_ARGUMENTS if not hasattr(args, name)
            ]
            if missing:
                raise ValueError(
                    "decisive P04 model requires explicit arguments: "
                    + ", ".join(missing)
                )

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
        self.low_order_cutoff = float(getattr(args, "low_order_cutoff", 4.0))
        order_band = list(getattr(args, "envelope_order_band", [8.0, 120.0]))
        self.envelope_order_low = float(order_band[0])
        self.envelope_order_high = float(order_band[1])
        self.filter_transition_order = float(
            getattr(args, "filter_transition_order", 0.25)
        )
        self.harmonic_order_max = int(getattr(args, "harmonic_order_max", 12))
        self.harmonic_order_bandwidth = float(
            getattr(args, "harmonic_order_bandwidth", 0.18)
        )
        self.load_reference_hp = float(getattr(args, "load_reference_hp", 3.0))
        self.speed_reference_rpm = float(
            getattr(args, "speed_reference_rpm", 1750.0)
        )
        self.speed_scale_rpm = float(getattr(args, "speed_scale_rpm", 100.0))
        default_router_mode = (
            "learned_only" if self.scientific_arm == "P1" else "learned_prior"
        )
        self.router_mode = str(getattr(args, "router_mode", default_router_mode))
        default_representation_mode = (
            "homogeneous_raw"
            if self.scientific_arm == "P1"
            else "role_constrained"
        )
        self.expert_representation_mode = str(
            getattr(args, "expert_representation_mode", default_representation_mode)
        )
        self.role_prior_assignment = str(
            getattr(args, "role_prior_assignment", "unspecified")
        )
        self.role_prior_max = float(getattr(args, "role_prior_max", 1.0))
        self.load_balance_weight = float(
            getattr(args, "load_balance_weight", 0.01)
        )
        self.physical_loss_weight = float(
            getattr(args, "physical_loss_weight", 0.0)
        )
        self.entropy_floor_weight = float(
            getattr(args, "entropy_floor_weight", 0.01)
        )
        self.entropy_floor = float(getattr(args, "entropy_floor", 0.25))
        self.compatibility_alpha = float(
            getattr(args, "compatibility_alpha", 1.0)
        )
        prior_initial = float(getattr(args, "role_prior_strength", 0.5))
        prior_permutation = tuple(
            int(index)
            for index in getattr(args, "role_prior_permutation", [0, 1, 2, 3])
        )
        semantic_alignment = tuple(
            int(index)
            for index in getattr(args, "semantic_alignment", prior_permutation)
        )
        self._validate_configuration(
            prior_initial, prior_permutation, semantic_alignment
        )
        self.register_buffer(
            "role_prior_permutation",
            torch.tensor(prior_permutation, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "semantic_alignment",
            torch.tensor(semantic_alignment, dtype=torch.long),
        )
        identity = torch.arange(len(ROLE_NAMES), dtype=torch.long)
        self.register_buffer("slot_to_structure", identity.clone())
        self.register_buffer("router_slot_to_structure", identity.clone())
        self.register_buffer("slot_to_origin", identity.clone())
        self.register_buffer(
            "compatibility_mean", torch.zeros(len(ROLE_NAMES), dtype=torch.float32)
        )
        self.register_buffer(
            "compatibility_std", torch.ones(len(ROLE_NAMES), dtype=torch.float32)
        )
        self.register_buffer(
            "compatibility_stats_fitted", torch.tensor(False, dtype=torch.bool)
        )

        if self.is_decisive_protocol:
            self.register_buffer(
                "_fixed_prior_strength", torch.tensor(self.compatibility_alpha)
            )
            self.register_parameter("_prior_strength_logit", None)
        elif self.role_prior_max == 0.0:
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
                    structure_id=structure_id,
                )
                for structure_id, _ in enumerate(ROLE_NAMES)
            ]
        )
        router_input_dim = 8 if self.is_decisive_protocol else 6
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, self.router_hidden_dim),
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
        semantic_alignment: Sequence[int],
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
        if self.is_decisive_protocol:
            if self.scientific_arm not in {"P0", "P1", "P2"}:
                raise ValueError("model.scientific_arm must be P0, P1, or P2")
            if self.low_order_cutoff <= 0.0:
                raise ValueError("model.low_order_cutoff must be positive")
            if not 0.0 < self.envelope_order_low < self.envelope_order_high:
                raise ValueError(
                    "model.envelope_order_band must satisfy 0 < low < high"
                )
            if self.filter_transition_order <= 0.0:
                raise ValueError("model.filter_transition_order must be positive")
            if self.harmonic_order_max < 1:
                raise ValueError("model.harmonic_order_max must be positive")
            if self.harmonic_order_bandwidth <= 0.0:
                raise ValueError("model.harmonic_order_bandwidth must be positive")
            if self.load_reference_hp <= 0.0 or self.speed_scale_rpm <= 0.0:
                raise ValueError("physical metadata reference scales must be positive")
        else:
            if not 0.0 < self.low_cutoff < 1.0:
                raise ValueError(
                    "model.low_cutoff must be a normalized frequency in (0, 1)"
                )
            if not 0.0 < self.envelope_low < self.envelope_high < 1.0:
                raise ValueError(
                    "model.envelope_band must satisfy 0 < low < high < 1"
                )
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
        if (
            self.load_balance_weight < 0.0
            or self.physical_loss_weight < 0.0
            or self.entropy_floor_weight < 0.0
        ):
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
        if tuple(sorted(semantic_alignment)) != tuple(range(len(ROLE_NAMES))):
            raise ValueError(
                "model.semantic_alignment must be a permutation of [0, 1, 2, 3]"
            )
        if self.role_prior_assignment == "external_deranged" and any(
            index == assigned
            for index, assigned in enumerate(prior_permutation)
        ):
            raise ValueError(
                "model.role_prior_assignment=external_deranged requires a "
                "fixed-point-free role_prior_permutation"
            )
        if self.is_decisive_protocol:
            if self.compatibility_alpha != 1.0:
                raise ValueError("P04 decisive compatibility_alpha is frozen at 1.0")
            if self.entropy_floor_weight != 0.0:
                raise ValueError(
                    "P04 decisive objective forbids entropy-floor/rescue losses"
                )
            if self.physical_loss_weight != 0.0:
                raise ValueError(
                    "fixed physical operators require physical_loss_weight=0.0"
                )
            identity = tuple(range(len(ROLE_NAMES)))
            if self.scientific_arm == "P0":
                if self.expert_representation_mode != "role_constrained":
                    raise ValueError("P0 requires role_constrained representations")
                if self.router_mode != "learned_prior":
                    raise ValueError("P0 requires learned_prior routing")
                if tuple(semantic_alignment) != identity:
                    raise ValueError("P0 requires identity semantic_alignment")
            elif self.scientific_arm == "P1":
                if self.expert_representation_mode != "homogeneous_raw":
                    raise ValueError("P1 requires homogeneous_raw representations")
                if self.router_mode != "learned_only":
                    raise ValueError("P1 requires learned_only routing")
                if tuple(semantic_alignment) != identity:
                    raise ValueError("P1 requires identity semantic_alignment")
            else:
                if self.expert_representation_mode != "role_constrained":
                    raise ValueError("P2 requires role_constrained representations")
                if self.router_mode != "learned_prior":
                    raise ValueError("P2 requires learned_prior routing")
                if tuple(semantic_alignment) == identity:
                    raise ValueError("P2 requires a non-identity semantic_alignment")
                if any(
                    structure == compatibility
                    for structure, compatibility in zip(identity, semantic_alignment)
                ):
                    raise ValueError(
                        "P2 semantic_alignment must be fixed-point-free"
                    )

    @property
    def role_prior_strength(self) -> torch.Tensor:
        """Return the learned prior coefficient, bounded by ``role_prior_max``."""
        if self._prior_strength_logit is None:
            return self._fixed_prior_strength
        return self.role_prior_max * torch.sigmoid(self._prior_strength_logit)

    @staticmethod
    def _file_id_vector(file_id: Any, batch_size: int) -> list[Any]:
        if isinstance(file_id, torch.Tensor):
            values = [value.item() for value in file_id.detach().cpu().view(-1)]
        elif isinstance(file_id, (list, tuple)):
            values = list(file_id)
        else:
            values = [file_id]
        if values == [None]:
            return []
        if len(values) == 1 and batch_size > 1:
            values = values * batch_size
        if len(values) != batch_size:
            raise ValueError(
                "file_id must contain one ID or one ID per sample: "
                f"received {len(values)} for batch_size={batch_size}"
            )
        return values

    def _metadata_row(self, file_id: Any) -> Any:
        if self.metadata is None:
            raise ValueError("physical metadata table is unavailable")
        candidates = [file_id]
        try:
            numeric = int(file_id)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and numeric not in candidates:
            candidates.append(numeric)
        text = str(file_id)
        if text not in candidates:
            candidates.append(text)
        for candidate in candidates:
            try:
                return self.metadata[candidate]
            except (KeyError, IndexError, TypeError):
                continue
        raise KeyError(f"file_id={file_id!r} is absent from physical metadata")

    @staticmethod
    def _numeric_metadata_value(
        row: Any,
        aliases: Sequence[str],
        *,
        field: str,
        file_id: Any,
    ) -> Optional[float]:
        values: list[tuple[str, float]] = []
        for alias in aliases:
            present = False
            value: Any = None
            if isinstance(row, Mapping):
                present = alias in row
                if present:
                    value = row[alias]
            elif hasattr(row, "index") and alias in row.index:
                present = True
                value = row[alias]
            elif hasattr(row, alias):
                present = True
                value = getattr(row, alias)
            if not present or value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field} for file_id={file_id!r} must be numeric"
                ) from exc
            if math.isnan(numeric):
                raise ValueError(
                    f"{field} for file_id={file_id!r} contains NaN"
                )
            if not math.isfinite(numeric):
                raise ValueError(
                    f"{field} for file_id={file_id!r} must be finite"
                )
            values.append((alias, numeric))
        if not values:
            return None
        reference = values[0][1]
        conflicting = [
            (name, value)
            for name, value in values[1:]
            if not math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-6)
        ]
        if conflicting:
            rendered = ", ".join(f"{name}={value}" for name, value in values)
            raise ValueError(
                f"contradictory {field} metadata for file_id={file_id!r}: {rendered}"
            )
        return reference

    @staticmethod
    def _metadata_tensor(
        value: Any,
        *,
        field: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(value, dtype=dtype, device=device).view(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if tensor.numel() == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size)
        if tensor.numel() != batch_size:
            raise ValueError(
                f"{field} must contain one value or batch_size values: "
                f"received {tensor.numel()} for batch_size={batch_size}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{field} must contain only finite values")
        return tensor

    def resolve_physical_metadata(
        self,
        file_id: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        explicit: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Resolve canonical per-sample sample-rate, speed, and load vectors.

        No value is inferred from signal length, dataset name, or another field.
        When table and explicit values are both supplied they must agree.
        """
        aliases = {
            "sample_rate_hz": (
                "sample_rate_hz",
                "Sample_rate",
                "Sample_Rate",
                "sample_rate",
            ),
            "rotation_speed_rpm": (
                "rotation_speed_rpm",
                "Rotational_speed_rpm",
                "Nominal_rpm",
                "RPM",
            ),
            "load_hp": ("load_hp", "Load_hp", "load", "Load"),
        }
        resolved: Dict[str, torch.Tensor] = {}
        file_ids = self._file_id_vector(file_id, batch_size)
        if file_ids and self.metadata is not None:
            rows = [self._metadata_row(value) for value in file_ids]
            for field, field_aliases in aliases.items():
                values = [
                    self._numeric_metadata_value(
                        row, field_aliases, field=field, file_id=current_id
                    )
                    for row, current_id in zip(rows, file_ids)
                ]
                if any(value is not None for value in values):
                    if any(value is None for value in values):
                        raise ValueError(
                            f"partial {field} metadata is not allowed within a batch"
                        )
                    resolved[field] = self._metadata_tensor(
                        values,
                        field=field,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    )
        if explicit is not None:
            unknown = set(explicit) - set(aliases)
            if unknown:
                raise ValueError(
                    "unknown physical metadata fields: " + ", ".join(sorted(unknown))
                )
            for field, value in explicit.items():
                explicit_tensor = self._metadata_tensor(
                    value,
                    field=field,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )
                if field in resolved and not torch.allclose(
                    resolved[field], explicit_tensor, rtol=1e-6, atol=1e-6
                ):
                    raise ValueError(
                        f"explicit {field} contradicts the metadata table"
                    )
                resolved[field] = explicit_tensor
        if self.requires_physical_metadata:
            missing = [field for field in aliases if field not in resolved]
            if missing:
                raise ValueError(
                    "missing required physical metadata: " + ", ".join(missing)
                )
            if torch.any(resolved["sample_rate_hz"] <= 0.0):
                raise ValueError("sample_rate_hz must be positive")
            if torch.any(resolved["rotation_speed_rpm"] <= 0.0):
                raise ValueError("rotation_speed_rpm must be positive for order analysis")
            if torch.any(resolved["load_hp"] < 0.0):
                raise ValueError("load_hp must be non-negative")
        return resolved

    def set_compatibility_statistics(
        self, mean: torch.Tensor | Sequence[float], std: torch.Tensor | Sequence[float]
    ) -> None:
        """Freeze train-partition-only compatibility standardization statistics."""
        mean_tensor = torch.as_tensor(
            mean, dtype=self.compatibility_mean.dtype, device=self.compatibility_mean.device
        ).view(-1)
        std_tensor = torch.as_tensor(
            std, dtype=self.compatibility_std.dtype, device=self.compatibility_std.device
        ).view(-1)
        expected = len(ROLE_NAMES)
        if mean_tensor.numel() != expected or std_tensor.numel() != expected:
            raise ValueError(f"compatibility statistics must each contain {expected} values")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("compatibility statistics must be finite")
        if torch.any(std_tensor <= 0.0):
            raise ValueError("compatibility standard deviations must be positive")
        self.compatibility_mean.copy_(mean_tensor)
        self.compatibility_std.copy_(std_tensor)
        self.compatibility_stats_fitted.fill_(True)

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
        if self.is_decisive_protocol:
            if not torch.isfinite(x).all():
                raise ValueError(
                    "decisive P04 input contains NaN or Inf; scientific inputs "
                    "must fail rather than be repaired"
                )
        else:
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

    @staticmethod
    def _frequency_axes(
        length: int,
        reference: torch.Tensor,
        physical_metadata: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cycles_per_sample = torch.fft.rfftfreq(
            length, d=1.0, device=reference.device
        ).to(reference.dtype)
        sample_rate = physical_metadata["sample_rate_hz"].to(
            device=reference.device, dtype=reference.dtype
        )
        speed_hz = physical_metadata["rotation_speed_rpm"].to(
            device=reference.device, dtype=reference.dtype
        ) / 60.0
        frequency_hz = sample_rate[:, None] * cycles_per_sample[None, :]
        order = frequency_hz / speed_hz[:, None]
        return frequency_hz, order

    def _smooth_lowpass(
        self, x: torch.Tensor, order_axis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        length = x.shape[-1]
        if order_axis is None:
            frequencies = self._normalized_frequency(length, x)
            mask = torch.sigmoid(
                (self.low_cutoff - frequencies) / self.filter_transition
            )
        else:
            mask = torch.sigmoid(
                (self.low_order_cutoff - order_axis)
                / self.filter_transition_order
            ).unsqueeze(1)
        return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=length, dim=-1)

    def _smooth_bandpass(
        self, x: torch.Tensor, order_axis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        length = x.shape[-1]
        if order_axis is None:
            frequencies = self._normalized_frequency(length, x)
            lower = torch.sigmoid(
                (frequencies - self.envelope_low) / self.filter_transition
            )
            upper = torch.sigmoid(
                (self.envelope_high - frequencies) / self.filter_transition
            )
        else:
            lower = torch.sigmoid(
                (order_axis - self.envelope_order_low)
                / self.filter_transition_order
            ).unsqueeze(1)
            upper = torch.sigmoid(
                (self.envelope_order_high - order_axis)
                / self.filter_transition_order
            ).unsqueeze(1)
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

    def _harmonic_order_mask(self, order_axis: torch.Tensor) -> torch.Tensor:
        target_orders = torch.arange(
            1,
            self.harmonic_order_max + 1,
            dtype=order_axis.dtype,
            device=order_axis.device,
        )
        distance = (order_axis.unsqueeze(-1) - target_orders).abs().amin(dim=-1)
        return torch.sigmoid(
            (self.harmonic_order_bandwidth - distance)
            / self.filter_transition_order
        )

    def _order_harmonic_representation(
        self, x: torch.Tensor, order_axis: torch.Tensor
    ) -> torch.Tensor:
        length = x.shape[-1]
        mask = self._harmonic_order_mask(order_axis).unsqueeze(1)
        return torch.fft.irfft(
            torch.fft.rfft(x, dim=-1) * mask, n=length, dim=-1
        )

    def _role_representations(
        self,
        normalized: torch.Tensor,
        order_axis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        low_frequency = self._smooth_lowpass(normalized, order_axis)
        harmonic = (
            self._harmonic_representation(normalized)
            if order_axis is None
            else self._order_harmonic_representation(normalized, order_axis)
        )
        bandpassed = self._smooth_bandpass(normalized, order_axis)
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
        return tuple(
            role_representations[int(structure_id)]
            for structure_id in self.slot_to_structure.tolist()
        )

    def _router_inputs(
        self,
        raw: torch.Tensor,
        normalized: torch.Tensor,
        representations: Sequence[torch.Tensor],
        scale: torch.Tensor,
        order_axis: Optional[torch.Tensor] = None,
        physical_metadata: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        generic_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.is_decisive_protocol:
            # Compute both router analyses for every arm. Only the selected view
            # affects logits, but the common operator schedule keeps active
            # compute matched across P0/P1/P2.
            generic_inputs = self._generic_router_inputs(raw, normalized, scale)

        mono = normalized.mean(dim=1)
        length = mono.shape[-1]
        power = torch.fft.rfft(mono, dim=-1).abs().square()
        total_power = power.sum(dim=-1).clamp_min(1e-8)
        if order_axis is None:
            frequencies = self._normalized_frequency(length, mono)
            low_ratio = (
                power[:, frequencies <= self.low_cutoff].sum(dim=-1) / total_power
            )
            centroid = (power * frequencies).sum(dim=-1) / total_power
            harmonic = representations[1].mean(dim=1)
            upper_lag = max(2, min(harmonic.shape[-1], harmonic.shape[-1] // 3))
            harmonic_score = harmonic[:, 1:upper_lag].amax(dim=-1).clamp(0.0, 1.0)
        else:
            low_mask = torch.sigmoid(
                (self.low_order_cutoff - order_axis)
                / self.filter_transition_order
            )
            harmonic_mask = self._harmonic_order_mask(order_axis)
            low_ratio = (power * low_mask).sum(dim=-1) / total_power
            harmonic_score = (power * harmonic_mask).sum(dim=-1) / total_power
            nyquist_order = order_axis[:, -1].clamp_min(1e-8)
            centroid = (
                (power * order_axis).sum(dim=-1) / total_power / nyquist_order
            )

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

        feature_values = [
            bounded_log_rms,
            low_ratio,
            harmonic_score,
            impulse_score,
            centroid,
            bounded_crest,
        ]
        if self.is_decisive_protocol:
            if physical_metadata is None:
                raise ValueError("decisive physical router requires physical metadata")
            speed = physical_metadata["rotation_speed_rpm"].to(raw)
            load = physical_metadata["load_hp"].to(raw)
            feature_values.extend(
                [
                    torch.tanh(
                        (speed - self.speed_reference_rpm) / self.speed_scale_rpm
                    ),
                    (load / self.load_reference_hp).clamp(0.0, 2.0),
                ]
            )
        router_features = torch.stack(feature_values, dim=-1)
        residual_score = 1.0 - torch.stack(
            [low_ratio, harmonic_score, impulse_score], dim=-1
        ).amax(dim=-1)
        role_cues = torch.stack(
            [low_ratio, harmonic_score, impulse_score, residual_score.clamp(0.0, 1.0)],
            dim=-1,
        )
        if self.scientific_arm == "P1":
            if generic_inputs is None:
                raise RuntimeError("P1 generic router inputs were not computed")
            return generic_inputs
        return router_features, role_cues

    @staticmethod
    def _generic_router_inputs(
        raw: torch.Tensor,
        normalized: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Time-domain generic P1 features with no frequency or role alignment."""
        mono = normalized.mean(dim=1)
        raw_mono = raw.mean(dim=1)
        rms = raw_mono.square().mean(dim=-1).add(1e-8).sqrt()
        peak = raw_mono.abs().amax(dim=-1)
        crest = peak / rms.clamp_min(1e-8)
        derivative = mono[:, 1:] - mono[:, :-1]
        derivative_rms = derivative.square().mean(dim=-1).add(1e-8).sqrt()
        zero_crossing = (mono[:, 1:] * mono[:, :-1] < 0.0).float().mean(dim=-1)
        variance = mono.square().mean(dim=-1).clamp_min(1e-8)
        skewness = mono.pow(3).mean(dim=-1) / variance.pow(1.5)
        kurtosis = mono.pow(4).mean(dim=-1) / variance.square()
        mean_abs = mono.abs().mean(dim=-1)
        channel_spread = normalized.mean(dim=-1).std(dim=1, unbiased=False)
        bounded_log_rms = torch.tanh(torch.log(scale.mean(dim=1).clamp_min(1e-8)))
        features = torch.stack(
            [
                bounded_log_rms,
                torch.sigmoid((crest - 3.0) / 2.0),
                torch.tanh(derivative_rms),
                zero_crossing,
                torch.tanh(skewness / 2.0),
                torch.tanh((kurtosis - 3.0) / 4.0),
                torch.tanh(mean_abs),
                torch.tanh(channel_spread),
            ],
            dim=-1,
        )
        role_cues = torch.zeros(
            raw.shape[0], len(ROLE_NAMES), dtype=raw.dtype, device=raw.device
        )
        return features, role_cues

    def _routing(
        self,
        router_features: torch.Tensor,
        role_cues: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in {"learned_prior", "learned_only", "prior_only", "uniform"}:
            raise ValueError(f"unsupported router mode: {mode!r}")
        learned_logits = self.router(router_features)
        if self.is_decisive_protocol:
            if self.scientific_arm in {"P0", "P2"}:
                if not bool(self.compatibility_stats_fitted.item()):
                    raise RuntimeError(
                        "compatibility statistics must be fitted on train data before routing"
                    )
                standardized = (
                    role_cues - self.compatibility_mean.to(role_cues)
                ) / self.compatibility_std.to(role_cues)
                prior_logits = standardized.index_select(
                    -1, self.semantic_alignment
                )
            else:
                prior_logits = torch.zeros_like(learned_logits)
        else:
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

    def _validate_slot_alignment(self) -> None:
        expected = self.slot_to_structure.detach().cpu().tolist()
        router = self.router_slot_to_structure.detach().cpu().tolist()
        experts = [int(expert.structure_id.item()) for expert in self.experts]
        if router != expected or experts != expected:
            raise RuntimeError(
                "inconsistent router/expert slot permutation: experts, router "
                "coordinates, and structural metadata must be permuted together"
            )
        semantic = self.semantic_alignment.detach().cpu().tolist()
        if self.scientific_arm in {"P0", "P1"} and semantic != expected:
            raise RuntimeError(
                f"{self.scientific_arm} semantic alignment no longer matches structure"
            )
        if self.scientific_arm == "P2" and any(
            left == right for left, right in zip(semantic, expected)
        ):
            raise RuntimeError("P2 semantic derangement acquired an aligned slot")

    def permute_slots_(self, permutation: Sequence[int]) -> "Model":
        """Apply one deterministic, complete numerical-slot permutation in-place.

        ``new_slot[i] = old_slot[permutation[i]]``. Expert modules, learned-router
        output rows, structural metadata, semantic alignment, and checkpoint
        buffers are moved together, preserving logits and loss.
        """
        indices = tuple(int(index) for index in permutation)
        if tuple(sorted(indices)) != tuple(range(len(ROLE_NAMES))):
            raise ValueError("slot permutation must be a permutation of [0, 1, 2, 3]")
        self._validate_slot_alignment()
        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=self.slot_to_structure.device
        )
        previous_experts = list(self.experts)
        self.experts = nn.ModuleList([previous_experts[index] for index in indices])
        output_layer = self.router[-1]
        if not isinstance(output_layer, nn.Linear):
            raise RuntimeError("router output layer must be Linear for slot permutation")
        parameter_index = index_tensor.to(output_layer.weight.device)
        with torch.no_grad():
            output_layer.weight.copy_(
                output_layer.weight.detach().index_select(0, parameter_index).clone()
            )
            if output_layer.bias is not None:
                output_layer.bias.copy_(
                    output_layer.bias.detach().index_select(0, parameter_index).clone()
                )
            self.slot_to_structure.copy_(
                self.slot_to_structure.index_select(0, index_tensor).clone()
            )
            self.router_slot_to_structure.copy_(
                self.router_slot_to_structure.index_select(0, index_tensor).clone()
            )
            self.slot_to_origin.copy_(
                self.slot_to_origin.index_select(0, index_tensor).clone()
            )
            self.semantic_alignment.copy_(
                self.semantic_alignment.index_select(0, index_tensor).clone()
            )
            self.role_prior_permutation.copy_(
                self.role_prior_permutation.index_select(0, index_tensor).clone()
            )
        self._validate_slot_alignment()
        return self

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._validate_slot_alignment()
        return result

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
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
        expert_mask: Optional[torch.Tensor | Iterable[float]] = None,
        renormalize: bool = True,
        router_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_slot_alignment()
        raw = self._as_bcl(x)
        fft_dtype = (
            torch.float32
            if raw.dtype in {torch.float16, torch.bfloat16}
            else raw.dtype
        )
        raw_analysis = raw.to(dtype=fft_dtype)
        resolved_metadata: Dict[str, torch.Tensor] = {}
        order_axis: Optional[torch.Tensor] = None
        if self.is_decisive_protocol:
            resolved_metadata = self.resolve_physical_metadata(
                file_id,
                batch_size=raw_analysis.shape[0],
                device=raw_analysis.device,
                dtype=raw_analysis.dtype,
                explicit=physical_metadata,
            )
            _, order_axis = self._frequency_axes(
                raw_analysis.shape[-1], raw_analysis, resolved_metadata
            )
        if self.is_decisive_protocol:
            # Decisive inputs are normalized once with train-partition statistics.
            # Per-window refitting would leak each held-out window's own moments and
            # erase the amplitude information used by the bounded router features.
            normalized = raw_analysis
            scale = raw_analysis.square().mean(dim=-1).add(1e-8).sqrt()
        else:
            normalized, scale = self._standardize_window(raw_analysis)
        role_representations = self._role_representations(normalized, order_axis)
        router_features, role_cues = self._router_inputs(
            raw_analysis,
            normalized,
            role_representations,
            scale,
            order_axis,
            resolved_metadata,
        )
        expert_representations = self._expert_representations(
            normalized, role_representations
        )
        mode = self.router_mode if router_mode is None else router_mode
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
        if self.is_decisive_protocol:
            self._pending_auxiliary_losses = {
                "moe_load_balance": weighted_load_balance,
            }
        else:
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
            "slot_to_structure": self.slot_to_structure.expand(x.shape[0], -1),
            "slot_to_origin": self.slot_to_origin.expand(x.shape[0], -1),
            "semantic_alignment": self.semantic_alignment.expand(x.shape[0], -1),
        }
        if resolved_metadata:
            diagnostics.update(resolved_metadata)
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
        physical_metadata: Optional[Mapping[str, Any]] = None,
    ):
        if task_id not in {None, False, "classification"}:
            raise ValueError(
                "M_04_RoleConstrainedMoE supports classification only, "
                f"got {task_id!r}"
            )
        logits, diagnostics = self.forward_with_diagnostics(
            x,
            file_id=file_id,
            physical_metadata=physical_metadata,
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
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run a prespecified expert-deletion intervention."""
        if self.is_decisive_protocol and isinstance(expert, str):
            raise ValueError(
                "decisive interventions require a recovered role-to-slot mapping; "
                "structural role names are not valid intervention targets"
            )
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
            x,
            file_id=file_id,
            physical_metadata=physical_metadata,
            expert_mask=mask,
            renormalize=renormalize,
        )

    def delete_recovered_role(
        self,
        x: torch.Tensor,
        role_id: int,
        role_to_slot: Mapping[int, int],
        *,
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
        renormalize: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Delete the slot assigned by a frozen blinded role-recovery map."""
        try:
            normalized_mapping = {
                int(role): int(slot) for role, slot in role_to_slot.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("role_to_slot must be an integer mapping") from exc
        expected = set(range(len(ROLE_NAMES)))
        if set(normalized_mapping) != expected or set(normalized_mapping.values()) != expected:
            raise ValueError("role_to_slot must be a complete bijection over [0, 1, 2, 3]")
        target = int(role_id)
        if target not in expected:
            raise ValueError(f"unknown recovered role target: {role_id!r}")
        return self.delete_expert(
            x,
            normalized_mapping[target],
            renormalize=renormalize,
            file_id=file_id,
            physical_metadata=physical_metadata,
        )

    def deletion_effects(
        self,
        x: torch.Tensor,
        *,
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
        renormalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return baseline and each single-expert deletion without rerouting."""
        baseline_logits, diagnostics = self.forward_with_diagnostics(
            x, file_id=file_id, physical_metadata=physical_metadata
        )
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

    def behavioral_signature(
        self,
        x: torch.Tensor,
        *,
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Return an observational four-statistic signature for each expert.

        Rows follow ``ROLE_NAMES``. Columns are mean feature response, routing
        weight, absolute expert logit magnitude, and representation energy.
        Held-out mechanism-cell aggregation and role matching are performed by
        ``role_identification.build_mechanism_signature``.
        """
        _, diagnostics = self.forward_with_diagnostics(
            x, file_id=file_id, physical_metadata=physical_metadata
        )
        return torch.stack(
            [
                diagnostics["expert_response"].mean(dim=0),
                diagnostics["routing_weights"].mean(dim=0),
                diagnostics["expert_logits"].abs().mean(dim=(0, 2)),
                diagnostics["representation_energy"].mean(dim=0),
            ],
            dim=-1,
        )

    def response_only_signature(
        self,
        x: torch.Tensor,
        *,
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Return pre-routing per-observation expert responses as ``[B, 4]``.

        This is the primary response-only evaluator input.  Mechanism/cell
        aggregation, blinding, standardization across experts, and assignment
        remain external protocol operations.
        """
        _, diagnostics = self.forward_with_diagnostics(
            x, file_id=file_id, physical_metadata=physical_metadata
        )
        return diagnostics["response_only_signature"]

    def compatibility_cues(
        self,
        x: torch.Tensor,
        *,
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Return unstandardized canonical cues for train-only statistic fitting."""
        if self.scientific_arm not in {"P0", "P2"}:
            raise ValueError("compatibility cues are defined only for P0 and P2")
        self._validate_slot_alignment()
        raw = self._as_bcl(x)
        dtype = (
            torch.float32
            if raw.dtype in {torch.float16, torch.bfloat16}
            else raw.dtype
        )
        raw_analysis = raw.to(dtype=dtype)
        resolved = self.resolve_physical_metadata(
            file_id,
            batch_size=raw_analysis.shape[0],
            device=raw_analysis.device,
            dtype=raw_analysis.dtype,
            explicit=physical_metadata,
        )
        _, order_axis = self._frequency_axes(
            raw_analysis.shape[-1], raw_analysis, resolved
        )
        normalized = raw_analysis
        scale = raw_analysis.square().mean(dim=-1).add(1e-8).sqrt()
        representations = self._role_representations(normalized, order_axis)
        _, cues = self._router_inputs(
            raw_analysis,
            normalized,
            representations,
            scale,
            order_axis,
            resolved,
        )
        return cues

    def probe_response_signature(
        self,
        x: torch.Tensor,
        transformed_x: torch.Tensor,
        *,
        file_id: Any = None,
        physical_metadata: Optional[Mapping[str, Any]] = None,
        transformed_physical_metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Return frozen paired-probe q as ``[batch, numerical_slot, 3]``."""
        if x.shape != transformed_x.shape:
            raise ValueError("paired probe tensors must have identical shape")
        _, baseline = self.forward_with_diagnostics(
            x, file_id=file_id, physical_metadata=physical_metadata
        )
        _, transformed = self.forward_with_diagnostics(
            transformed_x,
            file_id=file_id,
            physical_metadata=(
                transformed_physical_metadata
                if transformed_physical_metadata is not None
                else physical_metadata
            ),
        )
        baseline_features = baseline["expert_features"]
        transformed_features = transformed["expert_features"]
        delta_routing = (
            transformed["routing_weights"] - baseline["routing_weights"]
        )
        delta_feature_norm = (
            transformed_features.norm(dim=-1) - baseline_features.norm(dim=-1)
        )
        cosine_distance = 1.0 - F.cosine_similarity(
            baseline_features, transformed_features, dim=-1, eps=1e-8
        )
        return torch.stack(
            [delta_routing, delta_feature_norm, cosine_distance], dim=-1
        )

    def get_last_diagnostics(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_diagnostics)

    def consume_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """Return each pending weighted router loss once for task integration."""
        losses = self._pending_auxiliary_losses
        self._pending_auxiliary_losses = {}
        return losses


__all__ = ["Model", "ROLE_NAMES", "EXPERT_REPRESENTATION_MODES"]

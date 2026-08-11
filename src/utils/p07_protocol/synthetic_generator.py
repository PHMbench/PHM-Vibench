"""Deterministic synthetic signal and episode protocol for P07 E7.

The generator is intentionally independent of the trainable XOAN model and of
the private executable-path implementation.  Sample identity and randomness
come exclusively from :mod:`path_universe`; targets are produced exclusively
by that module's public, independently implemented oracle.

This module creates tensors and in-memory provenance objects only.  It does not
write data, train a model, or declare any result evidence-eligible.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Sequence, cast

import torch

from . import path_universe as _path_protocol


JsonValue = _path_protocol.JsonValue

SCHEMA_VERSION: Final[int] = 1
PROTOCOL_ID: Final[str] = "P07-E7-SYNTHETIC-GENERATOR-v1"
NORMALIZATION_PROTOCOL_ID: Final[str] = "P07-E7-NORMALIZATION-v1"
NUISANCE_PROTOCOL_ID: Final[str] = "P07-E7-NUISANCE-v1"
EPISODE_PROTOCOL_ID: Final[str] = "P07-E7-SYNTHETIC-EPISODE-v1"

TENSOR_LAYOUT: Final[str] = "BLC"
SEQUENCE_LENGTH: Final[int] = 256
CHANNEL_COUNT: Final[int] = 2
ALLOWED_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.float32, torch.float64)

MULTISINE_COMPONENTS: Final[int] = 3
MULTISINE_FREQUENCY_MIN: Final[float] = 0.03
MULTISINE_FREQUENCY_MAX: Final[float] = 0.35
MULTISINE_FREQUENCY_SEPARATION: Final[float] = 0.02
MULTISINE_AMPLITUDE_MIN: Final[float] = 0.4
MULTISINE_AMPLITUDE_MAX: Final[float] = 1.0

IMPULSE_COUNT_MIN: Final[int] = 1
IMPULSE_COUNT_MAX: Final[int] = 3
IMPULSE_MAGNITUDE_MIN: Final[float] = 0.4
IMPULSE_MAGNITUDE_MAX: Final[float] = 1.0
IMPULSE_DECAY_MIN_SAMPLES: Final[float] = 6.0
IMPULSE_DECAY_MAX_SAMPLES: Final[float] = 24.0

AR1_RHO_MIN: Final[float] = 0.25
AR1_RHO_MAX: Final[float] = 0.85
AR1_COMPONENT_RMS: Final[float] = 0.12

SNR_LEVELS_DB: Final[tuple[int | None, ...]] = (None, 20, 10)
SCALE_LEVELS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
CIRCULAR_SHIFTS: Final[tuple[int, ...]] = (-32, 0, 32)
NUISANCE_ORDER: Final[tuple[str, ...]] = (
    "normalize",
    "scale",
    "circular_shift",
    "additive_noise",
)

_NORMALIZATION_STATISTIC: Final[str] = (
    "per-channel population mean and centered RMS over fit samples and length"
)
_NORMALIZATION_HASH_KIND: Final[str] = "p07_fit_sample_ids"
_NUISANCE_CELL_PREFIX: Final[str] = "P07-NUISANCE-"


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_dtype(dtype: Any) -> torch.dtype:
    if dtype not in ALLOWED_DTYPES:
        raise TypeError("dtype must be torch.float32 or torch.float64.")
    return cast(torch.dtype, dtype)


def _validate_blc_tensor(x: Any, *, name: str, batch_size: int | None = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim != 3 or tuple(x.shape[1:]) != (SEQUENCE_LENGTH, CHANNEL_COUNT):
        raise ValueError(
            f"{name} must have BLC shape (B,{SEQUENCE_LENGTH},{CHANNEL_COUNT}), "
            f"got {tuple(x.shape)}."
        )
    if int(x.shape[0]) <= 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if batch_size is not None and int(x.shape[0]) != batch_size:
        raise ValueError(
            f"{name} batch size {int(x.shape[0])} does not match expected {batch_size}."
        )
    _require_dtype(x.dtype)
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return x


def _validate_sample_ids(
    sample_ids: Any, *, expected_count: int | None = None, require_fit: bool = False
) -> tuple[str, ...]:
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise TypeError("sample_ids must be a non-string sequence.")
    identifiers = tuple(_path_protocol.validate_sample_id(item) for item in sample_ids)
    if not identifiers:
        raise ValueError("sample_ids must be nonempty.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_ids must be unique.")
    if expected_count is not None and len(identifiers) != expected_count:
        raise ValueError(
            f"sample_ids count {len(identifiers)} does not match expected {expected_count}."
        )
    if require_fit:
        unknown = set(identifiers).difference(_registered_fit_sample_ids())
        if unknown:
            raise ValueError("Normalization may be fitted only on registered fit samples.")
    return identifiers


@lru_cache(maxsize=1)
def _registered_fit_sample_ids() -> frozenset[str]:
    return frozenset(
        _path_protocol.make_sample_id("fit", generator_seed, sample_index)
        for generator_seed in _path_protocol.GENERATOR_SEED_NAMESPACES["fit"]
        for sample_index in range(_path_protocol.SAMPLES_PER_GENERATOR_SEED["fit"])
    )


def _cpu_generator(sample_id: str, purpose: str, *components: str | int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        _path_protocol.derive_sample_seed(sample_id, purpose, *components)
    )
    return generator


def _multisine_component(sample_id: str, channel: int) -> torch.Tensor:
    generator = _cpu_generator(sample_id, "synthetic-multisine", channel)
    raw_locations = torch.sort(
        torch.rand(MULTISINE_COMPONENTS, generator=generator, dtype=torch.float64)
    ).values
    free_span = (
        MULTISINE_FREQUENCY_MAX
        - MULTISINE_FREQUENCY_MIN
        - (MULTISINE_COMPONENTS - 1) * MULTISINE_FREQUENCY_SEPARATION
    )
    offsets = torch.arange(MULTISINE_COMPONENTS, dtype=torch.float64)
    frequencies = (
        MULTISINE_FREQUENCY_MIN
        + offsets * MULTISINE_FREQUENCY_SEPARATION
        + free_span * raw_locations
    )
    amplitudes = MULTISINE_AMPLITUDE_MIN + (
        MULTISINE_AMPLITUDE_MAX - MULTISINE_AMPLITUDE_MIN
    ) * torch.rand(MULTISINE_COMPONENTS, generator=generator, dtype=torch.float64)
    phases = 2.0 * math.pi * torch.rand(
        MULTISINE_COMPONENTS, generator=generator, dtype=torch.float64
    ) - math.pi
    time = torch.arange(SEQUENCE_LENGTH, dtype=torch.float64)
    angles = 2.0 * math.pi * frequencies[:, None] * time[None, :] + phases[:, None]
    return (amplitudes[:, None] * torch.sin(angles)).sum(dim=0)


def _damped_impulse_component(sample_id: str, channel: int) -> torch.Tensor:
    generator = _cpu_generator(sample_id, "synthetic-damped-impulses", channel)
    count = int(
        torch.randint(
            IMPULSE_COUNT_MIN,
            IMPULSE_COUNT_MAX + 1,
            (1,),
            generator=generator,
        ).item()
    )
    positions = torch.randperm(SEQUENCE_LENGTH, generator=generator)[:count]
    magnitudes = IMPULSE_MAGNITUDE_MIN + (
        IMPULSE_MAGNITUDE_MAX - IMPULSE_MAGNITUDE_MIN
    ) * torch.rand(count, generator=generator, dtype=torch.float64)
    signs = 2.0 * torch.randint(
        0, 2, (count,), generator=generator, dtype=torch.int64
    ).to(torch.float64) - 1.0
    decays = IMPULSE_DECAY_MIN_SAMPLES + (
        IMPULSE_DECAY_MAX_SAMPLES - IMPULSE_DECAY_MIN_SAMPLES
    ) * torch.rand(count, generator=generator, dtype=torch.float64)

    time = torch.arange(SEQUENCE_LENGTH, dtype=torch.float64)
    delays = time[None, :] - positions.to(torch.float64)[:, None]
    causal = delays >= 0.0
    responses = signs[:, None] * magnitudes[:, None] * torch.exp(
        -torch.clamp_min(delays, 0.0) / decays[:, None]
    )
    return (responses * causal.to(torch.float64)).sum(dim=0)


def _ar1_component(sample_id: str, channel: int) -> torch.Tensor:
    generator = _cpu_generator(sample_id, "synthetic-ar1", channel)
    rho = AR1_RHO_MIN + (AR1_RHO_MAX - AR1_RHO_MIN) * float(
        torch.rand((), generator=generator, dtype=torch.float64).item()
    )
    innovations = torch.randn(
        SEQUENCE_LENGTH, generator=generator, dtype=torch.float64
    )
    state = torch.empty(SEQUENCE_LENGTH, dtype=torch.float64)
    state[0] = innovations[0]
    innovation_scale = math.sqrt(1.0 - rho * rho)
    for index in range(1, SEQUENCE_LENGTH):
        state[index] = rho * state[index - 1] + innovation_scale * innovations[index]
    state = state - state.mean()
    rms = state.square().mean().sqrt()
    if not bool(torch.isfinite(rms)) or float(rms) <= torch.finfo(torch.float64).eps:
        raise RuntimeError("AR(1) component unexpectedly has invalid RMS.")
    return state * (AR1_COMPONENT_RMS / rms)


def generate_root_signal(
    sample_id: str, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Generate one deterministic root signal with exact ``(1,256,2)`` BLC shape."""

    identifier = _path_protocol.validate_sample_id(sample_id)
    output_dtype = _require_dtype(dtype)
    channels = []
    for channel in range(CHANNEL_COUNT):
        signal = (
            _multisine_component(identifier, channel)
            + _damped_impulse_component(identifier, channel)
            + _ar1_component(identifier, channel)
        )
        channels.append(signal)
    root = torch.stack(channels, dim=-1).unsqueeze(0)
    _validate_blc_tensor(root, name="generated root signal", batch_size=1)
    return root.to(dtype=output_dtype)


def generate_root_batch(
    sample_ids: Sequence[str], *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Generate a deterministic BLC batch while preserving requested sample order."""

    identifiers = _validate_sample_ids(sample_ids)
    output_dtype = _require_dtype(dtype)
    return torch.cat(
        [generate_root_signal(item, dtype=output_dtype) for item in identifiers], dim=0
    )


def _fit_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return _path_protocol.canonical_json_sha256(
        {
            "fit_sample_ids": list(sample_ids),
            "kind": _NORMALIZATION_HASH_KIND,
            "protocol_id": NORMALIZATION_PROTOCOL_ID,
        }
    )


@dataclass(frozen=True, slots=True)
class NormalizationArtifact:
    """Canonical fitted normalization statistics and their fit-sample lineage."""

    fit_sample_ids: tuple[str, ...]
    fit_sample_ids_sha256: str
    mean: tuple[float, float]
    centered_rms: tuple[float, float]
    generator_manifest_sha256: str

    def payload(self) -> dict[str, JsonValue]:
        return {
            "centered_rms": list(self.centered_rms),
            "channel_count": CHANNEL_COUNT,
            "fit_sample_count": len(self.fit_sample_ids),
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_sample_ids_sha256": self.fit_sample_ids_sha256,
            "generator_manifest_sha256": self.generator_manifest_sha256,
            "mean": list(self.mean),
            "protocol_id": NORMALIZATION_PROTOCOL_ID,
            "schema_version": SCHEMA_VERSION,
            "sequence_length": SEQUENCE_LENGTH,
            "statistic": _NORMALIZATION_STATISTIC,
        }

    @property
    def artifact_sha256(self) -> str:
        return _path_protocol.canonical_json_sha256(self.payload())

    def manifest(self) -> dict[str, JsonValue]:
        return {
            "artifact": self.payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def to_json(self) -> str:
        return _path_protocol.canonical_json_bytes(self.manifest()).decode("utf-8")


def validate_normalization_artifact(artifact: Any) -> NormalizationArtifact:
    if not isinstance(artifact, NormalizationArtifact):
        raise TypeError("artifact must be a NormalizationArtifact.")
    if not isinstance(artifact.fit_sample_ids, tuple):
        raise TypeError("Normalization fit_sample_ids must be an immutable tuple.")
    if not isinstance(artifact.mean, tuple) or not isinstance(
        artifact.centered_rms, tuple
    ):
        raise TypeError("Normalization mean/RMS must be immutable tuples.")
    sample_ids = _validate_sample_ids(artifact.fit_sample_ids, require_fit=True)
    if sample_ids != tuple(sorted(sample_ids)):
        raise ValueError("Normalization fit_sample_ids must use canonical lexical order.")
    expected_sample_hash = _fit_sample_ids_sha256(sample_ids)
    if artifact.fit_sample_ids_sha256 != expected_sample_hash:
        raise ValueError("Normalization fit-sample hash mismatch.")
    expected_generator_hash = cast(
        str, build_synthetic_generator_manifest()["manifest_sha256"]
    )
    if artifact.generator_manifest_sha256 != expected_generator_hash:
        raise ValueError("Normalization artifact references a different generator protocol.")
    if len(artifact.mean) != CHANNEL_COUNT or len(artifact.centered_rms) != CHANNEL_COUNT:
        raise ValueError("Normalization mean/RMS must have one value per channel.")
    for value in (*artifact.mean, *artifact.centered_rms):
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("Normalization statistics must be finite floats.")
    if any(value <= torch.finfo(torch.float64).eps for value in artifact.centered_rms):
        raise ValueError("Normalization centered RMS values must be positive.")
    return artifact


def estimate_normalization_artifact(
    fit_sample_ids: Sequence[str],
) -> NormalizationArtifact:
    """Fit per-channel mean and centered population RMS from fit IDs only."""

    identifiers = tuple(sorted(_validate_sample_ids(fit_sample_ids, require_fit=True)))
    roots = generate_root_batch(identifiers, dtype=torch.float64)
    mean = roots.mean(dim=(0, 1))
    centered_rms = (roots - mean).square().mean(dim=(0, 1)).sqrt()
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(centered_rms).all()):
        raise RuntimeError("Normalization estimation produced non-finite statistics.")
    artifact = NormalizationArtifact(
        fit_sample_ids=identifiers,
        fit_sample_ids_sha256=_fit_sample_ids_sha256(identifiers),
        mean=cast(tuple[float, float], tuple(float(item) for item in mean.tolist())),
        centered_rms=cast(
            tuple[float, float], tuple(float(item) for item in centered_rms.tolist())
        ),
        generator_manifest_sha256=cast(
            str, build_synthetic_generator_manifest()["manifest_sha256"]
        ),
    )
    return validate_normalization_artifact(artifact)


def load_normalization_artifact(serialized: str | bytes) -> NormalizationArtifact:
    parsed = _path_protocol.strict_canonical_json_loads(serialized)
    if not isinstance(parsed, dict) or set(parsed) != {"artifact", "artifact_sha256"}:
        raise ValueError("Normalization manifest has an invalid envelope.")
    payload = parsed.get("artifact")
    digest = parsed.get("artifact_sha256")
    if not isinstance(payload, dict) or not _is_sha256(digest):
        raise ValueError("Normalization manifest payload or hash is invalid.")
    expected_keys = {
        "centered_rms",
        "channel_count",
        "fit_sample_count",
        "fit_sample_ids",
        "fit_sample_ids_sha256",
        "generator_manifest_sha256",
        "mean",
        "protocol_id",
        "schema_version",
        "sequence_length",
        "statistic",
    }
    if set(payload) != expected_keys:
        raise ValueError("Normalization payload has an invalid key set.")
    if _path_protocol.canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Normalization artifact hash mismatch.")
    fixed_fields = {
        "channel_count": CHANNEL_COUNT,
        "protocol_id": NORMALIZATION_PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "statistic": _NORMALIZATION_STATISTIC,
    }
    if any(payload.get(key) != value for key, value in fixed_fields.items()):
        raise ValueError("Normalization payload violates the frozen protocol.")
    raw_ids = payload.get("fit_sample_ids")
    raw_mean = payload.get("mean")
    raw_rms = payload.get("centered_rms")
    if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
        raise ValueError("Normalization fit_sample_ids must be a string list.")
    if payload.get("fit_sample_count") != len(raw_ids):
        raise ValueError("Normalization fit_sample_count mismatch.")
    if not isinstance(raw_mean, list) or not isinstance(raw_rms, list):
        raise ValueError("Normalization mean/RMS must be lists.")
    if len(raw_mean) != CHANNEL_COUNT or len(raw_rms) != CHANNEL_COUNT:
        raise ValueError("Normalization mean/RMS length mismatch.")
    if not all(isinstance(item, float) for item in [*raw_mean, *raw_rms]):
        raise ValueError("Normalization mean/RMS entries must be JSON floats.")
    sample_hash = payload.get("fit_sample_ids_sha256")
    generator_hash = payload.get("generator_manifest_sha256")
    if not _is_sha256(sample_hash) or not _is_sha256(generator_hash):
        raise ValueError("Normalization lineage hashes are invalid.")
    artifact = NormalizationArtifact(
        fit_sample_ids=tuple(raw_ids),
        fit_sample_ids_sha256=cast(str, sample_hash),
        mean=cast(tuple[float, float], tuple(raw_mean)),
        centered_rms=cast(tuple[float, float], tuple(raw_rms)),
        generator_manifest_sha256=cast(str, generator_hash),
    )
    validate_normalization_artifact(artifact)
    if artifact.to_json() != (
        serialized.decode("utf-8") if isinstance(serialized, bytes) else serialized
    ):
        raise ValueError("Normalization artifact is not in canonical encoded form.")
    return artifact


def apply_normalization(
    x: torch.Tensor,
    artifact: NormalizationArtifact,
    *,
    expected_artifact_sha256: str,
) -> torch.Tensor:
    """Apply pinned fitted statistics, rejecting hash, shape, dtype, or finite failures."""

    tensor = _validate_blc_tensor(x, name="normalization input")
    checked = validate_normalization_artifact(artifact)
    if not _is_sha256(expected_artifact_sha256):
        raise ValueError("expected_artifact_sha256 must be a lowercase SHA-256 digest.")
    if checked.artifact_sha256 != expected_artifact_sha256:
        raise ValueError("Pinned normalization artifact hash mismatch.")
    mean = torch.tensor(checked.mean, dtype=tensor.dtype, device=tensor.device)
    rms = torch.tensor(checked.centered_rms, dtype=tensor.dtype, device=tensor.device)
    output = (tensor - mean) / rms
    return _validate_blc_tensor(output, name="normalized output", batch_size=int(tensor.shape[0]))


def _nuisance_cell_payload(
    snr_db: int | None, scale: float, circular_shift: int
) -> dict[str, JsonValue]:
    return {
        "circular_shift": circular_shift,
        "operation_order": list(NUISANCE_ORDER),
        "protocol_id": NUISANCE_PROTOCOL_ID,
        "scale": scale,
        "schema_version": SCHEMA_VERSION,
        "snr_db": "inf" if snr_db is None else snr_db,
    }


@dataclass(frozen=True, slots=True)
class NuisanceCell:
    """One member of the frozen 3 x 3 x 3 nuisance Cartesian product."""

    cell_id: str
    cell_sha256: str
    snr_db: int | None
    scale: float
    circular_shift: int

    def payload(self) -> dict[str, JsonValue]:
        return _nuisance_cell_payload(self.snr_db, self.scale, self.circular_shift)

    def manifest_record(self) -> dict[str, JsonValue]:
        return {
            **self.payload(),
            "cell_id": self.cell_id,
            "cell_sha256": self.cell_sha256,
        }


def _make_nuisance_cell(
    snr_db: int | None, scale: float, circular_shift: int
) -> NuisanceCell:
    payload = _nuisance_cell_payload(snr_db, scale, circular_shift)
    digest = _path_protocol.canonical_json_sha256(payload)
    return NuisanceCell(
        cell_id=f"{_NUISANCE_CELL_PREFIX}{digest}",
        cell_sha256=digest,
        snr_db=snr_db,
        scale=scale,
        circular_shift=circular_shift,
    )


NUISANCE_CELLS: Final[tuple[NuisanceCell, ...]] = tuple(
    _make_nuisance_cell(snr_db, scale, circular_shift)
    for snr_db in SNR_LEVELS_DB
    for scale in SCALE_LEVELS
    for circular_shift in CIRCULAR_SHIFTS
)


def validate_nuisance_cell(cell: Any) -> NuisanceCell:
    if not isinstance(cell, NuisanceCell):
        raise TypeError("cell must be a NuisanceCell.")
    expected = {
        item.cell_id: item for item in NUISANCE_CELLS
    }.get(cell.cell_id)
    if expected is None or cell != expected:
        raise ValueError("Nuisance cell is not an exact member of the frozen 27-cell grid.")
    if cell.cell_sha256 != _path_protocol.canonical_json_sha256(cell.payload()):
        raise ValueError("Nuisance cell hash mismatch.")
    return cell


def build_nuisance_manifest() -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "cells": [item.manifest_record() for item in NUISANCE_CELLS],
        "circular_shifts": list(CIRCULAR_SHIFTS),
        "finite_snr_noise": {
            "calibration": (
                "per-sample, per-channel centered-noise RMS equals post-scale/shift "
                "signal RMS times 10^(-snr_db/20)"
            ),
            "distribution": "torch.randn CPU float64",
            "seed_key": "(sample_id,'synthetic-nuisance-noise',cell_id)",
        },
        "operation_order": list(NUISANCE_ORDER),
        "protocol_id": NUISANCE_PROTOCOL_ID,
        "scale_levels": list(SCALE_LEVELS),
        "schema_version": SCHEMA_VERSION,
        "snr_levels_db": ["inf" if item is None else item for item in SNR_LEVELS_DB],
    }
    return {**payload, "manifest_sha256": _path_protocol.canonical_json_sha256(payload)}


def validate_nuisance_manifest(manifest: Any) -> dict[str, JsonValue]:
    if not isinstance(manifest, dict):
        raise TypeError("Nuisance manifest must be a JSON object.")
    expected = build_nuisance_manifest()
    if set(manifest) != set(expected):
        raise ValueError("Nuisance manifest has an invalid key set.")
    digest = manifest.get("manifest_sha256")
    if not _is_sha256(digest):
        raise ValueError("Nuisance manifest hash is invalid.")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if _path_protocol.canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Nuisance manifest hash mismatch.")
    if _path_protocol.canonical_json_bytes(cast(JsonValue, manifest)) != (
        _path_protocol.canonical_json_bytes(expected)
    ):
        raise ValueError("Nuisance manifest does not match the frozen grid.")
    return cast(dict[str, JsonValue], manifest)


def nuisance_manifest_json() -> str:
    return _path_protocol.canonical_json_bytes(build_nuisance_manifest()).decode("utf-8")


def load_nuisance_manifest(serialized: str | bytes) -> dict[str, JsonValue]:
    return validate_nuisance_manifest(_path_protocol.strict_canonical_json_loads(serialized))


def apply_nuisance(
    normalized: torch.Tensor,
    sample_ids: Sequence[str],
    cell: NuisanceCell,
) -> torch.Tensor:
    """Apply scale, shift, and optional calibrated noise to normalized BLC input."""

    tensor = _validate_blc_tensor(normalized, name="nuisance input")
    identifiers = _validate_sample_ids(sample_ids, expected_count=int(tensor.shape[0]))
    checked_cell = validate_nuisance_cell(cell)
    transformed = torch.roll(
        tensor * checked_cell.scale, shifts=checked_cell.circular_shift, dims=1
    )
    if checked_cell.snr_db is None:
        return _validate_blc_tensor(
            transformed, name="nuisance output", batch_size=len(identifiers)
        )

    output_samples = []
    attenuation = 10.0 ** (-float(checked_cell.snr_db) / 20.0)
    for index, sample_id in enumerate(identifiers):
        signal64 = transformed[index].to(dtype=torch.float64)
        signal_rms = signal64.square().mean(dim=0).sqrt()
        if bool((signal_rms <= torch.finfo(torch.float64).eps).any()):
            raise ValueError("Finite-SNR nuisance requires positive signal RMS per channel.")
        generator = _cpu_generator(
            sample_id, "synthetic-nuisance-noise", checked_cell.cell_id
        )
        raw_noise = torch.randn(
            (SEQUENCE_LENGTH, CHANNEL_COUNT), generator=generator, dtype=torch.float64
        ).to(device=signal64.device)
        raw_noise = raw_noise - raw_noise.mean(dim=0)
        raw_rms = raw_noise.square().mean(dim=0).sqrt()
        if bool((raw_rms <= torch.finfo(torch.float64).eps).any()):
            raise RuntimeError("Generated nuisance noise unexpectedly has zero RMS.")
        noise = raw_noise / raw_rms * (signal_rms * attenuation)
        output_samples.append((signal64 + noise).to(dtype=tensor.dtype, device=tensor.device))
    output = torch.stack(output_samples, dim=0)
    return _validate_blc_tensor(output, name="nuisance output", batch_size=len(identifiers))


def _synthetic_generator_payload() -> dict[str, JsonValue]:
    path_manifest = _path_protocol.build_path_universe_manifest()
    seed_manifest = _path_protocol.build_seed_namespace_manifest()
    nuisance_manifest = build_nuisance_manifest()
    return {
        "components": {
            "ar1": {
                "center_each_component": True,
                "component_rms": AR1_COMPONENT_RMS,
                "initial_state": "standard_normal_innovation[0]",
                "rho_range_half_open": [AR1_RHO_MIN, AR1_RHO_MAX],
                "seed_purpose": "synthetic-ar1",
                "stationary_innovation_scale": "sqrt(1-rho^2)",
            },
            "damped_impulses": {
                "count_inclusive": [IMPULSE_COUNT_MIN, IMPULSE_COUNT_MAX],
                "decay_samples_range_half_open": [
                    IMPULSE_DECAY_MIN_SAMPLES,
                    IMPULSE_DECAY_MAX_SAMPLES,
                ],
                "magnitude_range_half_open": [
                    IMPULSE_MAGNITUDE_MIN,
                    IMPULSE_MAGNITUDE_MAX,
                ],
                "positions": "unique uniform indices without replacement",
                "seed_purpose": "synthetic-damped-impulses",
                "signs": "independent equiprobable {-1,+1}",
            },
            "multisine": {
                "amplitude_range_half_open": [
                    MULTISINE_AMPLITUDE_MIN,
                    MULTISINE_AMPLITUDE_MAX,
                ],
                "component_count": MULTISINE_COMPONENTS,
                "frequency_construction": (
                    "sort three U[0,1) values; f_i=f_min+i*minimum_separation+"
                    "(f_max-f_min-2*minimum_separation)*u_i"
                ),
                "frequency_cycles_per_sample_range_half_open": [
                    MULTISINE_FREQUENCY_MIN,
                    MULTISINE_FREQUENCY_MAX,
                ],
                "minimum_frequency_separation": MULTISINE_FREQUENCY_SEPARATION,
                "phase_range": "[-pi,pi)",
                "seed_purpose": "synthetic-multisine",
            },
            "sum_order": ["multisine", "damped_impulses", "ar1"],
        },
        "nuisance_manifest_sha256": cast(str, nuisance_manifest["manifest_sha256"]),
        "normalization": {
            "fit_namespace_only": True,
            "protocol_id": NORMALIZATION_PROTOCOL_ID,
            "statistic": _NORMALIZATION_STATISTIC,
        },
        "path_universe_sha256": cast(str, path_manifest["manifest_sha256"]),
        "protocol_id": PROTOCOL_ID,
        "sample_seed_derivation": (
            "path_universe.derive_sample_seed(sample_id,purpose,*components)"
        ),
        "stochastic_backend": {
            "device": "cpu",
            "generator": "torch.Generator",
            "parameter_and_innovation_dtype": "float64",
            "per_component_key": "(sample_id,seed_purpose,channel)",
        },
        "schema_version": SCHEMA_VERSION,
        "seed_namespace_sha256": cast(str, seed_manifest["manifest_sha256"]),
        "target": {
            "binding": "target=path_universe.oracle_execute_path(nuisance_input,raw_path)",
            "oracle": "src.utils.p07_protocol.path_universe.oracle_execute_path",
            "raw_path_stages": _path_protocol.K_STAGES,
        },
        "tensor_contract": {
            "channel_count": CHANNEL_COUNT,
            "generation_compute_dtype": "float64",
            "layout": TENSOR_LAYOUT,
            "output_dtypes": ["float32", "float64"],
            "sequence_length": SEQUENCE_LENGTH,
        },
    }


def build_synthetic_generator_manifest() -> dict[str, JsonValue]:
    payload = _synthetic_generator_payload()
    return {**payload, "manifest_sha256": _path_protocol.canonical_json_sha256(payload)}


def validate_synthetic_generator_manifest(manifest: Any) -> dict[str, JsonValue]:
    if not isinstance(manifest, dict):
        raise TypeError("Synthetic generator manifest must be a JSON object.")
    expected = build_synthetic_generator_manifest()
    if set(manifest) != set(expected):
        raise ValueError("Synthetic generator manifest has an invalid key set.")
    digest = manifest.get("manifest_sha256")
    if not _is_sha256(digest):
        raise ValueError("Synthetic generator manifest hash is invalid.")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if _path_protocol.canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Synthetic generator manifest hash mismatch.")
    if _path_protocol.canonical_json_bytes(cast(JsonValue, manifest)) != (
        _path_protocol.canonical_json_bytes(expected)
    ):
        raise ValueError("Synthetic generator manifest does not match the frozen protocol.")
    return cast(dict[str, JsonValue], manifest)


def synthetic_generator_manifest_json() -> str:
    return _path_protocol.canonical_json_bytes(
        build_synthetic_generator_manifest()
    ).decode("utf-8")


def load_synthetic_generator_manifest(serialized: str | bytes) -> dict[str, JsonValue]:
    return validate_synthetic_generator_manifest(
        _path_protocol.strict_canonical_json_loads(serialized)
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    checked = _validate_blc_tensor(tensor, name="tensor hash input")
    detached = checked.detach().cpu().contiguous()
    header = _path_protocol.canonical_json_bytes(
        {
            "dtype": str(detached.dtype).removeprefix("torch."),
            "shape": list(detached.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticEpisode:
    """One generated input/target episode plus immutable protocol references."""

    sample_ids: tuple[str, ...]
    raw_path: tuple[str, str, str]
    nuisance_cell_id: str
    normalization_artifact_sha256: str
    input: torch.Tensor
    target: torch.Tensor


def _validate_episode(episode: Any, *, verify_oracle: bool = True) -> SyntheticEpisode:
    if not isinstance(episode, SyntheticEpisode):
        raise TypeError("episode must be a SyntheticEpisode.")
    identifiers = _validate_sample_ids(
        episode.sample_ids, expected_count=int(episode.input.shape[0])
        if isinstance(episode.input, torch.Tensor) and episode.input.ndim > 0
        else None
    )
    input_tensor = _validate_blc_tensor(
        episode.input, name="episode input", batch_size=len(identifiers)
    )
    target = _validate_blc_tensor(
        episode.target, name="episode target", batch_size=len(identifiers)
    )
    if target.dtype != input_tensor.dtype or target.device != input_tensor.device:
        raise ValueError("Episode input and target must share dtype and device.")
    raw_path = _path_protocol.validate_raw_path(episode.raw_path)
    if tuple(raw_path) != episode.raw_path:
        raise ValueError("Episode raw_path must be the canonical validated tuple representation.")
    if episode.nuisance_cell_id not in {item.cell_id for item in NUISANCE_CELLS}:
        raise ValueError("Episode references an unknown nuisance cell.")
    if not _is_sha256(episode.normalization_artifact_sha256):
        raise ValueError("Episode normalization hash is invalid.")
    if verify_oracle:
        expected_target = _path_protocol.oracle_execute_path(input_tensor, raw_path)
        if not torch.equal(target, expected_target):
            raise ValueError("Episode target is not exactly bound to the independent oracle.")
    return episode


def generate_synthetic_episode(
    sample_ids: Sequence[str],
    raw_path: Sequence[str],
    normalization_artifact: NormalizationArtifact,
    nuisance_cell: NuisanceCell,
    *,
    expected_normalization_sha256: str,
    dtype: torch.dtype = torch.float64,
) -> SyntheticEpisode:
    """Generate normalized/nuisance input and its independently-oracled target."""

    identifiers = _validate_sample_ids(sample_ids)
    path = _path_protocol.validate_raw_path(raw_path)
    checked_cell = validate_nuisance_cell(nuisance_cell)
    roots = generate_root_batch(identifiers, dtype=_require_dtype(dtype))
    normalized = apply_normalization(
        roots,
        normalization_artifact,
        expected_artifact_sha256=expected_normalization_sha256,
    )
    nuisance_input = apply_nuisance(normalized, identifiers, checked_cell)
    target = _path_protocol.oracle_execute_path(nuisance_input, path)
    episode = SyntheticEpisode(
        sample_ids=identifiers,
        raw_path=cast(tuple[str, str, str], tuple(path)),
        nuisance_cell_id=checked_cell.cell_id,
        normalization_artifact_sha256=expected_normalization_sha256,
        input=nuisance_input,
        target=target,
    )
    return _validate_episode(episode)


def build_episode_manifest(episode: SyntheticEpisode) -> dict[str, JsonValue]:
    checked = _validate_episode(episode)
    payload: dict[str, JsonValue] = {
        "batch_size": len(checked.sample_ids),
        "dtype": str(checked.input.dtype).removeprefix("torch."),
        "generator_manifest_sha256": cast(
            str, build_synthetic_generator_manifest()["manifest_sha256"]
        ),
        "input_sha256": _tensor_sha256(checked.input),
        "normalization_artifact_sha256": checked.normalization_artifact_sha256,
        "nuisance_cell_id": checked.nuisance_cell_id,
        "protocol_id": EPISODE_PROTOCOL_ID,
        "raw_path": list(checked.raw_path),
        "sample_ids": list(checked.sample_ids),
        "sample_ids_sha256": _path_protocol.canonical_json_sha256(
            {
                "kind": "p07_episode_sample_ids",
                "sample_ids": list(checked.sample_ids),
            }
        ),
        "schema_version": SCHEMA_VERSION,
        "shape": list(checked.input.shape),
        "target_oracle": "src.utils.p07_protocol.path_universe.oracle_execute_path",
        "target_sha256": _tensor_sha256(checked.target),
    }
    return {**payload, "manifest_sha256": _path_protocol.canonical_json_sha256(payload)}


def validate_episode_manifest(
    manifest: Any, episode: SyntheticEpisode
) -> dict[str, JsonValue]:
    if not isinstance(manifest, dict):
        raise TypeError("Episode manifest must be a JSON object.")
    expected = build_episode_manifest(episode)
    if set(manifest) != set(expected):
        raise ValueError("Episode manifest has an invalid key set.")
    digest = manifest.get("manifest_sha256")
    if not _is_sha256(digest):
        raise ValueError("Episode manifest hash is invalid.")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if _path_protocol.canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Episode manifest hash mismatch.")
    if _path_protocol.canonical_json_bytes(cast(JsonValue, manifest)) != (
        _path_protocol.canonical_json_bytes(expected)
    ):
        raise ValueError("Episode manifest does not match the supplied episode.")
    return cast(dict[str, JsonValue], manifest)


__all__ = [
    "ALLOWED_DTYPES",
    "CHANNEL_COUNT",
    "CIRCULAR_SHIFTS",
    "EPISODE_PROTOCOL_ID",
    "NORMALIZATION_PROTOCOL_ID",
    "NUISANCE_CELLS",
    "NUISANCE_ORDER",
    "NUISANCE_PROTOCOL_ID",
    "NuisanceCell",
    "NormalizationArtifact",
    "PROTOCOL_ID",
    "SCALE_LEVELS",
    "SEQUENCE_LENGTH",
    "SNR_LEVELS_DB",
    "SyntheticEpisode",
    "TENSOR_LAYOUT",
    "apply_normalization",
    "apply_nuisance",
    "build_episode_manifest",
    "build_nuisance_manifest",
    "build_synthetic_generator_manifest",
    "estimate_normalization_artifact",
    "generate_root_batch",
    "generate_root_signal",
    "generate_synthetic_episode",
    "load_normalization_artifact",
    "load_nuisance_manifest",
    "load_synthetic_generator_manifest",
    "nuisance_manifest_json",
    "synthetic_generator_manifest_json",
    "validate_episode_manifest",
    "validate_normalization_artifact",
    "validate_nuisance_cell",
    "validate_nuisance_manifest",
    "validate_synthetic_generator_manifest",
]

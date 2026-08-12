"""Versioned measurement objects for explainable fault-diagnosis outputs.

The object identity is content based. Artifact locators are retained for audit
and intentionally excluded from the identity so moving an unchanged artifact
does not create a scientifically different measurement.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


MEASUREMENT_SCHEMA_VERSION = "p02.measurement.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MeasurementContractError(ValueError):
    """Raised when an object violates the frozen measurement contract."""


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping using the contract's canonical JSON representation."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MeasurementContractError(f"payload is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash numeric array content after deterministic C-order normalization."""

    array = np.asarray(values)
    if array.size == 0:
        raise MeasurementContractError("artifact arrays must not be empty")
    if array.dtype.hasobject:
        raise MeasurementContractError("object-dtype arrays are not admissible")
    if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
        raise MeasurementContractError("artifact arrays must contain only finite values")

    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _required_text(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise MeasurementContractError(f"{field} must be a non-empty string")
    return text


def _sha256(value: Any, field: str) -> str:
    digest = str(value).strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    if not _SHA256_RE.fullmatch(digest):
        raise MeasurementContractError(f"{field} must be a 64-character SHA-256 digest")
    return digest


def _exact_keys(payload: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise MeasurementContractError(f"{field} keys mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class ArtifactRef:
    """A typed, content-addressed artifact used by one measurement."""

    role: str
    locator: str
    sha256: str
    media_type: str
    dtype: str
    shape: Tuple[int, ...]
    axes: Tuple[str, ...]
    axis_units: Tuple[str, ...]
    coordinate_map_sha256: str
    output_kind: str
    value_semantics: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _required_text(self.role, "artifact.role"))
        object.__setattr__(self, "locator", _required_text(self.locator, "artifact.locator"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "artifact.sha256"))
        object.__setattr__(self, "media_type", _required_text(self.media_type, "artifact.media_type"))
        object.__setattr__(self, "dtype", _required_text(self.dtype, "artifact.dtype"))
        object.__setattr__(self, "output_kind", _required_text(self.output_kind, "artifact.output_kind"))
        object.__setattr__(
            self, "value_semantics", _required_text(self.value_semantics, "artifact.value_semantics")
        )
        object.__setattr__(
            self,
            "coordinate_map_sha256",
            _sha256(self.coordinate_map_sha256, "artifact.coordinate_map_sha256"),
        )
        if not self.shape or any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise MeasurementContractError("artifact.shape must contain positive integers")
        if len(self.axes) != len(self.shape):
            raise MeasurementContractError("artifact.axes length must equal artifact.shape rank")
        normalized_axes = tuple(_required_text(axis, "artifact.axes[]") for axis in self.axes)
        if len(set(normalized_axes)) != len(normalized_axes):
            raise MeasurementContractError("artifact.axes must be unique")
        object.__setattr__(self, "axes", normalized_axes)
        normalized_units = tuple(
            _required_text(unit, "artifact.axis_units[]") for unit in self.axis_units
        )
        if len(normalized_units) != len(normalized_axes):
            raise MeasurementContractError("artifact.axis_units length must equal artifact.axes length")
        object.__setattr__(self, "axis_units", normalized_units)

    @classmethod
    def from_array(
        cls,
        *,
        role: str,
        locator: str,
        values: np.ndarray,
        axes: Sequence[str],
        axis_units: Sequence[str],
        coordinate_map_sha256: str,
        output_kind: str,
        value_semantics: str,
    ) -> "ArtifactRef":
        array = np.asarray(values)
        return cls(
            role=role,
            locator=locator,
            sha256=array_sha256(array),
            media_type="application/x-npy",
            dtype=array.dtype.str,
            shape=tuple(int(size) for size in array.shape),
            axes=tuple(axes),
            axis_units=tuple(axis_units),
            coordinate_map_sha256=coordinate_map_sha256,
            output_kind=output_kind,
            value_semantics=value_semantics,
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "axis_units": list(self.axis_units),
            "coordinate_map_sha256": self.coordinate_map_sha256,
            "output_kind": self.output_kind,
            "value_semantics": self.value_semantics,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "locator": self.locator}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        expected = {
            "role",
            "locator",
            "sha256",
            "media_type",
            "dtype",
            "shape",
            "axes",
            "axis_units",
            "coordinate_map_sha256",
            "output_kind",
            "value_semantics",
        }
        _exact_keys(payload, expected, "artifact")
        return cls(
            role=payload["role"],
            locator=payload["locator"],
            sha256=payload["sha256"],
            media_type=payload["media_type"],
            dtype=payload["dtype"],
            shape=tuple(payload["shape"]),
            axes=tuple(payload["axes"]),
            axis_units=tuple(payload["axis_units"]),
            coordinate_map_sha256=payload["coordinate_map_sha256"],
            output_kind=payload["output_kind"],
            value_semantics=payload["value_semantics"],
        )


@dataclass(frozen=True)
class SourceIdentity:
    """Identity and provenance of a fixed sibling-model output."""

    paper_id: str
    method_id: str
    model_id: str
    dataset_id: str
    split_id: str
    seed: int
    hardware_id: str
    score_id: str
    source_artifact_sha256: str
    model_artifact_sha256: str
    config_sha256: str
    environment_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "paper_id",
            "method_id",
            "model_id",
            "dataset_id",
            "split_id",
            "hardware_id",
            "score_id",
        ):
            object.__setattr__(self, field, _required_text(getattr(self, field), f"source.{field}"))
        if not isinstance(self.seed, int) or self.seed < 0:
            raise MeasurementContractError("source.seed must be a non-negative integer")
        for field in (
            "source_artifact_sha256",
            "model_artifact_sha256",
            "config_sha256",
            "environment_sha256",
            "code_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), f"source.{field}"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "method_id": self.method_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "seed": self.seed,
            "hardware_id": self.hardware_id,
            "score_id": self.score_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "config_sha256": self.config_sha256,
            "environment_sha256": self.environment_sha256,
            "code_sha256": self.code_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceIdentity":
        expected = {
            "paper_id",
            "method_id",
            "model_id",
            "dataset_id",
            "split_id",
            "seed",
            "hardware_id",
            "score_id",
            "source_artifact_sha256",
            "model_artifact_sha256",
            "config_sha256",
            "environment_sha256",
            "code_sha256",
        }
        _exact_keys(payload, expected, "source")
        return cls(**dict(payload))


@dataclass(frozen=True)
class AdapterIdentity:
    """Versioned deterministic conversion from one source output kind."""

    adapter_id: str
    adapter_version: str
    adapter_sha256: str
    input_kind: str
    output_kind: str
    capabilities: Tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("adapter_id", "adapter_version", "input_kind", "output_kind"):
            object.__setattr__(self, field, _required_text(getattr(self, field), f"adapter.{field}"))
        object.__setattr__(self, "adapter_sha256", _sha256(self.adapter_sha256, "adapter.adapter_sha256"))
        capabilities = tuple(_required_text(item, "adapter.capabilities[]") for item in self.capabilities)
        if not capabilities or len(set(capabilities)) != len(capabilities):
            raise MeasurementContractError("adapter.capabilities must be non-empty and unique")
        object.__setattr__(self, "capabilities", tuple(sorted(capabilities)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "adapter_sha256": self.adapter_sha256,
            "input_kind": self.input_kind,
            "output_kind": self.output_kind,
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterIdentity":
        expected = {
            "adapter_id",
            "adapter_version",
            "adapter_sha256",
            "input_kind",
            "output_kind",
            "capabilities",
        }
        _exact_keys(payload, expected, "adapter")
        values = dict(payload)
        values["capabilities"] = tuple(values["capabilities"])
        return cls(**values)


@dataclass(frozen=True)
class MeasurementObject:
    """Immutable tuple binding one explanation to its scientific identity."""

    schema_version: str
    source: SourceIdentity
    adapter: AdapterIdentity
    explanation: ArtifactRef
    supporting_artifacts: Tuple[ArtifactRef, ...]
    sample_ids_sha256: str
    target_id: str
    measurement_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != MEASUREMENT_SCHEMA_VERSION:
            raise MeasurementContractError(
                f"unsupported schema_version={self.schema_version!r}; expected {MEASUREMENT_SCHEMA_VERSION!r}"
            )
        if self.explanation.role != "explanation":
            raise MeasurementContractError("measurement.explanation must have role='explanation'")
        if self.explanation.output_kind != self.adapter.output_kind:
            raise MeasurementContractError("adapter.output_kind must equal explanation.output_kind")
        if not self.explanation.axes or self.explanation.axes[0] != "sample":
            raise MeasurementContractError("explanation first axis must be 'sample'")
        object.__setattr__(self, "target_id", _required_text(self.target_id, "target_id"))

        artifacts = (self.explanation, *self.supporting_artifacts)
        roles = [artifact.role for artifact in artifacts]
        if len(set(roles)) != len(roles):
            raise MeasurementContractError("artifact roles must be unique within one measurement")
        object.__setattr__(self, "supporting_artifacts", tuple(self.supporting_artifacts))
        object.__setattr__(self, "sample_ids_sha256", _sha256(self.sample_ids_sha256, "sample_ids_sha256"))

        computed = canonical_sha256(self.identity_payload())
        if self.measurement_id and _sha256(self.measurement_id, "measurement_id") != computed:
            raise MeasurementContractError("measurement_id does not match canonical object content")
        object.__setattr__(self, "measurement_id", computed)

    def identity_payload(self) -> dict[str, Any]:
        supporting = sorted(
            (artifact.identity_dict() for artifact in self.supporting_artifacts),
            key=lambda item: item["role"],
        )
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "adapter": self.adapter.to_dict(),
            "explanation": self.explanation.identity_dict(),
            "supporting_artifacts": supporting,
            "sample_ids_sha256": self.sample_ids_sha256,
            "target_id": self.target_id,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "measurement_id": self.measurement_id,
            "source": self.source.to_dict(),
            "adapter": self.adapter.to_dict(),
            "explanation": self.explanation.to_dict(),
            "supporting_artifacts": [
                artifact.to_dict() for artifact in sorted(self.supporting_artifacts, key=lambda item: item.role)
            ],
            "sample_ids_sha256": self.sample_ids_sha256,
            "target_id": self.target_id,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "MeasurementObject":
        expected = {
            "schema_version",
            "measurement_id",
            "source",
            "adapter",
            "explanation",
            "supporting_artifacts",
            "sample_ids_sha256",
            "target_id",
        }
        _exact_keys(payload, expected, "measurement")
        return cls(
            schema_version=payload["schema_version"],
            measurement_id=payload["measurement_id"],
            source=SourceIdentity.from_dict(payload["source"]),
            adapter=AdapterIdentity.from_dict(payload["adapter"]),
            explanation=ArtifactRef.from_dict(payload["explanation"]),
            supporting_artifacts=tuple(
                ArtifactRef.from_dict(item) for item in payload["supporting_artifacts"]
            ),
            sample_ids_sha256=payload["sample_ids_sha256"],
            target_id=payload["target_id"],
        )


class ArrayMeasurementAdapter:
    """Reference adapter for deterministic numeric sibling outputs."""

    def __init__(self, identity: AdapterIdentity):
        self.identity = identity

    def adapt(
        self,
        *,
        values: np.ndarray,
        sample_ids: Sequence[str],
        axes: Sequence[str],
        axis_units: Sequence[str],
        coordinate_map_sha256: str,
        value_semantics: str,
        locator: str,
        source: SourceIdentity,
        target_id: str,
        supporting_artifacts: Sequence[ArtifactRef] = (),
    ) -> MeasurementObject:
        array = np.asarray(values)
        normalized_axes = tuple(axes)
        if not normalized_axes or normalized_axes[0] != "sample":
            raise MeasurementContractError("adapter axes must begin with 'sample'")
        normalized_ids = tuple(_required_text(item, "sample_ids[]") for item in sample_ids)
        if len(normalized_ids) != array.shape[0]:
            raise MeasurementContractError("sample_ids length must equal explanation sample dimension")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise MeasurementContractError("sample_ids must be unique")

        explanation = ArtifactRef.from_array(
            role="explanation",
            locator=locator,
            values=array,
            axes=normalized_axes,
            axis_units=tuple(axis_units),
            coordinate_map_sha256=coordinate_map_sha256,
            output_kind=self.identity.output_kind,
            value_semantics=value_semantics,
        )
        return MeasurementObject(
            schema_version=MEASUREMENT_SCHEMA_VERSION,
            source=source,
            adapter=self.identity,
            explanation=explanation,
            supporting_artifacts=tuple(supporting_artifacts),
            sample_ids_sha256=canonical_sha256({"sample_ids": list(normalized_ids)}),
            target_id=target_id,
        )


def assert_measurements_aligned(first: MeasurementObject, second: MeasurementObject) -> None:
    """Require explicit identity and coordinate agreement before paired metrics."""

    checks = {
        "sample_ids_sha256": (first.sample_ids_sha256, second.sample_ids_sha256),
        "target_id": (first.target_id, second.target_id),
        "output_kind": (first.explanation.output_kind, second.explanation.output_kind),
        "shape": (first.explanation.shape, second.explanation.shape),
        "axes": (first.explanation.axes, second.explanation.axes),
        "axis_units": (first.explanation.axis_units, second.explanation.axis_units),
        "coordinate_map_sha256": (
            first.explanation.coordinate_map_sha256,
            second.explanation.coordinate_map_sha256,
        ),
        "value_semantics": (
            first.explanation.value_semantics,
            second.explanation.value_semantics,
        ),
    }
    mismatches = sorted(name for name, (left, right) in checks.items() if left != right)
    if mismatches:
        raise MeasurementContractError(f"paired measurements are not aligned: {mismatches}")

"""Typed, provider-neutral explanation state for interpretable PHM models.

The state is deliberately smaller than a model dump. It retains only public,
model-native quantities that an LLM may verbalize: prediction, typed evidence,
active paths, class contributions, uncertainty, operating conditions, and
explicitly supplied mechanism relations. It never infers a physical mechanism
from a tensor name or an attention weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "phm-eir/v1"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_optional(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_value(value: Any, name: str) -> JsonValue:
    """Convert a public value to a JSON-safe object without importing Torch."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {
            _nonempty_text(key, f"{name} key"): _json_value(item, f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{name}[]") for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item(), name)
        except (TypeError, ValueError, RuntimeError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist(), name)

    raise TypeError(f"{name} must be JSON serializable, got {type(value).__name__}")


def freeze_mapping(
    values: Mapping[str, Any] | None,
    *,
    name: str,
) -> tuple[tuple[str, JsonValue], ...]:
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return tuple(
        (str(key), _json_value(value, f"{name}.{key}"))
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    )


def thaw_mapping(values: Sequence[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    return {key: value for key, value in values}


@dataclass(frozen=True)
class PredictionState:
    """Public prediction and its declared numerical support."""

    label: str
    class_index: int | None = None
    confidence: float | None = None
    logits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _nonempty_text(self.label, "prediction.label"))
        if self.class_index is not None:
            if isinstance(self.class_index, bool) or int(self.class_index) < 0:
                raise ValueError("prediction.class_index must be a non-negative integer")
            object.__setattr__(self, "class_index", int(self.class_index))
        confidence = _finite_optional(self.confidence, "prediction.confidence")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("prediction.confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        normalized_logits = tuple(
            _finite_optional(value, "prediction.logits") for value in self.logits
        )
        object.__setattr__(self, "logits", tuple(float(v) for v in normalized_logits))
        if self.class_index is not None and self.logits:
            if self.class_index >= len(self.logits):
                raise ValueError("prediction.class_index is outside prediction.logits")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "label": self.label,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "logits": list(self.logits),
        }


@dataclass(frozen=True)
class EvidenceAtom:
    """One typed, directly exported model quantity."""

    id: str
    kind: str
    name: str
    value: JsonValue = None
    unit: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _nonempty_text(self.id, "evidence.id"))
        object.__setattr__(self, "kind", _nonempty_text(self.kind, "evidence.kind"))
        object.__setattr__(self, "name", _nonempty_text(self.name, "evidence.name"))
        object.__setattr__(self, "value", _json_value(self.value, f"evidence[{self.id}].value"))
        if self.unit is not None:
            object.__setattr__(self, "unit", _nonempty_text(self.unit, "evidence.unit"))
        if self.source is not None:
            object.__setattr__(self, "source", _nonempty_text(self.source, "evidence.source"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
        }


@dataclass(frozen=True)
class EvidencePath:
    """An ordered or grouped path through model-native evidence atoms."""

    id: str
    atom_ids: tuple[str, ...]
    relation: str
    score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _nonempty_text(self.id, "path.id"))
        if not self.atom_ids:
            raise ValueError("path.atom_ids must not be empty")
        normalized_ids = tuple(_nonempty_text(item, "path.atom_ids[]") for item in self.atom_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"path {self.id!r} contains duplicate atom IDs")
        object.__setattr__(self, "atom_ids", normalized_ids)
        object.__setattr__(self, "relation", _nonempty_text(self.relation, "path.relation"))
        object.__setattr__(self, "score", _finite_optional(self.score, "path.score"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "atom_ids": list(self.atom_ids),
            "relation": self.relation,
            "score": self.score,
        }


@dataclass(frozen=True)
class ClassContribution:
    """A same-forward contribution exported by the interpretable model."""

    source_id: str
    target_label: str
    value: float
    kind: str = "logit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _nonempty_text(self.source_id, "contribution.source_id"))
        object.__setattr__(self, "target_label", _nonempty_text(self.target_label, "contribution.target_label"))
        object.__setattr__(self, "kind", _nonempty_text(self.kind, "contribution.kind"))
        value = _finite_optional(self.value, "contribution.value")
        assert value is not None
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source_id": self.source_id,
            "target_label": self.target_label,
            "value": self.value,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class MechanismRelation:
    """A model-native or externally mechanism-constrained relation."""

    id: str
    source_ids: tuple[str, ...]
    predicate: str
    target_claim: str
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _nonempty_text(self.id, "relation.id"))
        if not self.source_ids:
            raise ValueError("relation.source_ids must not be empty")
        object.__setattr__(
            self,
            "source_ids",
            tuple(_nonempty_text(item, "relation.source_ids[]") for item in self.source_ids),
        )
        object.__setattr__(self, "predicate", _nonempty_text(self.predicate, "relation.predicate"))
        object.__setattr__(self, "target_claim", _nonempty_text(self.target_claim, "relation.target_claim"))
        allowed = {"model-native", "mechanism-constrained", "hypothesis"}
        status = _nonempty_text(self.status, "relation.status")
        if status not in allowed:
            raise ValueError(f"relation.status must be one of {sorted(allowed)}")
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "source_ids": list(self.source_ids),
            "predicate": self.predicate,
            "target_claim": self.target_claim,
            "status": self.status,
        }


@dataclass(frozen=True)
class UncertaintyState:
    """Named uncertainty and trace-fidelity quantities."""

    metrics: tuple[tuple[str, float], ...] = ()
    calibration_state: str = "not_provided"

    def __post_init__(self) -> None:
        seen: set[str] = set()
        normalized: list[tuple[str, float]] = []
        for name, value in self.metrics:
            clean_name = _nonempty_text(name, "uncertainty.metrics name")
            if clean_name in seen:
                raise ValueError(f"duplicate uncertainty metric: {clean_name}")
            seen.add(clean_name)
            normalized_value = _finite_optional(value, f"uncertainty.{clean_name}")
            assert normalized_value is not None
            normalized.append((clean_name, normalized_value))
        object.__setattr__(self, "metrics", tuple(normalized))
        object.__setattr__(
            self,
            "calibration_state",
            _nonempty_text(self.calibration_state, "uncertainty.calibration_state"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "metrics": {name: value for name, value in self.metrics},
            "calibration_state": self.calibration_state,
        }


@dataclass(frozen=True)
class PHMExplanationState:
    """PHM Explanation Intermediate Representation (PHM-EIR)."""

    sample_id: str
    task: str
    model_family: str
    trace_kind: str
    prediction: PredictionState
    evidence_atoms: tuple[EvidenceAtom, ...]
    evidence_paths: tuple[EvidencePath, ...] = ()
    contributions: tuple[ClassContribution, ...] = ()
    mechanism_relations: tuple[MechanismRelation, ...] = ()
    uncertainty: UncertaintyState = field(default_factory=UncertaintyState)
    operating_conditions: tuple[tuple[str, JsonValue], ...] = ()
    capabilities: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metadata: tuple[tuple[str, JsonValue], ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "task", "model_family", "trace_kind", "schema_version"):
            object.__setattr__(self, field_name, _nonempty_text(getattr(self, field_name), field_name))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported PHM-EIR schema_version: {self.schema_version}")
        if not isinstance(self.prediction, PredictionState):
            raise TypeError("prediction must be PredictionState")
        if not self.evidence_atoms:
            raise ValueError("PHM-EIR requires at least one evidence atom")

        atom_ids = [atom.id for atom in self.evidence_atoms]
        path_ids = [path.id for path in self.evidence_paths]
        relation_ids = [relation.id for relation in self.mechanism_relations]
        for label, values in (
            ("evidence atom", atom_ids),
            ("evidence path", path_ids),
            ("mechanism relation", relation_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {label} ID")

        known_sources = set(atom_ids) | set(path_ids)
        for path in self.evidence_paths:
            unknown = set(path.atom_ids) - set(atom_ids)
            if unknown:
                raise ValueError(f"path {path.id!r} references unknown atoms: {sorted(unknown)}")
        for contribution in self.contributions:
            if contribution.source_id not in known_sources:
                raise ValueError(
                    f"contribution references unknown source: {contribution.source_id}"
                )
        for relation in self.mechanism_relations:
            unknown = set(relation.source_ids) - known_sources
            if unknown:
                raise ValueError(
                    f"relation {relation.id!r} references unknown sources: {sorted(unknown)}"
                )

        normalized_capabilities = tuple(
            _nonempty_text(item, "capabilities[]") for item in self.capabilities
        )
        if len(set(normalized_capabilities)) != len(normalized_capabilities):
            raise ValueError("capabilities must be unique")
        object.__setattr__(self, "capabilities", normalized_capabilities)
        object.__setattr__(
            self,
            "limitations",
            tuple(_nonempty_text(item, "limitations[]") for item in self.limitations),
        )

        operating_values = (
            self.operating_conditions
            if isinstance(self.operating_conditions, Mapping)
            else thaw_mapping(self.operating_conditions)
        )
        metadata_values = (
            self.metadata if isinstance(self.metadata, Mapping) else thaw_mapping(self.metadata)
        )
        object.__setattr__(
            self,
            "operating_conditions",
            freeze_mapping(operating_values, name="operating_conditions"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(metadata_values, name="metadata"),
        )

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(atom.id for atom in self.evidence_atoms)

    @property
    def path_ids(self) -> frozenset[str]:
        return frozenset(path.id for path in self.evidence_paths)

    @property
    def relation_ids(self) -> frozenset[str]:
        return frozenset(relation.id for relation in self.mechanism_relations)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "task": self.task,
            "model_family": self.model_family,
            "trace_kind": self.trace_kind,
            "prediction": self.prediction.to_dict(),
            "evidence_atoms": [item.to_dict() for item in self.evidence_atoms],
            "evidence_paths": [item.to_dict() for item in self.evidence_paths],
            "contributions": [item.to_dict() for item in self.contributions],
            "mechanism_relations": [item.to_dict() for item in self.mechanism_relations],
            "uncertainty": self.uncertainty.to_dict(),
            "operating_conditions": thaw_mapping(self.operating_conditions),
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
            "metadata": thaw_mapping(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


__all__ = [
    "SCHEMA_VERSION",
    "ClassContribution",
    "EvidenceAtom",
    "EvidencePath",
    "MechanismRelation",
    "PHMExplanationState",
    "PredictionState",
    "UncertaintyState",
    "freeze_mapping",
    "thaw_mapping",
]

"""Deterministic, score-free P07 E8 intervention registry.

The registry materializes protocol conditions only.  It does not run a model,
inspect gate weights, estimate effects, or assign causal or physical meaning.
Every truth class and exported raw path is authenticated independently against
:mod:`path_universe`; a recovery error is retained rather than excluded.  All
dictionary actions are validated by the executable operator registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Any, Final, Literal, Sequence, cast

import torch

from src.model_factory.X_model.UXFD.operator_attention.executable_operator_path_1d import (
    DictionaryIntervention,
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
    OperatorCorruption,
)

from . import path_universe
from .path_universe import (
    CORRUPTION_SEED_BY_OPTIMIZATION_SEED,
    K_STAGES,
    NON_IDENTITY_OPERATORS,
    OPERATORS,
    EquivalenceClass,
    OperatorName,
    PathRecord,
    RawPath,
)


SCHEMA_VERSION: Final[int] = 2
PROTOCOL_ID: Final[str] = "P07-E8-INTERVENTION-REGISTRY-v2"
CORRUPTION_RMS_LEVELS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)
WRONG_DICTIONARY_DERANGEMENT: Final[tuple[tuple[str, str], ...]] = (
    ("D1", "MA3"),
    ("ABS", "SQUARE"),
    ("SQUARE", "HT"),
    ("MA3", "D1"),
    ("HT", "ABS"),
)
INTERPRETATION_SCOPE: Final[str] = (
    "registered_mechanism_diagnostic_only_no_causal_or_physical_meaning"
)

_BASE_PATH_BINDING = "registered_exported_path_no_direct_path_edit"
_DIRECT_PATH_BINDING = "direct_registered_path_edit_or_control"
_ABSENT_CONTROL_RANK_DOMAIN = "P07-E8-absent-operator-control-v1"
_PATH_REPLACEMENT_RANK_DOMAIN = "P07-E8-path-replacement-v1"
_NON_SELECTED_STAGE_RANK_DOMAIN = "P07-E8-non-selected-stage-control-v1"
_EQUIVALENT_PATH_RANK_DOMAIN = "P07-E8-equivalent-raw-path-control-v1"

DictionaryDecisionRole = Literal[
    "supported_base",
    "serialization_sham",
    "targeted",
    "matched_control",
    "secondary_diagnostic",
]
PathDecisionRole = Literal["targeted", "matched_control", "serialization_sham"]


def _claim_boundary_payload() -> dict[str, Any]:
    return {
        "causal_claim_eligible": False,
        "physical_meaning_claimed": False,
        "scope": INTERPRETATION_SCOPE,
    }


def _effective_payload(serialized: str) -> dict[str, Any]:
    parsed = path_universe.strict_canonical_json_loads(serialized)
    if not isinstance(parsed, dict):
        raise ValueError("Effective dictionary payload must be a canonical JSON object.")
    return cast(dict[str, Any], parsed)


@dataclass(frozen=True, slots=True)
class DictionaryInterventionRecord:
    """One immutable class-level dictionary condition."""

    schema_version: int
    record_id: str
    order_index: int
    condition: str
    decision_role: DictionaryDecisionRole
    truth_class_id: str
    truth_class_sha256: str
    original_path: RawPath
    original_path_id: str
    original_path_sha256: str
    intervened_path: RawPath
    intervened_path_id: str
    intervened_path_sha256: str
    path_binding_kind: str
    target_operators: tuple[str, ...]
    control_operators: tuple[str, ...]
    corruption_magnitude: float | None
    intervention: DictionaryIntervention | None
    base_dictionary_sha256: str
    effective_dictionary_sha256: str
    effective_dictionary_payload_json: str
    optimization_seed: int
    corruption_seed: int
    paired_record_ids: tuple[str, ...]
    primary_eligible: bool
    secondary_eligible: bool

    def to_payload(self) -> dict[str, Any]:
        effective = _effective_payload(self.effective_dictionary_payload_json)
        return {
            "schema_version": self.schema_version,
            "record_family": "dictionary",
            "record_id": self.record_id,
            "order_index": self.order_index,
            "condition": self.condition,
            "decision_role": self.decision_role,
            "truth_class_id": self.truth_class_id,
            "truth_class_sha256": self.truth_class_sha256,
            "original_path": list(self.original_path),
            "original_path_id": self.original_path_id,
            "original_path_sha256": self.original_path_sha256,
            "intervened_path": list(self.intervened_path),
            "intervened_path_id": self.intervened_path_id,
            "intervened_path_sha256": self.intervened_path_sha256,
            "path_binding_kind": self.path_binding_kind,
            "target_operators": list(self.target_operators),
            "control_operators": list(self.control_operators),
            "corruption_magnitude": self.corruption_magnitude,
            "dictionary_intervention": effective["dictionary_intervention"],
            "base_dictionary_sha256": self.base_dictionary_sha256,
            "effective_dictionary_sha256": self.effective_dictionary_sha256,
            "effective_dictionary_payload": effective,
            "optimization_seed": self.optimization_seed,
            "corruption_seed": self.corruption_seed,
            "paired_record_ids": list(self.paired_record_ids),
            "primary_eligible": self.primary_eligible,
            "secondary_eligible": self.secondary_eligible,
            "claim_boundary": _claim_boundary_payload(),
        }

    @property
    def record_sha256(self) -> str:
        return path_universe.canonical_json_sha256(self.to_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_payload(), "record_sha256": self.record_sha256}

    def canonical_json(self) -> str:
        return path_universe.canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True, slots=True)
class PathInterventionRecord:
    """One immutable selected-edge intervention or matched path control."""

    schema_version: int
    record_id: str
    order_index: int
    block_id: str
    condition: str
    decision_role: PathDecisionRole
    truth_class_id: str
    truth_class_sha256: str
    target_stage: int
    intervention_stage: int | None
    original_operator: str | None
    replacement_operator: str | None
    original_path: RawPath
    original_path_id: str
    original_path_sha256: str
    intervened_path: RawPath
    intervened_path_id: str
    intervened_path_sha256: str
    path_binding_kind: str
    base_dictionary_sha256: str
    effective_dictionary_sha256: str
    effective_dictionary_payload_json: str
    optimization_seed: int
    corruption_seed: int
    paired_record_ids: tuple[str, ...]
    primary_eligible: bool
    secondary_eligible: bool

    def to_payload(self) -> dict[str, Any]:
        effective = _effective_payload(self.effective_dictionary_payload_json)
        return {
            "schema_version": self.schema_version,
            "record_family": "path",
            "record_id": self.record_id,
            "order_index": self.order_index,
            "block_id": self.block_id,
            "condition": self.condition,
            "decision_role": self.decision_role,
            "truth_class_id": self.truth_class_id,
            "truth_class_sha256": self.truth_class_sha256,
            "target_stage": self.target_stage,
            "intervention_stage": self.intervention_stage,
            "original_operator": self.original_operator,
            "replacement_operator": self.replacement_operator,
            "original_path": list(self.original_path),
            "original_path_id": self.original_path_id,
            "original_path_sha256": self.original_path_sha256,
            "intervened_path": list(self.intervened_path),
            "intervened_path_id": self.intervened_path_id,
            "intervened_path_sha256": self.intervened_path_sha256,
            "path_binding_kind": self.path_binding_kind,
            "base_dictionary_sha256": self.base_dictionary_sha256,
            "effective_dictionary_sha256": self.effective_dictionary_sha256,
            "effective_dictionary_payload": effective,
            "optimization_seed": self.optimization_seed,
            "corruption_seed": self.corruption_seed,
            "paired_record_ids": list(self.paired_record_ids),
            "primary_eligible": self.primary_eligible,
            "secondary_eligible": self.secondary_eligible,
            "claim_boundary": _claim_boundary_payload(),
        }

    @property
    def record_sha256(self) -> str:
        return path_universe.canonical_json_sha256(self.to_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_payload(), "record_sha256": self.record_sha256}

    def canonical_json(self) -> str:
        return path_universe.canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True, slots=True)
class InterventionRegistry:
    """Complete immutable E8 registry for one truth class and exported path."""

    schema_version: int
    protocol_id: str
    truth_class_id: str
    truth_class_sha256: str
    selected_path: RawPath
    selected_path_id: str
    selected_path_sha256: str
    selected_path_class_id: str
    selected_path_class_sha256: str
    selected_path_semantic_match: bool
    optimization_seed: int
    corruption_seed: int
    essential_operators: tuple[str, ...]
    provably_absent_operators: tuple[str, ...]
    dictionary_records: tuple[DictionaryInterventionRecord, ...]
    path_records: tuple[PathInterventionRecord, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "truth_class_id": self.truth_class_id,
            "truth_class_sha256": self.truth_class_sha256,
            "selected_path": list(self.selected_path),
            "selected_path_id": self.selected_path_id,
            "selected_path_sha256": self.selected_path_sha256,
            "selected_path_class_id": self.selected_path_class_id,
            "selected_path_class_sha256": self.selected_path_class_sha256,
            "selected_path_semantic_match": self.selected_path_semantic_match,
            "optimization_seed": self.optimization_seed,
            "corruption_seed": self.corruption_seed,
            "essential_operators": list(self.essential_operators),
            "provably_absent_operators": list(self.provably_absent_operators),
            "dictionary_records": [record.to_dict() for record in self.dictionary_records],
            "path_records": [record.to_dict() for record in self.path_records],
            "claim_boundary": _claim_boundary_payload(),
        }

    @property
    def manifest_sha256(self) -> str:
        return path_universe.canonical_json_sha256(self.to_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_payload(), "manifest_sha256": self.manifest_sha256}

    def canonical_json(self) -> str:
        return path_universe.canonical_json_bytes(self.to_dict()).decode("utf-8")


@lru_cache(maxsize=1)
def _class_by_id() -> dict[str, EquivalenceClass]:
    return {item.class_id: item for item in path_universe.enumerate_equivalence_classes()}


@lru_cache(maxsize=1)
def _path_by_id() -> dict[str, PathRecord]:
    return {item.raw_path_id: item for item in path_universe.enumerate_path_records()}


@lru_cache(maxsize=1)
def _path_by_value() -> dict[RawPath, PathRecord]:
    return {item.raw_path: item for item in path_universe.enumerate_path_records()}


@lru_cache(maxsize=1)
def _operator_core() -> ExecutableOperatorPath1D:
    # Construction initializes unused gate parameters.  fork_rng prevents this
    # registry-only operation from advancing the caller's optimization RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return ExecutableOperatorPath1D(1, ExecutableOperatorPathConfig())


def _validate_truth_class(truth_class: EquivalenceClass) -> EquivalenceClass:
    if not isinstance(truth_class, EquivalenceClass):
        raise TypeError("truth_class must be a path_universe.EquivalenceClass.")
    authoritative = _class_by_id().get(truth_class.class_id)
    if authoritative is None or truth_class != authoritative:
        raise ValueError("truth_class is not an exact registered equivalence class.")
    return authoritative


def _validate_selected_path(selected_path: PathRecord) -> PathRecord:
    if not isinstance(selected_path, PathRecord):
        raise TypeError("selected_path must be a path_universe.PathRecord.")
    authoritative = _path_by_id().get(selected_path.raw_path_id)
    if authoritative is None or selected_path != authoritative:
        raise ValueError("selected_path is not an exact registered raw path.")
    path_universe.validate_raw_path(selected_path.raw_path)
    return authoritative


def _validate_optimization_seed(optimization_seed: int) -> tuple[int, int]:
    if isinstance(optimization_seed, bool) or not isinstance(optimization_seed, Integral):
        raise TypeError("optimization_seed must be an integer.")
    normalized = int(optimization_seed)
    if normalized not in CORRUPTION_SEED_BY_OPTIMIZATION_SEED:
        raise ValueError("optimization_seed is outside the frozen 25-seed namespace.")
    return normalized, CORRUPTION_SEED_BY_OPTIMIZATION_SEED[normalized]


def _essential_unchecked(truth_class: EquivalenceClass) -> tuple[str, ...]:
    member_sets = [set(member.raw_path).difference({"I"}) for member in truth_class.members]
    intersection = set.intersection(*member_sets)
    return tuple(operator for operator in NON_IDENTITY_OPERATORS if operator in intersection)


def _absent_unchecked(truth_class: EquivalenceClass) -> tuple[str, ...]:
    member_sets = [set(member.raw_path).difference({"I"}) for member in truth_class.members]
    union = set.union(*member_sets)
    return tuple(operator for operator in NON_IDENTITY_OPERATORS if operator not in union)


def semantically_essential_operators(
    truth_class: EquivalenceClass,
) -> tuple[str, ...]:
    """Return non-I operators present in every registered raw class member."""

    return _essential_unchecked(_validate_truth_class(truth_class))


def provably_absent_control_operators(
    truth_class: EquivalenceClass,
) -> tuple[str, ...]:
    """Return non-I operators absent from the union of all class members."""

    return _absent_unchecked(_validate_truth_class(truth_class))


def _rank_digest(domain: str, bindings: dict[str, Any]) -> str:
    return path_universe.canonical_json_sha256(
        {
            "domain": domain,
            "protocol_id": PROTOCOL_ID,
            **bindings,
        }
    )


def _ranked_operator(
    candidates: Sequence[str],
    *,
    domain: str,
    truth_class: EquivalenceClass,
    bindings: dict[str, Any],
) -> str:
    values = tuple(candidates)
    if not values:
        raise ValueError(f"No eligible registered candidate is available for {domain}.")
    if len(set(values)) != len(values) or any(value not in OPERATORS for value in values):
        raise ValueError("Rank candidates must be unique base-registry operators.")
    order = {operator: index for index, operator in enumerate(OPERATORS)}
    return min(
        values,
        key=lambda candidate: (
            _rank_digest(
                domain,
                {
                    "truth_class_id": truth_class.class_id,
                    "truth_class_sha256": truth_class.class_sha256,
                    "candidate_operator": candidate,
                    **bindings,
                },
            ),
            order[candidate],
        ),
    )


def _matched_absent_operator(
    truth_class: EquivalenceClass,
    target_operator: str,
    absent_operators: Sequence[str],
) -> str:
    if target_operator not in _essential_unchecked(truth_class):
        raise ValueError("Cannot match a control to a nonessential target operator.")
    return _ranked_operator(
        absent_operators,
        domain=_ABSENT_CONTROL_RANK_DOMAIN,
        truth_class=truth_class,
        bindings={"target_operator": target_operator},
    )


def _same_signature_replacement(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    stage: int,
) -> str:
    operator = selected_path.raw_path[stage]
    base_manifest = _operator_core().dictionary_manifest(None)
    specs = {
        item["name"]: (item["input_kind"], item["output_kind"])
        for item in base_manifest["operator_specs"]
    }
    candidates = tuple(
        candidate
        for candidate in NON_IDENTITY_OPERATORS
        if candidate != operator and specs[candidate] == specs[operator]
    )
    return _ranked_operator(
        candidates,
        domain=_PATH_REPLACEMENT_RANK_DOMAIN,
        truth_class=truth_class,
        bindings={
            "selected_path_id": selected_path.raw_path_id,
            "stage": stage,
            "selected_operator": operator,
        },
    )


def _ranked_other_stage(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    target_stage: int,
) -> int | None:
    candidates = tuple(
        stage
        for stage, operator in enumerate(selected_path.raw_path)
        if stage != target_stage and operator != "I"
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda stage: (
            _rank_digest(
                _NON_SELECTED_STAGE_RANK_DOMAIN,
                {
                    "truth_class_id": truth_class.class_id,
                    "selected_path_id": selected_path.raw_path_id,
                    "target_stage": target_stage,
                    "candidate_stage": stage,
                    "candidate_operator": selected_path.raw_path[stage],
                },
            ),
            stage,
        ),
    )


def _ranked_equivalent_member(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    target_stage: int,
) -> PathRecord | None:
    selected_class = _class_by_id().get(selected_path.class_id)
    if selected_class is None or selected_path not in selected_class.members:
        raise ValueError("Selected path has no authoritative semantic class.")
    candidates = tuple(
        member
        for member in selected_class.members
        if member.raw_path_id != selected_path.raw_path_id
    )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda member: (
            _rank_digest(
                _EQUIVALENT_PATH_RANK_DOMAIN,
                {
                    "truth_class_id": truth_class.class_id,
                    "selected_path_class_id": selected_class.class_id,
                    "selected_path_id": selected_path.raw_path_id,
                    "target_stage": target_stage,
                    "candidate_path_id": member.raw_path_id,
                },
            ),
            member.raw_path_id,
        ),
    )


def _record_id(family: str, bindings: dict[str, Any]) -> str:
    digest = _rank_digest(f"P07-E8-{family}-record-id-v1", bindings)
    return f"P07-E8-{family.upper()}-{digest}"


def _block_id(selected_path: PathRecord, stage: int) -> str:
    digest = _rank_digest(
        "P07-E8-path-block-id-v1",
        {"selected_path_id": selected_path.raw_path_id, "target_stage": stage},
    )
    return f"P07-E8-PATH-BLOCK-{digest}"


def _dictionary_binding(
    intervention: DictionaryIntervention | None,
) -> tuple[str, str, str]:
    try:
        manifest = _operator_core().dictionary_manifest(intervention)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid registered dictionary intervention: {error}") from error
    serialized = path_universe.canonical_json_bytes(manifest).decode("utf-8")
    return (
        cast(str, manifest["base_dictionary_sha256"]),
        cast(str, manifest["effective_dictionary_sha256"]),
        serialized,
    )


def _path_record_after_replacement(
    selected_path: PathRecord, stage: int, replacement_operator: str
) -> PathRecord:
    if stage < 0 or stage >= K_STAGES:
        raise ValueError("Path intervention stage is outside the frozen three-stage path.")
    if replacement_operator not in OPERATORS:
        raise ValueError("Path replacement is not registered in the base path universe.")
    values = list(selected_path.raw_path)
    values[stage] = replacement_operator
    raw_path = path_universe.validate_raw_path(values)
    return _path_by_value()[raw_path]


def _dictionary_record(
    *,
    order_index: int,
    record_id: str,
    condition: str,
    decision_role: DictionaryDecisionRole,
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    target_operators: tuple[str, ...] = (),
    control_operators: tuple[str, ...] = (),
    corruption_magnitude: float | None = None,
    intervention: DictionaryIntervention | None = None,
    optimization_seed: int,
    corruption_seed: int,
    paired_record_ids: tuple[str, ...],
    primary_eligible: bool = True,
    secondary_eligible: bool = True,
) -> DictionaryInterventionRecord:
    base_hash, effective_hash, payload_json = _dictionary_binding(intervention)
    return DictionaryInterventionRecord(
        schema_version=SCHEMA_VERSION,
        record_id=record_id,
        order_index=order_index,
        condition=condition,
        decision_role=decision_role,
        truth_class_id=truth_class.class_id,
        truth_class_sha256=truth_class.class_sha256,
        original_path=selected_path.raw_path,
        original_path_id=selected_path.raw_path_id,
        original_path_sha256=selected_path.raw_path_sha256,
        intervened_path=selected_path.raw_path,
        intervened_path_id=selected_path.raw_path_id,
        intervened_path_sha256=selected_path.raw_path_sha256,
        path_binding_kind=_BASE_PATH_BINDING,
        target_operators=target_operators,
        control_operators=control_operators,
        corruption_magnitude=corruption_magnitude,
        intervention=intervention,
        base_dictionary_sha256=base_hash,
        effective_dictionary_sha256=effective_hash,
        effective_dictionary_payload_json=payload_json,
        optimization_seed=optimization_seed,
        corruption_seed=corruption_seed,
        paired_record_ids=paired_record_ids,
        primary_eligible=primary_eligible,
        secondary_eligible=secondary_eligible,
    )


def _build_dictionary_records(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    optimization_seed: int,
    corruption_seed: int,
    essential: tuple[str, ...],
    absent: tuple[str, ...],
) -> tuple[DictionaryInterventionRecord, ...]:
    def identifier(condition: str, **extra: Any) -> str:
        return _record_id(
            "dictionary",
            {
                "truth_class_id": truth_class.class_id,
                "selected_path_id": selected_path.raw_path_id,
                "optimization_seed": optimization_seed,
                "condition": condition,
                **extra,
            },
        )

    base_id = identifier("supported_base")
    sham_id = identifier("serialization_sham")
    removal_pairs = []
    for operator in essential:
        control = _matched_absent_operator(truth_class, operator, absent)
        removal_pairs.append(
            (
                operator,
                control,
                identifier("essential_operator_removal", target_operator=operator),
                identifier(
                    "matched_absent_operator_removal_control",
                    target_operator=operator,
                    control_operator=control,
                ),
            )
        )
    wrong_id = identifier("wrong_dictionary_derangement")
    corruption_ids = {
        magnitude: identifier("operator_output_corruption", magnitude=magnitude)
        for magnitude in CORRUPTION_RMS_LEVELS
    }
    ma5_ids = {
        stage: identifier("ma5_expansion_secondary", stage=stage)
        for stage in range(K_STAGES)
    }

    records: list[DictionaryInterventionRecord] = []

    def append(**kwargs: Any) -> None:
        records.append(_dictionary_record(order_index=len(records), **kwargs))

    append(
        record_id=base_id,
        condition="supported_base",
        decision_role="supported_base",
        truth_class=truth_class,
        selected_path=selected_path,
        optimization_seed=optimization_seed,
        corruption_seed=corruption_seed,
        paired_record_ids=(sham_id,),
    )
    append(
        record_id=sham_id,
        condition="serialization_sham",
        decision_role="serialization_sham",
        truth_class=truth_class,
        selected_path=selected_path,
        intervention=DictionaryIntervention(),
        optimization_seed=optimization_seed,
        corruption_seed=corruption_seed,
        paired_record_ids=(base_id,),
    )
    for operator, control, target_id, control_id in removal_pairs:
        removed = tuple((stage, operator) for stage in range(K_STAGES))
        control_removed = tuple((stage, control) for stage in range(K_STAGES))
        append(
            record_id=target_id,
            condition="essential_operator_removal",
            decision_role="targeted",
            truth_class=truth_class,
            selected_path=selected_path,
            target_operators=(operator,),
            intervention=DictionaryIntervention(removed=removed),
            optimization_seed=optimization_seed,
            corruption_seed=corruption_seed,
            paired_record_ids=(control_id, sham_id, base_id),
        )
        append(
            record_id=control_id,
            condition="matched_absent_operator_removal_control",
            decision_role="matched_control",
            truth_class=truth_class,
            selected_path=selected_path,
            target_operators=(operator,),
            control_operators=(control,),
            intervention=DictionaryIntervention(removed=control_removed),
            optimization_seed=optimization_seed,
            corruption_seed=corruption_seed,
            paired_record_ids=(target_id,),
        )

    wrong_replacements = tuple(
        sorted(
            (stage, registered, executed)
            for stage in range(K_STAGES)
            for registered, executed in WRONG_DICTIONARY_DERANGEMENT
        )
    )
    append(
        record_id=wrong_id,
        condition="wrong_dictionary_derangement",
        decision_role="targeted",
        truth_class=truth_class,
        selected_path=selected_path,
        target_operators=tuple(NON_IDENTITY_OPERATORS),
        intervention=DictionaryIntervention(replacements=wrong_replacements),
        optimization_seed=optimization_seed,
        corruption_seed=corruption_seed,
        paired_record_ids=(sham_id, base_id),
    )
    for magnitude in CORRUPTION_RMS_LEVELS:
        corruptions = tuple(
            sorted(
                (
                    OperatorCorruption(
                        stage=stage,
                        registered_operator=operator,
                        magnitude=magnitude,
                        seed=corruption_seed,
                    )
                    for stage in range(K_STAGES)
                    for operator in NON_IDENTITY_OPERATORS
                ),
                key=lambda item: (item.stage, item.registered_operator),
            )
        )
        append(
            record_id=corruption_ids[magnitude],
            condition="operator_output_corruption",
            decision_role="targeted",
            truth_class=truth_class,
            selected_path=selected_path,
            target_operators=tuple(NON_IDENTITY_OPERATORS),
            corruption_magnitude=magnitude,
            intervention=DictionaryIntervention(corruptions=corruptions),
            optimization_seed=optimization_seed,
            corruption_seed=corruption_seed,
            paired_record_ids=(sham_id, base_id),
        )
    for stage in range(K_STAGES):
        append(
            record_id=ma5_ids[stage],
            condition="ma5_expansion_secondary",
            decision_role="secondary_diagnostic",
            truth_class=truth_class,
            selected_path=selected_path,
            target_operators=("MA5",),
            intervention=DictionaryIntervention(added=((stage, "MA5"),)),
            optimization_seed=optimization_seed,
            corruption_seed=corruption_seed,
            paired_record_ids=(base_id,),
            primary_eligible=False,
            secondary_eligible=True,
        )
    return tuple(records)


def _path_record(
    *,
    order_index: int,
    record_id: str,
    block_id: str,
    condition: str,
    decision_role: PathDecisionRole,
    truth_class: EquivalenceClass,
    target_stage: int,
    intervention_stage: int | None,
    original_operator: str | None,
    replacement_operator: str | None,
    selected_path: PathRecord,
    intervened_path: PathRecord,
    dictionary_binding: tuple[str, str, str],
    optimization_seed: int,
    corruption_seed: int,
    paired_record_ids: tuple[str, ...],
) -> PathInterventionRecord:
    base_hash, effective_hash, payload_json = dictionary_binding
    return PathInterventionRecord(
        schema_version=SCHEMA_VERSION,
        record_id=record_id,
        order_index=order_index,
        block_id=block_id,
        condition=condition,
        decision_role=decision_role,
        truth_class_id=truth_class.class_id,
        truth_class_sha256=truth_class.class_sha256,
        target_stage=target_stage,
        intervention_stage=intervention_stage,
        original_operator=original_operator,
        replacement_operator=replacement_operator,
        original_path=selected_path.raw_path,
        original_path_id=selected_path.raw_path_id,
        original_path_sha256=selected_path.raw_path_sha256,
        intervened_path=intervened_path.raw_path,
        intervened_path_id=intervened_path.raw_path_id,
        intervened_path_sha256=intervened_path.raw_path_sha256,
        path_binding_kind=_DIRECT_PATH_BINDING,
        base_dictionary_sha256=base_hash,
        effective_dictionary_sha256=effective_hash,
        effective_dictionary_payload_json=payload_json,
        optimization_seed=optimization_seed,
        corruption_seed=corruption_seed,
        paired_record_ids=paired_record_ids,
        primary_eligible=True,
        secondary_eligible=True,
    )


def _build_path_records(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    optimization_seed: int,
    corruption_seed: int,
) -> tuple[PathInterventionRecord, ...]:
    base_binding = _dictionary_binding(None)
    records: list[PathInterventionRecord] = []

    for target_stage, target_operator in enumerate(selected_path.raw_path):
        if target_operator == "I":
            continue
        block_id = _block_id(selected_path, target_stage)

        def identifier(condition: str, **extra: Any) -> str:
            return _record_id(
                "path",
                {
                    "truth_class_id": truth_class.class_id,
                    "selected_path_id": selected_path.raw_path_id,
                    "optimization_seed": optimization_seed,
                    "target_stage": target_stage,
                    "condition": condition,
                    **extra,
                },
            )

        replacement = _same_signature_replacement(
            truth_class, selected_path, target_stage
        )
        other_stage = _ranked_other_stage(truth_class, selected_path, target_stage)
        equivalent = _ranked_equivalent_member(
            truth_class, selected_path, target_stage
        )
        deletion_id = identifier("selected_edge_deletion_to_identity")
        replacement_id = identifier(
            "selected_edge_registered_replacement", replacement=replacement
        )
        sham_id = identifier("unchanged_replay_sham")
        nonselected_id = (
            None
            if other_stage is None
            else identifier("non_selected_stage_control", control_stage=other_stage)
        )
        equivalent_id = (
            None
            if equivalent is None
            else identifier(
                "registered_equivalent_raw_path_control",
                equivalent_path_id=equivalent.raw_path_id,
            )
        )
        control_ids = tuple(
            item for item in (sham_id, nonselected_id, equivalent_id) if item is not None
        )

        def append(**kwargs: Any) -> None:
            records.append(
                _path_record(
                    order_index=len(records),
                    block_id=block_id,
                    truth_class=truth_class,
                    target_stage=target_stage,
                    selected_path=selected_path,
                    dictionary_binding=base_binding,
                    optimization_seed=optimization_seed,
                    corruption_seed=corruption_seed,
                    **kwargs,
                )
            )

        append(
            record_id=deletion_id,
            condition="selected_edge_deletion_to_identity",
            decision_role="targeted",
            intervention_stage=target_stage,
            original_operator=target_operator,
            replacement_operator="I",
            intervened_path=_path_record_after_replacement(
                selected_path, target_stage, "I"
            ),
            paired_record_ids=control_ids,
        )
        append(
            record_id=replacement_id,
            condition="selected_edge_registered_replacement",
            decision_role="targeted",
            intervention_stage=target_stage,
            original_operator=target_operator,
            replacement_operator=replacement,
            intervened_path=_path_record_after_replacement(
                selected_path, target_stage, replacement
            ),
            paired_record_ids=control_ids,
        )
        append(
            record_id=sham_id,
            condition="unchanged_replay_sham",
            decision_role="serialization_sham",
            intervention_stage=target_stage,
            original_operator=target_operator,
            replacement_operator=target_operator,
            intervened_path=selected_path,
            paired_record_ids=(deletion_id, replacement_id),
        )
        if other_stage is not None and nonselected_id is not None:
            other_operator = selected_path.raw_path[other_stage]
            append(
                record_id=nonselected_id,
                condition="non_selected_stage_control",
                decision_role="matched_control",
                intervention_stage=other_stage,
                original_operator=other_operator,
                replacement_operator="I",
                intervened_path=_path_record_after_replacement(
                    selected_path, other_stage, "I"
                ),
                paired_record_ids=(deletion_id, replacement_id),
            )
        if equivalent is not None and equivalent_id is not None:
            append(
                record_id=equivalent_id,
                condition="registered_equivalent_raw_path_control",
                decision_role="matched_control",
                intervention_stage=None,
                original_operator=None,
                replacement_operator=None,
                intervened_path=equivalent,
                paired_record_ids=(deletion_id, replacement_id),
            )
    return tuple(records)


def _build_registry_unchecked(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    optimization_seed: int,
    corruption_seed: int,
) -> InterventionRegistry:
    essential = _essential_unchecked(truth_class)
    absent = _absent_unchecked(truth_class)
    dictionary_records = _build_dictionary_records(
        truth_class,
        selected_path,
        optimization_seed,
        corruption_seed,
        essential,
        absent,
    )
    path_records = _build_path_records(
        truth_class, selected_path, optimization_seed, corruption_seed
    )
    return InterventionRegistry(
        schema_version=SCHEMA_VERSION,
        protocol_id=PROTOCOL_ID,
        truth_class_id=truth_class.class_id,
        truth_class_sha256=truth_class.class_sha256,
        selected_path=selected_path.raw_path,
        selected_path_id=selected_path.raw_path_id,
        selected_path_sha256=selected_path.raw_path_sha256,
        selected_path_class_id=selected_path.class_id,
        selected_path_class_sha256=_class_by_id()[selected_path.class_id].class_sha256,
        selected_path_semantic_match=selected_path.class_id == truth_class.class_id,
        optimization_seed=optimization_seed,
        corruption_seed=corruption_seed,
        essential_operators=essential,
        provably_absent_operators=absent,
        dictionary_records=dictionary_records,
        path_records=path_records,
    )


def build_intervention_registry(
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    optimization_seed: int,
) -> InterventionRegistry:
    """Build the score-free E8 registry without conditioning on recovery."""

    checked_class = _validate_truth_class(truth_class)
    checked_path = _validate_selected_path(selected_path)
    checked_seed, corruption_seed = _validate_optimization_seed(optimization_seed)
    registry = _build_registry_unchecked(
        checked_class, checked_path, checked_seed, corruption_seed
    )
    return validate_intervention_registry(registry)


def _validate_record_common(
    record: DictionaryInterventionRecord | PathInterventionRecord,
    *,
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
    optimization_seed: int,
    corruption_seed: int,
    all_ids: set[str],
) -> None:
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError("Intervention record schema version drifted.")
    if record.truth_class_id != truth_class.class_id or (
        record.truth_class_sha256 != truth_class.class_sha256
    ):
        raise ValueError("Intervention record truth-class binding is invalid.")
    if (
        record.original_path != selected_path.raw_path
        or record.original_path_id != selected_path.raw_path_id
        or record.original_path_sha256 != selected_path.raw_path_sha256
    ):
        raise ValueError("Intervention record original-path binding is invalid.")
    intervened = _path_by_id().get(record.intervened_path_id)
    if (
        intervened is None
        or intervened.raw_path != record.intervened_path
        or intervened.raw_path_sha256 != record.intervened_path_sha256
    ):
        raise ValueError("Intervention record intervened path is unregistered or hash-invalid.")
    if record.optimization_seed != optimization_seed or record.corruption_seed != corruption_seed:
        raise ValueError("Intervention record seed-domain binding is invalid.")
    if record.record_id in record.paired_record_ids or any(
        paired not in all_ids for paired in record.paired_record_ids
    ):
        raise ValueError("Intervention record has missing or self-referential controls.")
    if record.primary_eligible not in {True, False} or record.secondary_eligible not in {
        True,
        False,
    }:
        raise ValueError("Intervention eligibility flags must be boolean.")


def _validate_dictionary_record(
    record: DictionaryInterventionRecord,
    *,
    essential: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    if record.condition == "essential_operator_removal":
        if len(record.target_operators) != 1 or record.target_operators[0] not in essential:
            raise ValueError("Dictionary registry contains a nonessential targeted removal.")
    if record.condition == "matched_absent_operator_removal_control":
        if len(record.control_operators) != 1 or record.control_operators[0] not in absent:
            raise ValueError("Dictionary control is not provably absent from the truth class.")
    if record.condition == "ma5_expansion_secondary" and record.primary_eligible:
        raise ValueError("MA5 expansion must remain secondary-only.")

    base_hash, effective_hash, payload_json = _dictionary_binding(record.intervention)
    if (
        record.base_dictionary_sha256 != base_hash
        or record.effective_dictionary_sha256 != effective_hash
        or record.effective_dictionary_payload_json != payload_json
    ):
        raise ValueError("Dictionary intervention effective-payload binding is invalid.")
    if (
        record.intervened_path != record.original_path
        or record.intervened_path_id != record.original_path_id
        or record.intervened_path_sha256 != record.original_path_sha256
        or record.path_binding_kind != _BASE_PATH_BINDING
    ):
        raise ValueError("Dictionary condition must not predeclare a direct path edit.")


def _validate_path_record(
    record: PathInterventionRecord,
    *,
    truth_class: EquivalenceClass,
    selected_path: PathRecord,
) -> None:
    if record.target_stage < 0 or record.target_stage >= K_STAGES:
        raise ValueError("Path record target stage is invalid.")
    if selected_path.raw_path[record.target_stage] == "I":
        raise ValueError("Path registry may target only selected non-I edges.")
    if record.path_binding_kind != _DIRECT_PATH_BINDING:
        raise ValueError("Path record binding kind is invalid.")
    base_hash, effective_hash, payload_json = _dictionary_binding(None)
    if (
        record.base_dictionary_sha256 != base_hash
        or record.effective_dictionary_sha256 != effective_hash
        or record.effective_dictionary_payload_json != payload_json
    ):
        raise ValueError("Path intervention must retain the frozen base dictionary.")

    if record.condition == "selected_edge_deletion_to_identity":
        if (
            record.intervention_stage != record.target_stage
            or record.replacement_operator != "I"
            or record.original_operator != selected_path.raw_path[record.target_stage]
        ):
            raise ValueError("Selected-edge deletion record is malformed.")
    elif record.condition == "selected_edge_registered_replacement":
        if (
            record.intervention_stage != record.target_stage
            or record.replacement_operator not in NON_IDENTITY_OPERATORS
            or record.replacement_operator == selected_path.raw_path[record.target_stage]
        ):
            raise ValueError("Selected-edge replacement is unregistered or a no-op.")
    elif record.condition == "unchanged_replay_sham":
        if record.intervened_path != selected_path.raw_path:
            raise ValueError("Unchanged replay sham changed the path.")
    elif record.condition == "non_selected_stage_control":
        if (
            record.intervention_stage is None
            or record.intervention_stage == record.target_stage
            or selected_path.raw_path[record.intervention_stage] == "I"
            or record.replacement_operator != "I"
        ):
            raise ValueError("Non-selected-stage path control is ineligible or malformed.")
    elif record.condition == "registered_equivalent_raw_path_control":
        intervened = _path_by_id()[record.intervened_path_id]
        if (
            intervened.class_id != selected_path.class_id
            or intervened.raw_path_id == selected_path.raw_path_id
        ):
            raise ValueError(
                "Equivalent raw-path control is not equivalent to the exported path."
            )
    else:
        raise ValueError(f"Unknown path intervention condition {record.condition!r}.")


def validate_intervention_registry(
    registry: InterventionRegistry,
) -> InterventionRegistry:
    """Fail closed on any truth, path, seed, action, control, or ordering drift."""

    if not isinstance(registry, InterventionRegistry):
        raise TypeError("registry must be an InterventionRegistry.")
    if registry.schema_version != SCHEMA_VERSION or registry.protocol_id != PROTOCOL_ID:
        raise ValueError("Intervention registry schema or protocol ID drifted.")
    truth_class = _class_by_id().get(registry.truth_class_id)
    if truth_class is None or truth_class.class_sha256 != registry.truth_class_sha256:
        raise ValueError("Intervention registry truth-class binding is invalid.")
    selected_path = _path_by_id().get(registry.selected_path_id)
    if (
        selected_path is None
        or selected_path.raw_path != registry.selected_path
        or selected_path.raw_path_sha256 != registry.selected_path_sha256
    ):
        raise ValueError("Intervention registry selected-path binding is invalid.")
    _validate_selected_path(selected_path)
    selected_class = _class_by_id().get(selected_path.class_id)
    if (
        selected_class is None
        or registry.selected_path_class_id != selected_class.class_id
        or registry.selected_path_class_sha256 != selected_class.class_sha256
        or registry.selected_path_semantic_match
        != (selected_class.class_id == truth_class.class_id)
    ):
        raise ValueError("Intervention registry selected-path class binding is invalid.")
    optimization_seed, corruption_seed = _validate_optimization_seed(
        registry.optimization_seed
    )
    if registry.corruption_seed != corruption_seed:
        raise ValueError("Intervention registry corruption seed is outside the frozen binding.")

    essential = _essential_unchecked(truth_class)
    absent = _absent_unchecked(truth_class)
    if registry.essential_operators != essential or (
        registry.provably_absent_operators != absent
    ):
        raise ValueError("Essential/absent operator derivation drifted.")

    all_records: tuple[DictionaryInterventionRecord | PathInterventionRecord, ...] = (
        *registry.dictionary_records,
        *registry.path_records,
    )
    ids = [record.record_id for record in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Intervention registry contains duplicate record IDs.")
    all_ids = set(ids)
    for index, record in enumerate(registry.dictionary_records):
        if record.order_index != index:
            raise ValueError("Dictionary intervention ordering is invalid.")
        _validate_record_common(
            record,
            truth_class=truth_class,
            selected_path=selected_path,
            optimization_seed=optimization_seed,
            corruption_seed=corruption_seed,
            all_ids=all_ids,
        )
        _validate_dictionary_record(record, essential=essential, absent=absent)
    for index, record in enumerate(registry.path_records):
        if record.order_index != index:
            raise ValueError("Path intervention ordering is invalid.")
        _validate_record_common(
            record,
            truth_class=truth_class,
            selected_path=selected_path,
            optimization_seed=optimization_seed,
            corruption_seed=corruption_seed,
            all_ids=all_ids,
        )
        _validate_path_record(
            record, truth_class=truth_class, selected_path=selected_path
        )

    expected = _build_registry_unchecked(
        truth_class, selected_path, optimization_seed, corruption_seed
    )
    expected_control_ids = {
        record.record_id
        for record in (*expected.dictionary_records, *expected.path_records)
        if record.decision_role in {
            "matched_control",
            "serialization_sham",
        }
    }
    missing_controls = sorted(expected_control_ids.difference(all_ids))
    if missing_controls:
        raise ValueError(f"Intervention registry is missing required controls: {missing_controls}.")
    if registry != expected:
        raise ValueError("Intervention registry does not match the frozen score-free construction.")
    return registry


__all__ = [
    "CORRUPTION_RMS_LEVELS",
    "INTERPRETATION_SCOPE",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "WRONG_DICTIONARY_DERANGEMENT",
    "DictionaryInterventionRecord",
    "InterventionRegistry",
    "PathInterventionRecord",
    "build_intervention_registry",
    "provably_absent_control_operators",
    "semantically_essential_operators",
    "validate_intervention_registry",
]

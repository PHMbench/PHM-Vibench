"""Deterministic work graph for the frozen P07 E7--E11 protocol.

This module enumerates work; it does not train, open confirmatory test data, or
promote evidence.  Explicit fit/select and confirmatory-test nodes make the
one-way test boundary auditable.  Deterministic comparators are scheduled once
per composition/dataset fold and joined to the 25 stochastic method seeds
during analysis rather than misrepresented as 25 independent fits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Optional

from .path_universe import (
    OPTIMIZATION_SEEDS,
    build_composition_split_manifest,
)


PLAN_SCHEMA_VERSION: Final[int] = 2
PLAN_DOMAIN: Final[str] = "P07-E7-E11-EXECUTION-PLAN-v2"
CWRU_FOLDS: Final[tuple[str, ...]] = ("D1", "D2", "D3")
DIRG_FOLDS: Final[tuple[str, ...]] = ("S1", "S2", "S3")
LEARNED_SYNTHETIC_ARMS: Final[tuple[str, ...]] = (
    "proposed",
    "dense_operator_mixture",
)
NEURAL_CWRU_ARMS: Final[tuple[str, ...]] = (
    "proposed",
    "dense_operator_mixture",
    "random_dictionary",
    "attention_cnn",
    "explainable_cnn",
)
NEURAL_DIRG_ARMS: Final[tuple[str, ...]] = NEURAL_CWRU_ARMS

WorkStage = Literal[
    "synthetic_fit_select",
    "synthetic_confirmatory_test",
    "synthetic_threshold_calibration",
    "synthetic_intervention_test",
    "cwru_fit_select",
    "cwru_confirmatory_test",
    "dirg_fit_select",
    "dirg_confirmatory_test",
]


@dataclass(frozen=True, slots=True)
class WorkUnit:
    """One indivisible scheduled action with explicit test-access semantics."""

    work_unit_id: str
    stage: WorkStage
    experiment_ids: tuple[str, ...]
    arm_id: str
    composition_class_id: Optional[str]
    fold_id: Optional[str]
    optimization_seed: Optional[int]
    depends_on: tuple[str, ...]
    uses_confirmatory_test: bool
    stochastic_fit: bool
    replication_policy: str
    required_outputs: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "work_unit_id": self.work_unit_id,
            "stage": self.stage,
            "experiment_ids": list(self.experiment_ids),
            "arm_id": self.arm_id,
            "composition_class_id": self.composition_class_id,
            "fold_id": self.fold_id,
            "optimization_seed": self.optimization_seed,
            "depends_on": list(self.depends_on),
            "uses_confirmatory_test": self.uses_confirmatory_test,
            "stochastic_fit": self.stochastic_fit,
            "replication_policy": self.replication_policy,
            "required_outputs": list(self.required_outputs),
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Canonical P07 work graph bound to one protocol digest and gate snapshot."""

    protocol_sha256: str
    composition_split_sha256: str
    seed_namespace_sha256: str
    human_gate_snapshot: bool
    thresholds_approved_snapshot: bool
    evidence_execution_allowed: bool
    units: tuple[WorkUnit, ...]
    plan_sha256: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "domain": PLAN_DOMAIN,
            "protocol_sha256": self.protocol_sha256,
            "composition_split_sha256": self.composition_split_sha256,
            "seed_namespace_sha256": self.seed_namespace_sha256,
            "human_gate_snapshot": self.human_gate_snapshot,
            "thresholds_approved_snapshot": self.thresholds_approved_snapshot,
            "evidence_execution_allowed": self.evidence_execution_allowed,
            "counts": _plan_counts(self.units),
            "units": [unit.to_payload() for unit in self.units],
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.payload_without_hash(), "plan_sha256": self.plan_sha256}

    def serialize(self) -> str:
        validate_execution_plan(self)
        return _canonical_json(self.to_payload())


def build_execution_plan(
    *,
    protocol_sha256: str,
    human_gate_snapshot: bool,
    thresholds_approved_snapshot: bool,
) -> ExecutionPlan:
    """Build the exact graph; false approvals retain a non-evidence plan."""

    protocol_digest = _require_sha256(protocol_sha256, "protocol_sha256")
    if not isinstance(human_gate_snapshot, bool):
        raise TypeError("human_gate_snapshot must be boolean.")
    if not isinstance(thresholds_approved_snapshot, bool):
        raise TypeError("thresholds_approved_snapshot must be boolean.")
    split_manifest = build_composition_split_manifest()
    test_class_ids = tuple(split_manifest["composition_splits"]["test"]["class_ids"])
    if len(test_class_ids) != 18 or len(set(test_class_ids)) != 18:
        raise RuntimeError("Frozen confirmatory composition bank must contain 18 classes.")

    units: list[WorkUnit] = []
    dependencies: dict[tuple[Any, ...], str] = {}
    proposed_fit_dependencies: dict[tuple[str, int], str] = {}

    for class_id in test_class_ids:
        for arm_id in LEARNED_SYNTHETIC_ARMS:
            for seed in OPTIMIZATION_SEEDS:
                fit = _make_unit(
                    protocol_digest,
                    stage="synthetic_fit_select",
                    experiment_ids=("E7", "E9", "E11"),
                    arm_id=arm_id,
                    composition_class_id=class_id,
                    fold_id=None,
                    optimization_seed=seed,
                    depends_on=(),
                    uses_confirmatory_test=False,
                    stochastic_fit=True,
                    replication_policy="fresh_fit_for_each_registered_seed",
                    required_outputs=(
                        "run_meta.yaml",
                        "checkpoint.pt",
                        "normalization_artifact.json",
                        "validation_metrics.json",
                        "exported_paths.jsonl",
                    ),
                )
                units.append(fit)
                dependencies[("synthetic", class_id, arm_id, seed)] = fit.work_unit_id
                if arm_id == "proposed":
                    proposed_fit_dependencies[(class_id, seed)] = fit.work_unit_id
                units.append(
                    _make_unit(
                        protocol_digest,
                        stage="synthetic_confirmatory_test",
                        experiment_ids=("E7", "E9", "E11"),
                        arm_id=arm_id,
                        composition_class_id=class_id,
                        fold_id=None,
                        optimization_seed=seed,
                        depends_on=(fit.work_unit_id,),
                        uses_confirmatory_test=True,
                        stochastic_fit=False,
                        replication_policy="evaluate_frozen_seed_checkpoint_once",
                        required_outputs=(
                            "per_case_metrics.parquet",
                            "latency.json",
                        ),
                    )
                )

        exhaustive_fit = _make_unit(
            protocol_digest,
            stage="synthetic_fit_select",
            experiment_ids=("E7", "E9", "E11"),
            arm_id="full_216_discrete_search",
            composition_class_id=class_id,
            fold_id=None,
            optimization_seed=None,
            depends_on=(),
            uses_confirmatory_test=False,
            stochastic_fit=False,
            replication_policy=(
                "one_deterministic_validation_search_joined_to_all_method_seeds"
            ),
            required_outputs=(
                "run_meta.yaml",
                "normalization_artifact.json",
                "validation_search_ledger.parquet",
                "exported_paths.jsonl",
            ),
        )
        units.append(exhaustive_fit)
        units.append(
            _make_unit(
                protocol_digest,
                stage="synthetic_confirmatory_test",
                experiment_ids=("E7", "E9", "E11"),
                arm_id="full_216_discrete_search",
                composition_class_id=class_id,
                fold_id=None,
                optimization_seed=None,
                depends_on=(exhaustive_fit.work_unit_id,),
                uses_confirmatory_test=True,
                stochastic_fit=False,
                replication_policy="evaluate_deterministic_selected_path_once",
                required_outputs=(
                    "per_case_metrics.parquet",
                    "latency.json",
                ),
            )
        )

    calibration = _make_unit(
        protocol_digest,
        stage="synthetic_threshold_calibration",
        experiment_ids=("E8",),
        arm_id="proposed",
        composition_class_id=None,
        fold_id=None,
        optimization_seed=None,
        depends_on=tuple(proposed_fit_dependencies.values()),
        uses_confirmatory_test=False,
        stochastic_fit=False,
        replication_policy=(
            "one_pooled_supported_validation_threshold_across_all_compositions_and_seeds"
        ),
        required_outputs=(
            "validation_scores.pt",
            "validation_error_indicators.pt",
            "threshold_artifact.json",
            "risk_coverage.csv",
        ),
    )
    units.append(calibration)
    for (class_id, seed), proposed_fit_id in proposed_fit_dependencies.items():
        units.append(
            _make_unit(
                protocol_digest,
                stage="synthetic_intervention_test",
                experiment_ids=("E8", "E9"),
                arm_id="proposed",
                composition_class_id=class_id,
                fold_id=None,
                optimization_seed=seed,
                depends_on=(proposed_fit_id, calibration.work_unit_id),
                uses_confirmatory_test=True,
                stochastic_fit=False,
                replication_policy=(
                    "reuse_frozen_checkpoint_and_one_pooled_validation_threshold"
                ),
                required_outputs=(
                    "path_intervention_manifest.jsonl",
                    "dictionary_manifest.json",
                    "intervention_results.parquet",
                ),
            )
        )

    for fold_id in CWRU_FOLDS:
        for arm_id in NEURAL_CWRU_ARMS:
            for seed in OPTIMIZATION_SEEDS:
                fit = _make_unit(
                    protocol_digest,
                    stage="cwru_fit_select",
                    experiment_ids=("E9", "E10", "E11"),
                    arm_id=arm_id,
                    composition_class_id=None,
                    fold_id=fold_id,
                    optimization_seed=seed,
                    depends_on=(),
                    uses_confirmatory_test=False,
                    stochastic_fit=True,
                    replication_policy="fresh_fold_model_for_each_registered_seed",
                    required_outputs=(
                        "run_meta.yaml",
                        "checkpoint.pt",
                        "validation_metrics.json",
                    ),
                )
                units.append(fit)
                units.append(
                    _make_unit(
                        protocol_digest,
                        stage="cwru_confirmatory_test",
                        experiment_ids=("E9", "E10", "E11"),
                        arm_id=arm_id,
                        composition_class_id=None,
                        fold_id=fold_id,
                        optimization_seed=seed,
                        depends_on=(fit.work_unit_id,),
                        uses_confirmatory_test=True,
                        stochastic_fit=False,
                        replication_policy="evaluate_frozen_fold_checkpoint_once",
                        required_outputs=(
                            "per_case_metrics.parquet",
                            "latency.json",
                        ),
                    )
                )

        ridge_fit = _make_unit(
            protocol_digest,
            stage="cwru_fit_select",
            experiment_ids=("E9", "E10", "E11"),
            arm_id="full_216_discrete_search",
            composition_class_id=None,
            fold_id=fold_id,
            optimization_seed=None,
            depends_on=(),
            uses_confirmatory_test=False,
            stochastic_fit=False,
            replication_policy=(
                "one_deterministic_216_ridge_search_joined_to_all_method_seeds"
            ),
            required_outputs=(
                "run_meta.yaml",
                "validation_search_ledger.parquet",
                "checkpoint.pt",
            ),
        )
        units.append(ridge_fit)
        units.append(
            _make_unit(
                protocol_digest,
                stage="cwru_confirmatory_test",
                experiment_ids=("E9", "E10", "E11"),
                arm_id="full_216_discrete_search",
                composition_class_id=None,
                fold_id=fold_id,
                optimization_seed=None,
                depends_on=(ridge_fit.work_unit_id,),
                uses_confirmatory_test=True,
                stochastic_fit=False,
                replication_policy="evaluate_deterministic_selected_ridge_once",
                required_outputs=(
                    "per_case_metrics.parquet",
                    "latency.json",
                ),
            )
        )

    for fold_id in DIRG_FOLDS:
        for arm_id in NEURAL_DIRG_ARMS:
            for seed in OPTIMIZATION_SEEDS:
                fit = _make_unit(
                    protocol_digest,
                    stage="dirg_fit_select",
                    experiment_ids=("E9", "E10", "E11"),
                    arm_id=arm_id,
                    composition_class_id=None,
                    fold_id=fold_id,
                    optimization_seed=seed,
                    depends_on=(),
                    uses_confirmatory_test=False,
                    stochastic_fit=True,
                    replication_policy=(
                        "fresh_held_severity_fold_model_for_each_registered_seed"
                    ),
                    required_outputs=(
                        "run_meta.yaml",
                        "checkpoint.pt",
                        "validation_metrics.json",
                    ),
                )
                units.append(fit)
                units.append(
                    _make_unit(
                        protocol_digest,
                        stage="dirg_confirmatory_test",
                        experiment_ids=("E9", "E10", "E11"),
                        arm_id=arm_id,
                        composition_class_id=None,
                        fold_id=fold_id,
                        optimization_seed=seed,
                        depends_on=(fit.work_unit_id,),
                        uses_confirmatory_test=True,
                        stochastic_fit=False,
                        replication_policy=(
                            "evaluate_frozen_held_severity_fold_checkpoint_once"
                        ),
                        required_outputs=(
                            "per_case_metrics.parquet",
                            "latency.json",
                        ),
                    )
                )

        ridge_fit = _make_unit(
            protocol_digest,
            stage="dirg_fit_select",
            experiment_ids=("E9", "E10", "E11"),
            arm_id="full_216_discrete_search",
            composition_class_id=None,
            fold_id=fold_id,
            optimization_seed=None,
            depends_on=(),
            uses_confirmatory_test=False,
            stochastic_fit=False,
            replication_policy=(
                "one_deterministic_216_ridge_search_per_dirg_fold_"
                "joined_to_all_method_seeds"
            ),
            required_outputs=(
                "run_meta.yaml",
                "validation_search_ledger.parquet",
                "checkpoint.pt",
            ),
        )
        units.append(ridge_fit)
        units.append(
            _make_unit(
                protocol_digest,
                stage="dirg_confirmatory_test",
                experiment_ids=("E9", "E10", "E11"),
                arm_id="full_216_discrete_search",
                composition_class_id=None,
                fold_id=fold_id,
                optimization_seed=None,
                depends_on=(ridge_fit.work_unit_id,),
                uses_confirmatory_test=True,
                stochastic_fit=False,
                replication_policy="evaluate_deterministic_selected_dirg_ridge_once",
                required_outputs=(
                    "per_case_metrics.parquet",
                    "latency.json",
                ),
            )
        )

    if len({unit.work_unit_id for unit in units}) != len(units):
        raise RuntimeError("Execution-plan work-unit hash collision.")
    evidence_allowed = human_gate_snapshot and thresholds_approved_snapshot
    provisional = ExecutionPlan(
        protocol_sha256=protocol_digest,
        composition_split_sha256=split_manifest["manifest_sha256"],
        seed_namespace_sha256=split_manifest["seed_namespace_sha256"],
        human_gate_snapshot=human_gate_snapshot,
        thresholds_approved_snapshot=thresholds_approved_snapshot,
        evidence_execution_allowed=evidence_allowed,
        units=tuple(units),
        plan_sha256="0" * 64,
    )
    result = ExecutionPlan(
        protocol_sha256=provisional.protocol_sha256,
        composition_split_sha256=provisional.composition_split_sha256,
        seed_namespace_sha256=provisional.seed_namespace_sha256,
        human_gate_snapshot=provisional.human_gate_snapshot,
        thresholds_approved_snapshot=provisional.thresholds_approved_snapshot,
        evidence_execution_allowed=provisional.evidence_execution_allowed,
        units=provisional.units,
        plan_sha256=_sha256_payload(provisional.payload_without_hash()),
    )
    validate_execution_plan(result)
    return result


def validate_execution_plan(plan: Any) -> ExecutionPlan:
    """Reject schedule drift, broken dependencies, and implicit test access."""

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan.")
    _require_sha256(plan.protocol_sha256, "protocol_sha256")
    _require_sha256(plan.composition_split_sha256, "composition_split_sha256")
    _require_sha256(plan.seed_namespace_sha256, "seed_namespace_sha256")
    _require_sha256(plan.plan_sha256, "plan_sha256")
    if plan.evidence_execution_allowed != (
        plan.human_gate_snapshot and plan.thresholds_approved_snapshot
    ):
        raise ValueError("Evidence execution state does not match approval snapshots.")
    if not plan.units:
        raise ValueError("Execution plan contains no work units.")
    ids = tuple(unit.work_unit_id for unit in plan.units)
    if len(set(ids)) != len(ids):
        raise ValueError("Execution plan contains duplicate work-unit IDs.")
    seen: set[str] = set()
    for unit in plan.units:
        _validate_unit(unit, protocol_sha256=plan.protocol_sha256)
        if any(dependency not in seen for dependency in unit.depends_on):
            raise ValueError("Work-unit dependency is missing or not topologically prior.")
        if unit.uses_confirmatory_test and not unit.depends_on:
            raise ValueError("Every confirmatory-test unit must bind a prior dependency.")
        if unit.stage.endswith("fit_select") and unit.depends_on:
            raise ValueError("Fit/select units must not depend on prior work.")
        seen.add(unit.work_unit_id)
    if _sha256_payload(plan.payload_without_hash()) != plan.plan_sha256:
        raise ValueError("Execution plan hash is invalid.")
    return plan


def ready_work_units(
    plan: ExecutionPlan,
    *,
    completed_work_unit_ids: Iterable[str],
    allow_confirmatory_test: bool,
) -> tuple[WorkUnit, ...]:
    """Return incomplete nodes whose dependencies are complete.

    Confirmatory nodes remain hidden unless the caller both requests them and
    the immutable plan snapshot records all approvals.
    """

    validate_execution_plan(plan)
    if not isinstance(allow_confirmatory_test, bool):
        raise TypeError("allow_confirmatory_test must be boolean.")
    completed = set(completed_work_unit_ids)
    known = {unit.work_unit_id for unit in plan.units}
    if not completed.issubset(known):
        raise ValueError("completed_work_unit_ids contains an unknown ID.")
    by_id = {unit.work_unit_id: unit for unit in plan.units}
    if any(
        not set(by_id[identifier].depends_on).issubset(completed)
        for identifier in completed
    ):
        raise ValueError("Completed work-unit set is not dependency closed.")
    test_allowed = allow_confirmatory_test and plan.evidence_execution_allowed
    return tuple(
        unit
        for unit in plan.units
        if unit.work_unit_id not in completed
        and set(unit.depends_on).issubset(completed)
        and (not unit.uses_confirmatory_test or test_allowed)
    )


def select_work_shard(
    units: Sequence[WorkUnit],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[WorkUnit, ...]:
    """Partition ready units by stable work-unit hash without changing IDs."""

    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
    ):
        raise ValueError("shard_count must be a positive integer.")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("shard_index must lie in [0, shard_count).")
    values = tuple(units)
    if any(not isinstance(unit, WorkUnit) for unit in values):
        raise TypeError("units must contain only WorkUnit objects.")
    return tuple(
        unit
        for unit in values
        if int(unit.work_unit_id.rsplit("-", 1)[1], 16) % shard_count == shard_index
    )


def _make_unit(
    protocol_sha256: str,
    *,
    stage: WorkStage,
    experiment_ids: tuple[str, ...],
    arm_id: str,
    composition_class_id: Optional[str],
    fold_id: Optional[str],
    optimization_seed: Optional[int],
    depends_on: tuple[str, ...],
    uses_confirmatory_test: bool,
    stochastic_fit: bool,
    replication_policy: str,
    required_outputs: tuple[str, ...],
) -> WorkUnit:
    identity = {
        "domain": PLAN_DOMAIN,
        "protocol_sha256": protocol_sha256,
        "stage": stage,
        "experiment_ids": list(experiment_ids),
        "arm_id": arm_id,
        "composition_class_id": composition_class_id,
        "fold_id": fold_id,
        "optimization_seed": optimization_seed,
        "depends_on": list(depends_on),
        "uses_confirmatory_test": uses_confirmatory_test,
        "stochastic_fit": stochastic_fit,
        "replication_policy": replication_policy,
        "required_outputs": list(required_outputs),
    }
    identifier = "P07-WORK-" + _sha256_payload(identity)[:24]
    return WorkUnit(
        work_unit_id=identifier,
        stage=stage,
        experiment_ids=tuple(experiment_ids),
        arm_id=arm_id,
        composition_class_id=composition_class_id,
        fold_id=fold_id,
        optimization_seed=optimization_seed,
        depends_on=tuple(depends_on),
        uses_confirmatory_test=uses_confirmatory_test,
        stochastic_fit=stochastic_fit,
        replication_policy=replication_policy,
        required_outputs=tuple(required_outputs),
    )


def _validate_unit(unit: WorkUnit, *, protocol_sha256: str) -> None:
    if not isinstance(unit, WorkUnit):
        raise TypeError("Execution plan units must be WorkUnit objects.")
    rebuilt = _make_unit(
        protocol_sha256,
        stage=unit.stage,
        experiment_ids=unit.experiment_ids,
        arm_id=unit.arm_id,
        composition_class_id=unit.composition_class_id,
        fold_id=unit.fold_id,
        optimization_seed=unit.optimization_seed,
        depends_on=unit.depends_on,
        uses_confirmatory_test=unit.uses_confirmatory_test,
        stochastic_fit=unit.stochastic_fit,
        replication_policy=unit.replication_policy,
        required_outputs=unit.required_outputs,
    )
    if rebuilt != unit:
        raise ValueError("Work-unit content does not match its deterministic ID.")
    if unit.composition_class_id is not None and unit.fold_id is not None:
        raise ValueError("A work unit cannot target both a composition and a dataset fold.")
    if unit.stage.startswith("cwru_") and unit.fold_id not in CWRU_FOLDS:
        raise ValueError("A CWRU work unit must target one frozen CWRU fold.")
    if unit.stage.startswith("dirg_") and unit.fold_id not in DIRG_FOLDS:
        raise ValueError("A DIRG work unit must target one frozen severity fold.")
    if unit.stochastic_fit and unit.optimization_seed not in OPTIMIZATION_SEEDS:
        raise ValueError("Stochastic fit lacks a frozen optimization seed.")
    if not unit.stochastic_fit and unit.stage.endswith("fit_select"):
        if unit.arm_id != "full_216_discrete_search":
            raise ValueError("Only the frozen full search may be seed-invariant.")
    if unit.stage == "synthetic_threshold_calibration":
        if (
            unit.uses_confirmatory_test
            or unit.stochastic_fit
            or unit.arm_id != "proposed"
            or unit.composition_class_id is not None
            or unit.fold_id is not None
            or unit.optimization_seed is not None
            or len(unit.depends_on) != 18 * len(OPTIMIZATION_SEEDS)
        ):
            raise ValueError("Synthetic threshold calibration contract is invalid.")


def _plan_counts(units: Sequence[WorkUnit]) -> dict[str, Any]:
    by_stage: dict[str, int] = {}
    for unit in units:
        by_stage[unit.stage] = by_stage.get(unit.stage, 0) + 1
    return {
        "total": len(units),
        "by_stage": {key: by_stage[key] for key in sorted(by_stage)},
        "confirmatory_test": sum(unit.uses_confirmatory_test for unit in units),
        "fit_select": sum(not unit.uses_confirmatory_test for unit in units),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


__all__ = [
    "CWRU_FOLDS",
    "DIRG_FOLDS",
    "LEARNED_SYNTHETIC_ARMS",
    "NEURAL_CWRU_ARMS",
    "NEURAL_DIRG_ARMS",
    "PLAN_DOMAIN",
    "PLAN_SCHEMA_VERSION",
    "ExecutionPlan",
    "WorkUnit",
    "build_execution_plan",
    "ready_work_units",
    "select_work_shard",
    "validate_execution_plan",
]

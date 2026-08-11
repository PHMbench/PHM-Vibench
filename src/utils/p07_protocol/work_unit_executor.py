"""Fail-closed execution boundary for one P07 work-unit ID.

The module deliberately separates orchestration from scientific computation.
It authenticates the frozen plan, approval snapshot, exact seed cohort,
upstream stores, data roles, nuisance cells, and source paths before dispatching
to an injected backend.  A backend returns in-memory artifacts; only an explicit
``execute=True`` request may seal them in :class:`DerivedArtifactStore`.

This executor does not promote claim evidence.  A completed store is a
traceable candidate artifact whose scientific eligibility is decided later by
the evidence guard and the independent experiment audit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, Optional, Protocol

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from . import (
    artifact_store,
    cwru_manifest,
    cwru_preprocessing,
    dirg_manifest,
    evidence_guard,
    execution_plan,
    experiment_runner,
    intervention_registry,
    path_universe,
    statistics_engine,
    synthetic_generator,
)


PROTOCOL_ID: Final[str] = "P07-G040-v3"
EXECUTOR_SCHEMA_VERSION: Final[int] = 1
EXECUTOR_DOMAIN: Final[str] = "P07-WORK-UNIT-EXECUTOR-v1"
EXECUTION_RECORD_NAME: Final[str] = "execution_record.json"
FAILURE_RECORD_NAME: Final[str] = "failure_record.json"
FROZEN_OPTIMIZATION_SEEDS: Final[tuple[int, ...]] = path_universe.OPTIMIZATION_SEEDS

DataRole = Literal[
    "fit",
    "validation_checkpoint_selection",
    "validation_threshold_calibration",
    "train",
    "confirmatory_test",
]
ExecutionState = Literal[
    "dry_run_ready",
    "dry_run_blocked",
    "complete",
    "failed",
]


class WorkUnitGuardError(RuntimeError):
    """A stable fail-closed rejection with a machine-readable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = _nonempty_text(code, "guard error code")


@dataclass(frozen=True, slots=True)
class HardwareRequest:
    """One process on CPU or one permitted physical GPU; never DDP."""

    device: Literal["cpu", "cuda"] = "cpu"
    physical_gpu_index: Optional[int] = None
    world_size: int = 1
    distributed_backend: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CWRUSourcePaths:
    """Read-only paths used to authenticate the official CWRU subset."""

    metadata_path: Path
    raw_dir: Path
    reader_source_path: Path
    preprocessing_source_path: Path


@dataclass(frozen=True, slots=True)
class DIRGSourcePaths:
    """Read-only paths used to authenticate the official DIRG subset."""

    metadata_path: Path
    raw_dir: Path
    reader_source_path: Path
    preprocessing_source_path: Path


@dataclass(frozen=True, slots=True)
class DependencyBinding:
    """Externally pinned hashes for one finalized upstream work-unit store."""

    work_unit_id: str
    output_root: Path
    artifact_index_sha256: str
    completion_marker_sha256: str

    def to_payload(self) -> dict[str, str]:
        return {
            "work_unit_id": self.work_unit_id,
            "output_root": str(self.output_root),
            "artifact_index_sha256": self.artifact_index_sha256,
            "completion_marker_sha256": self.completion_marker_sha256,
        }


@dataclass(frozen=True, slots=True)
class BackendArtifact:
    """One nonempty artifact returned by an injected scientific backend."""

    relative_path: str
    payload: bytes
    role: str


@dataclass(frozen=True, slots=True)
class BackendExecution:
    """Backend output plus an explicit declaration of every data boundary used."""

    artifacts: tuple[BackendArtifact, ...]
    accessed_data_roles: tuple[DataRole, ...]
    accessed_nuisance_cell_ids: tuple[str, ...]
    accessed_generator_seeds: tuple[int, ...]
    consumed_optimization_seeds: tuple[int, ...]
    input_sha256s: Mapping[str, str] = field(default_factory=dict)
    truth_raw_path_id: Optional[str] = None
    exported_raw_path_ids: tuple[str, ...] = ()
    intervention_registry_sha256: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CWRUAccessView:
    """Fold-scoped view that refuses paths outside the authorized split roles."""

    fold_id: str
    manifest_fold_id: str
    manifest_root_sha256: str
    allowed_data_roles: tuple[DataRole, ...]
    specimens: tuple[cwru_manifest.ManifestSpecimen, ...]
    split_by_specimen_key: Mapping[str, str]
    reader_source_path: Path
    preprocessing_source_path: Path
    _raw_dir: Path = field(repr=False)

    def raw_path(self, specimen_key: str) -> Path:
        """Return one authenticated raw path only when its split is authorized."""

        key = _nonempty_text(specimen_key, "specimen_key")
        by_key = {item.specimen_key: item for item in self.specimens}
        specimen = by_key.get(key)
        if specimen is None:
            raise WorkUnitGuardError(
                "cwru_split_access_forbidden",
                f"Specimen {key!r} is outside this work unit's authorized fold view.",
            )
        target = (self._raw_dir / specimen.file_name).resolve(strict=False)
        if target.parent != self._raw_dir or target.name != specimen.file_name:
            raise WorkUnitGuardError(
                "cwru_raw_path_escape",
                "Resolved CWRU specimen path escaped the authenticated raw directory.",
            )
        if _sha256_file(target) != specimen.raw_sha256:
            raise WorkUnitGuardError(
                "cwru_raw_hash_drift",
                f"CWRU raw bytes drifted for {specimen.file_name}.",
            )
        return target


@dataclass(frozen=True, slots=True)
class DIRGAccessView:
    """Severity-fold view that exposes only authenticated split specimens."""

    fold_id: str
    manifest_fold_id: str
    manifest_root_sha256: str
    metadata_file_sha256: str
    metadata_name_subset_sha256: str
    metadata_selected_subset_sha256: str
    raw_inventory_name_size_sha256: str
    reader_source_sha256: str
    preprocessing_source_sha256: str
    allowed_data_roles: tuple[DataRole, ...]
    specimens: tuple[dirg_manifest.DIRGSpecimen, ...]
    split_by_specimen_key: Mapping[str, str]
    reader_source_path: Path
    preprocessing_source_path: Path
    _raw_dir: Path = field(repr=False)

    def raw_path(self, specimen_key: str) -> Path:
        """Return one authenticated raw path only for the authorized split."""

        key = _nonempty_text(specimen_key, "specimen_key")
        by_key = {item.specimen_key: item for item in self.specimens}
        specimen = by_key.get(key)
        if specimen is None:
            raise WorkUnitGuardError(
                "dirg_split_access_forbidden",
                f"Specimen {key!r} is outside this work unit's authorized fold view.",
            )
        target = (self._raw_dir / specimen.file_name).resolve(strict=False)
        if target.parent != self._raw_dir or target.name != specimen.file_name:
            raise WorkUnitGuardError(
                "dirg_raw_path_escape",
                "Resolved DIRG specimen path escaped the authenticated raw directory.",
            )
        if _sha256_file(target) != specimen.raw_sha256:
            raise WorkUnitGuardError(
                "dirg_raw_hash_drift",
                f"DIRG raw bytes drifted for {specimen.file_name}.",
            )
        return target


@dataclass(frozen=True, slots=True)
class WorkUnitContext:
    """Guarded, stage-specific input delivered to exactly one backend method."""

    work_unit: execution_plan.WorkUnit
    plan_sha256: str
    protocol_sha256: str
    resolved_config_sha256: str
    runtime_commit: str
    command: tuple[str, ...]
    hardware: HardwareRequest
    dependencies: tuple[DependencyBinding, ...]
    dependency_output_roots: Mapping[str, Path]
    allowed_data_roles: tuple[DataRole, ...]
    allowed_nuisance_cells: tuple[synthetic_generator.NuisanceCell, ...]
    allowed_generator_seeds: tuple[int, ...]
    expected_consumed_optimization_seeds: tuple[int, ...]
    truth_class: Optional[path_universe.EquivalenceClass]
    truth_path: Optional[path_universe.PathRecord]
    intervention: Optional[intervention_registry.InterventionRegistry]
    threshold_artifact: Optional[evidence_guard.DictionaryFamilyThresholdArtifact]
    cwru_access: Optional[CWRUAccessView]
    dirg_access: Optional[DIRGAccessView]
    deterministic_bookkeeping_seed: Optional[int]


class WorkUnitBackend(Protocol):
    """Typed backend surface; there is intentionally no generic fallback."""

    def synthetic_fit_select(self, context: WorkUnitContext) -> BackendExecution: ...

    def synthetic_confirmatory_test(
        self, context: WorkUnitContext
    ) -> BackendExecution: ...

    def synthetic_threshold_calibration(
        self, context: WorkUnitContext
    ) -> BackendExecution: ...

    def synthetic_intervention_test(
        self, context: WorkUnitContext
    ) -> BackendExecution: ...

    def cwru_fit_select(self, context: WorkUnitContext) -> BackendExecution: ...

    def cwru_confirmatory_test(
        self, context: WorkUnitContext
    ) -> BackendExecution: ...

    def dirg_fit_select(self, context: WorkUnitContext) -> BackendExecution: ...

    def dirg_confirmatory_test(
        self, context: WorkUnitContext
    ) -> BackendExecution: ...


@dataclass(frozen=True, slots=True)
class WorkUnitRequest:
    """All caller-controlled facts for one dry-run or explicit execution."""

    plan: execution_plan.ExecutionPlan
    work_unit_id: str
    config_path: Path
    approved_protocol_sha256: Optional[str]
    runtime_commit: str
    command: tuple[str, ...]
    execute: bool = False
    output_root: Optional[Path] = None
    dependencies: tuple[DependencyBinding, ...] = ()
    immutable_source_roots: tuple[Path, ...] = ()
    cwru_sources: Optional[CWRUSourcePaths] = None
    dirg_sources: Optional[DIRGSourcePaths] = None
    hardware: HardwareRequest = HardwareRequest()


@dataclass(frozen=True, slots=True)
class WorkUnitResult:
    """Truthful terminal state for a dry-run, completed run, or failed run."""

    state: ExecutionState
    work_unit_id: str
    stage: Optional[str]
    backend_invoked: bool
    evidence_state: Literal["not_evidence"]
    reason_codes: tuple[str, ...]
    message: str
    output_root: Optional[Path]
    failure_record_root: Optional[Path]
    artifact_index_sha256: Optional[str]
    completion_marker_sha256: Optional[str]
    output_sha256s: Mapping[str, str]
    provenance: Mapping[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.state in {"dry_run_ready", "complete"}

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTOR_SCHEMA_VERSION,
            "domain": EXECUTOR_DOMAIN,
            "state": self.state,
            "work_unit_id": self.work_unit_id,
            "stage": self.stage,
            "backend_invoked": self.backend_invoked,
            "claim_evidence": False,
            "evidence_state": self.evidence_state,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "output_root": None if self.output_root is None else str(self.output_root),
            "failure_record_root": (
                None
                if self.failure_record_root is None
                else str(self.failure_record_root)
            ),
            "artifact_index_sha256": self.artifact_index_sha256,
            "completion_marker_sha256": self.completion_marker_sha256,
            "output_sha256s": dict(sorted(self.output_sha256s.items())),
            "provenance": _canonical_json_value(dict(self.provenance), "provenance"),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_payload())


@dataclass(frozen=True, slots=True)
class _LoadedConfig:
    path: Path
    value: Mapping[str, Any]
    source_sha256: str
    resolved_sha256: str


@dataclass(frozen=True, slots=True)
class _Prepared:
    request: WorkUnitRequest
    unit: execution_plan.WorkUnit
    config: _LoadedConfig
    source_sha256s: Mapping[str, str]
    input_sha256s: Mapping[str, str]
    dependencies: tuple[DependencyBinding, ...]
    dependency_output_roots: Mapping[str, Path]
    immutable_source_roots: tuple[Path, ...]
    context: WorkUnitContext
    cwru_manifest_value: Optional[cwru_manifest.CWRUManifest]
    dirg_manifest_value: Optional[dirg_manifest.DIRGManifest]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_protocol_config(path: Path | str) -> dict[str, Any]:
    """Load a strict YAML object for CLI plan construction and executor checks."""

    return dict(_load_config(Path(path)).value)


def run_work_unit(
    request: WorkUnitRequest,
    *,
    backend: Optional[WorkUnitBackend] = None,
) -> WorkUnitResult:
    """Dry-run or execute exactly one plan node.

    Dry-run is the default and performs no write.  Every explicit execution
    failure attempts to seal a separate ``<output_root>.failure`` store; that
    store is marked ``not_evidence`` and can never satisfy an upstream
    dependency's required-output contract.
    """

    if not isinstance(request, WorkUnitRequest):
        raise TypeError("request must be a WorkUnitRequest.")
    if not isinstance(request.execute, bool):
        raise TypeError("request.execute must be boolean.")

    prepared: Optional[_Prepared] = None
    backend_invoked = False
    backend_source_sha256s: Mapping[str, str] = MappingProxyType({})
    try:
        static, authorization_reasons = _prepare_static(request)
        if authorization_reasons:
            message = "Execution authorization is incomplete: " + ", ".join(
                authorization_reasons
            )
            if not request.execute:
                return _blocked_result(
                    request,
                    unit=static["unit"],
                    reason_codes=authorization_reasons,
                    message=message,
                    provenance=static["provenance"],
                )
            raise WorkUnitGuardError(authorization_reasons[0], message)

        prepared = _prepare_authorized(request, static)
        if not request.execute:
            return WorkUnitResult(
                state="dry_run_ready",
                work_unit_id=prepared.unit.work_unit_id,
                stage=prepared.unit.stage,
                backend_invoked=False,
                evidence_state="not_evidence",
                reason_codes=(),
                message="Dry-run validated; no backend was invoked and no file was written.",
                output_root=request.output_root,
                failure_record_root=None,
                artifact_index_sha256=None,
                completion_marker_sha256=None,
                output_sha256s=MappingProxyType({}),
                provenance=_result_provenance(prepared),
            )

        if backend is None:
            raise WorkUnitGuardError(
                "backend_required_for_execute",
                "Explicit execution requires an injected typed backend.",
            )
        backend_source_sha256s = _backend_source_sha256s(
            backend,
            prepared.unit.stage,
        )
        backend_invoked = True
        backend_result = _dispatch_backend(backend, prepared.context)
        _validate_backend_execution(prepared, backend_result)
        if prepared.cwru_manifest_value is not None:
            _verify_cwru_sources_unchanged(prepared)
        if prepared.dirg_manifest_value is not None:
            _verify_dirg_sources_unchanged(prepared)
        return _seal_success(
            prepared,
            backend_result,
            backend_source_sha256s=backend_source_sha256s,
        )
    except Exception as error:
        code = error.code if isinstance(error, WorkUnitGuardError) else (
            "backend_execution_failed" if backend_invoked else "executor_validation_failed"
        )
        message = f"{type(error).__name__}: {error}"
        if not request.execute:
            unit = _best_effort_unit(request)
            return _blocked_result(
                request,
                unit=unit,
                reason_codes=(code,),
                message=message,
                provenance=(
                    _result_provenance(prepared)
                    if prepared is not None
                    else _best_effort_provenance(request)
                ),
            )
        failure_root, failure_error = _write_failure_record(
            request,
            prepared=prepared,
            backend_invoked=backend_invoked,
            reason_code=code,
            error=error,
            backend_source_sha256s=backend_source_sha256s,
        )
        reasons = (code,) if failure_error is None else (
            code,
            "failure_record_write_failed",
        )
        if failure_error is not None:
            message = f"{message}; failure record error: {failure_error}"
        unit = prepared.unit if prepared is not None else _best_effort_unit(request)
        return WorkUnitResult(
            state="failed",
            work_unit_id=request.work_unit_id,
            stage=None if unit is None else unit.stage,
            backend_invoked=backend_invoked,
            evidence_state="not_evidence",
            reason_codes=reasons,
            message=message,
            output_root=request.output_root,
            failure_record_root=failure_root,
            artifact_index_sha256=None,
            completion_marker_sha256=None,
            output_sha256s=MappingProxyType({}),
            provenance=(
                _result_provenance(
                    prepared,
                    backend_source_sha256s=backend_source_sha256s,
                )
                if prepared is not None
                else _best_effort_provenance(request)
            ),
        )


def _prepare_static(
    request: WorkUnitRequest,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    plan = execution_plan.validate_execution_plan(request.plan)
    config = _load_config(request.config_path)
    _validate_hardware(request.hardware, config.value)
    _validate_runtime_identity(request.runtime_commit, request.command)
    _validate_exact_config_contract(config.value)

    approval = _mapping(config.value.get("approval"), "approval")
    human_snapshot = approval.get("experiment_protocol_approved")
    threshold_snapshot = approval.get("thresholds_approved")
    if not isinstance(human_snapshot, bool) or not isinstance(
        threshold_snapshot, bool
    ):
        raise WorkUnitGuardError(
            "approval_snapshot_invalid",
            "Approval snapshots must be explicit booleans.",
        )
    if approval.get("evidence_execution_allowed") is not (
        human_snapshot and threshold_snapshot
    ):
        raise WorkUnitGuardError(
            "approval_snapshot_inconsistent",
            "evidence_execution_allowed must equal the conjunction of both approvals.",
        )
    expected_plan = execution_plan.build_execution_plan(
        protocol_sha256=plan.protocol_sha256,
        human_gate_snapshot=human_snapshot,
        thresholds_approved_snapshot=threshold_snapshot,
    )
    if (
        execution_plan.PLAN_SCHEMA_VERSION != 2
        or execution_plan.PLAN_DOMAIN != "P07-E7-E11-EXECUTION-PLAN-v2"
        or len(plan.units) != 3_799
        or plan != expected_plan
    ):
        raise WorkUnitGuardError(
            "execution_plan_not_exact_frozen_graph",
            "Execution plan is not the exact deterministic v2 3,799-unit graph.",
        )
    by_id = {unit.work_unit_id: unit for unit in plan.units}
    unit = by_id.get(request.work_unit_id)
    if unit is None:
        raise WorkUnitGuardError(
            "unknown_work_unit_id",
            f"Unknown work-unit ID: {request.work_unit_id!r}.",
        )
    _validate_dataset_source_scope(request, unit)
    _validate_full_search_replication(plan, unit)

    source_sha256s = _source_sha256s()
    input_sha256s: dict[str, str] = {
        "config_source_sha256": config.source_sha256,
        "resolved_config_sha256": config.resolved_sha256,
        "execution_plan_sha256": plan.plan_sha256,
        "protocol_sha256": plan.protocol_sha256,
        "composition_split_sha256": plan.composition_split_sha256,
        "seed_namespace_sha256": plan.seed_namespace_sha256,
    }
    provenance = {
        "protocol_sha256": plan.protocol_sha256,
        "plan_sha256": plan.plan_sha256,
        "resolved_config_sha256": config.resolved_sha256,
        "runtime_commit": request.runtime_commit,
        "source_sha256s": dict(source_sha256s),
        "input_sha256s": dict(input_sha256s),
        "optimization_seed": unit.optimization_seed,
        "paired_optimization_seeds": list(FROZEN_OPTIMIZATION_SEEDS),
        "command": list(request.command),
        "hardware": asdict(request.hardware),
    }
    reasons = _authorization_reasons(
        request,
        config.value,
        unit=unit,
        human_snapshot=human_snapshot,
        threshold_snapshot=threshold_snapshot,
    )
    return (
        {
            "plan": plan,
            "unit": unit,
            "config": config,
            "source_sha256s": source_sha256s,
            "input_sha256s": input_sha256s,
            "provenance": provenance,
        },
        reasons,
    )


def _prepare_authorized(
    request: WorkUnitRequest,
    static: Mapping[str, Any],
) -> _Prepared:
    plan = static["plan"]
    unit = static["unit"]
    config = static["config"]
    source_sha256s = dict(static["source_sha256s"])
    input_sha256s = dict(static["input_sha256s"])

    _validate_output_root(request)
    dependencies, dependency_roots, dependency_inputs = _validate_dependencies(
        plan,
        unit,
        request.dependencies,
    )
    input_sha256s.update(dependency_inputs)

    truth_class, truth_path = _truth_binding(unit)
    threshold = _threshold_barrier(plan, unit, dependencies)
    intervention = _build_intervention_binding(
        plan,
        unit,
        dependencies,
        truth_class=truth_class,
    )

    manifest_value: Optional[cwru_manifest.CWRUManifest] = None
    dirg_manifest_value: Optional[dirg_manifest.DIRGManifest] = None
    cwru_access: Optional[CWRUAccessView] = None
    dirg_access: Optional[DIRGAccessView] = None
    if unit.stage.startswith("cwru_"):
        if request.cwru_sources is None:
            raise WorkUnitGuardError(
                "cwru_sources_required",
                "CWRU work units require all four authenticated source paths.",
            )
        manifest_value = _build_and_validate_cwru_manifest(
            request.cwru_sources,
            config.value,
        )
        input_sha256s.update(
            {
                "cwru_manifest_sha256": manifest_value.root_sha256,
                "cwru_metadata_subset_sha256": manifest_value.metadata_subset_sha256,
                "cwru_reader_source_sha256": manifest_value.reader_source_sha256,
                "cwru_preprocessing_source_sha256": (
                    manifest_value.preprocessing_source_sha256
                ),
            }
        )
        cwru_access = _make_cwru_access_view(
            manifest_value,
            request.cwru_sources,
            unit,
        )
    elif unit.stage.startswith("dirg_"):
        if request.dirg_sources is None:
            raise WorkUnitGuardError(
                "dirg_sources_required",
                "DIRG work units require all four authenticated source paths.",
            )
        dirg_manifest_value = _build_and_validate_dirg_manifest(
            request.dirg_sources,
            config.value,
        )
        input_sha256s.update(
            {
                "dirg_manifest_sha256": dirg_manifest_value.root_sha256,
                "dirg_metadata_file_sha256": (
                    dirg_manifest_value.metadata_file_sha256
                ),
                "dirg_metadata_name_subset_sha256": (
                    dirg_manifest_value.metadata_name_subset_sha256
                ),
                "dirg_metadata_selected_subset_sha256": (
                    dirg_manifest_value.metadata_selected_subset_sha256
                ),
                "dirg_raw_inventory_name_size_sha256": (
                    dirg_manifest_value.raw_inventory_name_size_sha256
                ),
                "dirg_reader_source_sha256": (
                    dirg_manifest_value.reader_source_sha256
                ),
                "dirg_preprocessing_source_sha256": (
                    dirg_manifest_value.preprocessing_source_sha256
                ),
            }
        )
        dirg_access = _make_dirg_access_view(
            dirg_manifest_value,
            request.dirg_sources,
            unit,
        )

    data_roles, nuisance_cells, generator_seeds = _stage_access_contract(unit)
    consumed_seeds = _expected_consumed_optimization_seeds(unit)
    bookkeeping_seed = (
        int(hashlib.sha256(unit.work_unit_id.encode("ascii")).hexdigest()[:8], 16)
        if unit.arm_id == "full_216_discrete_search"
        else None
    )
    immutable_roots = _immutable_source_roots(request, config.path)
    context = WorkUnitContext(
        work_unit=unit,
        plan_sha256=plan.plan_sha256,
        protocol_sha256=plan.protocol_sha256,
        resolved_config_sha256=config.resolved_sha256,
        runtime_commit=request.runtime_commit,
        command=tuple(request.command),
        hardware=request.hardware,
        dependencies=dependencies,
        dependency_output_roots=MappingProxyType(dict(dependency_roots)),
        allowed_data_roles=data_roles,
        allowed_nuisance_cells=nuisance_cells,
        allowed_generator_seeds=generator_seeds,
        expected_consumed_optimization_seeds=consumed_seeds,
        truth_class=truth_class,
        truth_path=truth_path,
        intervention=intervention,
        threshold_artifact=threshold,
        cwru_access=cwru_access,
        dirg_access=dirg_access,
        deterministic_bookkeeping_seed=bookkeeping_seed,
    )
    if threshold is not None:
        input_sha256s["threshold_artifact_sha256"] = threshold.artifact_sha256
    if intervention is not None:
        input_sha256s["intervention_registry_sha256"] = (
            intervention.manifest_sha256
        )
    return _Prepared(
        request=request,
        unit=unit,
        config=config,
        source_sha256s=MappingProxyType(source_sha256s),
        input_sha256s=MappingProxyType(input_sha256s),
        dependencies=dependencies,
        dependency_output_roots=MappingProxyType(dict(dependency_roots)),
        immutable_source_roots=immutable_roots,
        context=context,
        cwru_manifest_value=manifest_value,
        dirg_manifest_value=dirg_manifest_value,
    )


def _authorization_reasons(
    request: WorkUnitRequest,
    config: Mapping[str, Any],
    *,
    unit: execution_plan.WorkUnit,
    human_snapshot: bool,
    threshold_snapshot: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []

    def reject(code: str) -> None:
        if code not in reasons:
            reasons.append(code)

    if not human_snapshot:
        reject("human_gate_not_approved")
    approved = request.approved_protocol_sha256
    if not _is_sha256(approved) or approved != request.plan.protocol_sha256:
        reject("approved_protocol_sha256_missing_or_mismatch")
    approval = _mapping(config.get("approval"), "approval")
    declared_candidates = (
        approval.get("approved_protocol_sha256"),
        _mapping(config.get("protocol_sha256"), "protocol_sha256").get("value"),
    )
    declared = tuple(value for value in declared_candidates if value is not None)
    if not declared or any(value != approved for value in declared):
        reject("config_not_bound_to_approved_protocol_sha256")
    if unit.stage in {
        "synthetic_threshold_calibration",
        "synthetic_confirmatory_test",
        "synthetic_intervention_test",
        "cwru_confirmatory_test",
        "dirg_confirmatory_test",
    }:
        if not threshold_snapshot or not _all_thresholds_approved(config):
            reject("thresholds_not_approved")
    if unit.uses_confirmatory_test and not request.plan.evidence_execution_allowed:
        reject("confirmatory_test_not_approved")
    return tuple(reasons)


def _validate_exact_config_contract(config: Mapping[str, Any]) -> None:
    for key, expected in {
        "schema_version": 1,
        "paper_id": "P07",
        "protocol_id": PROTOCOL_ID,
    }.items():
        if config.get(key) != expected:
            raise WorkUnitGuardError(
                "config_identity_drift",
                f"Config {key} drifted from {expected!r}.",
            )
    runtime = _mapping(config.get("runtime"), "runtime")
    if runtime.get("conda_environment") != "LQ_signal":
        raise WorkUnitGuardError(
            "conda_environment_drift",
            "Evidence-bearing execution requires conda environment LQ_signal.",
        )
    seeds = _mapping(config.get("seeds"), "seeds")
    observed_seeds = seeds.get("optimization")
    if not isinstance(observed_seeds, list) or tuple(observed_seeds) != (
        FROZEN_OPTIMIZATION_SEEDS
    ):
        raise WorkUnitGuardError(
            "seed_cohort_not_exactly_frozen_25",
            "Optimization seeds must equal the exact ordered frozen 25-seed cohort.",
        )
    if seeds.get("optimization_count") != 25 or len(FROZEN_OPTIMIZATION_SEEDS) != 25:
        raise WorkUnitGuardError(
            "seed_cohort_not_exactly_frozen_25",
            "optimization_count must be exactly 25.",
        )

    manifests = _mapping(config.get("manifests"), "manifests")
    actual_manifests = {
        "path_universe_sha256": path_universe.build_path_universe_manifest()[
            "manifest_sha256"
        ],
        "composition_split_sha256": path_universe.build_composition_split_manifest()[
            "manifest_sha256"
        ],
        "seed_namespace_sha256": path_universe.build_seed_namespace_manifest()[
            "manifest_sha256"
        ],
        "synthetic_generator_sha256": synthetic_generator.build_synthetic_generator_manifest()[
            "manifest_sha256"
        ],
        "nuisance_sha256": synthetic_generator.build_nuisance_manifest()[
            "manifest_sha256"
        ],
    }
    if dict(manifests) != actual_manifests:
        raise WorkUnitGuardError(
            "protocol_dependency_hash_drift",
            "One or more frozen protocol-module manifest hashes drifted.",
        )

    if dict(_mapping(config.get("training_budget"), "training_budget")) != asdict(
        experiment_runner.TrainingBudget()
    ):
        raise WorkUnitGuardError(
            "training_budget_drift",
            "Training budget does not match experiment_runner.TrainingBudget.",
        )
    policy = _mapping(config.get("execution_policy"), "execution_policy")
    if policy.get("primary_exhaustive_evaluation_budget") != (
        experiment_runner.PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET
    ):
        raise WorkUnitGuardError(
            "full_search_budget_drift",
            "Primary full search must evaluate exactly 216 paths.",
        )
    if policy.get("replacement_seeds_allowed") is not False:
        raise WorkUnitGuardError(
            "replacement_seeds_forbidden",
            "Replacement optimization seeds are forbidden.",
        )
    analysis = _mapping(config.get("analysis_budget"), "analysis_budget")
    expected_analysis = {
        "bootstrap_replicates": statistics_engine.DEFAULT_BOOTSTRAP_DRAWS,
        "bootstrap_seed": statistics_engine.DEFAULT_RANDOM_SEED,
        "confidence_level": 0.95,
        "familywise_alpha": 0.05,
        "missing_seed_rule": "all_25_required_no_replacement",
    }
    if dict(analysis) != expected_analysis:
        raise WorkUnitGuardError(
            "statistics_contract_drift",
            "Analysis budget does not match the frozen statistics contract.",
        )


def _validate_full_search_replication(
    plan: execution_plan.ExecutionPlan,
    unit: execution_plan.WorkUnit,
) -> None:
    full_units = tuple(
        candidate
        for candidate in plan.units
        if candidate.arm_id == "full_216_discrete_search"
        and candidate.stage.endswith("fit_select")
    )
    synthetic = tuple(
        item for item in full_units if item.stage == "synthetic_fit_select"
    )
    cwru = tuple(item for item in full_units if item.stage == "cwru_fit_select")
    dirg = tuple(item for item in full_units if item.stage == "dirg_fit_select")
    expected_compositions = set(
        path_universe.build_composition_split_manifest()["composition_splits"][
            "test"
        ]["class_ids"]
    )
    if (
        len(full_units) != 24
        or len(synthetic) != 18
        or len(cwru) != 3
        or len(dirg) != 3
        or {item.composition_class_id for item in synthetic}
        != expected_compositions
        or {item.fold_id for item in cwru} != set(execution_plan.CWRU_FOLDS)
        or {item.fold_id for item in dirg} != set(execution_plan.DIRG_FOLDS)
    ):
        raise WorkUnitGuardError(
            "full_search_fake_seed_replication",
            "Full-216 search must occur once per each of 18 compositions, "
            "3 CWRU folds, and 3 DIRG folds only.",
        )
    if any(
        item.optimization_seed is not None
        or item.stochastic_fit
        or "joined_to_all_method_seeds" not in item.replication_policy
        for item in full_units
    ):
        raise WorkUnitGuardError(
            "full_search_fake_seed_replication",
            "Full-216 units must be deterministic and seed-invariant.",
        )
    if unit.arm_id == "full_216_discrete_search" and unit.optimization_seed is not None:
        raise WorkUnitGuardError(
            "full_search_fake_seed_replication",
            "A full-216 work unit may not carry an optimization seed.",
        )


def _validate_hardware(hardware: HardwareRequest, config: Mapping[str, Any]) -> None:
    if not isinstance(hardware, HardwareRequest):
        raise TypeError("hardware must be a HardwareRequest.")
    if hardware.device not in {"cpu", "cuda"}:
        raise WorkUnitGuardError("invalid_device", "device must be cpu or cuda.")
    if (
        isinstance(hardware.world_size, bool)
        or not isinstance(hardware.world_size, int)
        or hardware.world_size != 1
    ):
        raise WorkUnitGuardError(
            "multi_gpu_or_ddp_forbidden",
            "P07 permits exactly one process and world_size=1.",
        )
    if hardware.distributed_backend not in {None, "", "none"}:
        raise WorkUnitGuardError(
            "multi_gpu_or_ddp_forbidden",
            "DDP and every distributed backend are forbidden.",
        )
    index = hardware.physical_gpu_index
    if hardware.device == "cpu":
        if index is not None:
            raise WorkUnitGuardError(
                "cpu_with_gpu_index",
                "CPU execution must not declare a physical GPU index.",
            )
    elif isinstance(index, bool) or not isinstance(index, int) or index not in {0, 1}:
        code = "physical_gpu_2_forbidden" if index == 2 else "gpu_index_forbidden"
        raise WorkUnitGuardError(
            code,
            "CUDA execution requires exactly physical GPU 0 or 1; GPU 2 is forbidden.",
        )
    declared = _mapping(_mapping(config.get("runtime"), "runtime").get("hardware"), "runtime.hardware")
    if declared.get("multi_gpu_allowed") is not False or 2 not in tuple(
        declared.get("forbidden_physical_gpu_indices", ())
    ):
        raise WorkUnitGuardError(
            "hardware_config_drift",
            "Runtime hardware policy no longer explicitly forbids multi-GPU and GPU 2.",
        )


def _validate_runtime_identity(runtime_commit: str, command: Sequence[str]) -> None:
    if not _is_commit(runtime_commit):
        raise WorkUnitGuardError(
            "runtime_commit_invalid",
            "runtime_commit must be a lowercase 40-hex Git commit.",
        )
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be a non-string sequence.")
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise WorkUnitGuardError(
            "command_invalid",
            "command must contain nonempty argv strings.",
        )


def _validate_output_root(request: WorkUnitRequest) -> None:
    if request.output_root is None:
        if request.execute:
            raise WorkUnitGuardError(
                "output_root_required_for_execute",
                "Explicit execution requires a new absolute output_root.",
            )
        return
    root = _absolute_path(request.output_root, "output_root")
    _reject_rotor_simulation_path(root, "output_root")
    if root.exists() or root.is_symlink():
        raise WorkUnitGuardError(
            "output_root_not_create_only",
            "output_root must not already exist.",
        )


def _validate_dataset_source_scope(
    request: WorkUnitRequest,
    unit: execution_plan.WorkUnit,
) -> None:
    """Reject cross-dataset source capabilities before any source is opened."""

    is_cwru = unit.stage.startswith("cwru_")
    is_dirg = unit.stage.startswith("dirg_")
    if request.cwru_sources is not None and not is_cwru:
        raise WorkUnitGuardError(
            "unexpected_cwru_sources",
            "CWRU source paths may be supplied only to CWRU work units.",
        )
    if request.dirg_sources is not None and not is_dirg:
        raise WorkUnitGuardError(
            "unexpected_dirg_sources",
            "DIRG source paths may be supplied only to DIRG work units.",
        )


def _validate_dependencies(
    plan: execution_plan.ExecutionPlan,
    unit: execution_plan.WorkUnit,
    supplied: Sequence[DependencyBinding],
) -> tuple[tuple[DependencyBinding, ...], dict[str, Path], dict[str, str]]:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("dependencies must be a sequence of DependencyBinding objects.")
    values = tuple(supplied)
    if any(not isinstance(item, DependencyBinding) for item in values):
        raise TypeError("Every dependency must be a DependencyBinding.")
    ids = tuple(item.work_unit_id for item in values)
    if len(set(ids)) != len(ids):
        raise WorkUnitGuardError(
            "duplicate_dependency_binding",
            "Dependency bindings contain duplicate work-unit IDs.",
        )
    if set(ids) != set(unit.depends_on):
        raise WorkUnitGuardError(
            "dependency_closure_incomplete",
            "Supplied dependencies must exactly equal the work unit's declared dependencies.",
        )
    by_plan_id = {item.work_unit_id: item for item in plan.units}
    by_supplied = {item.work_unit_id: item for item in values}
    ordered: list[DependencyBinding] = []
    roots: dict[str, Path] = {}
    input_hashes: dict[str, str] = {}
    for dependency_id in unit.depends_on:
        binding = by_supplied[dependency_id]
        if not _is_sha256(binding.artifact_index_sha256) or not _is_sha256(
            binding.completion_marker_sha256
        ):
            raise WorkUnitGuardError(
                "dependency_hash_invalid",
                f"Dependency {dependency_id} has a noncanonical pinned hash.",
            )
        root = _absolute_path(binding.output_root, "dependency output_root")
        _reject_rotor_simulation_path(root, "dependency output_root")
        try:
            inventory = artifact_store.audit_finalized_store(root)
        except Exception as error:
            raise WorkUnitGuardError(
                "dependency_store_audit_failed",
                f"Dependency {dependency_id} failed store audit: {error}",
            ) from error
        if (
            inventory.artifact_index_sha256 != binding.artifact_index_sha256
            or inventory.completion_marker_sha256
            != binding.completion_marker_sha256
        ):
            raise WorkUnitGuardError(
                "dependency_hash_mismatch",
                f"Dependency {dependency_id} does not match its pinned store hashes.",
            )
        index = _load_json_object(root / artifact_store.ARTIFACT_INDEX_NAME)
        bindings = _mapping(index.get("bindings"), "dependency store bindings")
        expected_unit = by_plan_id[dependency_id]
        expected_bindings = {
            "work_unit_id": dependency_id,
            "plan_sha256": plan.plan_sha256,
            "protocol_sha256": plan.protocol_sha256,
            "status": "complete",
            "evidence_state": "not_evidence",
        }
        if any(bindings.get(key) != value for key, value in expected_bindings.items()):
            raise WorkUnitGuardError(
                "dependency_binding_mismatch",
                f"Dependency {dependency_id} is not bound to this plan and protocol.",
            )
        required = set(index.get("required_artifacts", ()))
        expected_required = set(expected_unit.required_outputs) | {EXECUTION_RECORD_NAME}
        if not expected_required.issubset(required):
            raise WorkUnitGuardError(
                "dependency_required_output_missing",
                f"Dependency {dependency_id} lacks one or more required work outputs.",
            )
        execution_record = _load_json_object(root / EXECUTION_RECORD_NAME)
        if (
            execution_record.get("status") != "complete"
            or execution_record.get("work_unit_id") != dependency_id
            or execution_record.get("claim_evidence") is not False
            or execution_record.get("evidence_state") != "not_evidence"
        ):
            raise WorkUnitGuardError(
                "dependency_not_complete_candidate",
                f"Dependency {dependency_id} is not a complete non-promoted candidate.",
            )
        ordered.append(binding)
        roots[dependency_id] = root
        input_hashes[
            f"dependency_{dependency_id}_artifact_index_sha256"
        ] = inventory.artifact_index_sha256
        input_hashes[
            f"dependency_{dependency_id}_completion_marker_sha256"
        ] = inventory.completion_marker_sha256
    return tuple(ordered), roots, input_hashes


def _threshold_barrier(
    plan: execution_plan.ExecutionPlan,
    unit: execution_plan.WorkUnit,
    dependencies: Sequence[DependencyBinding],
) -> Optional[evidence_guard.DictionaryFamilyThresholdArtifact]:
    if unit.stage != "synthetic_intervention_test":
        return None
    by_id = {item.work_unit_id: item for item in plan.units}
    calibration_dependencies = tuple(
        item
        for item in dependencies
        if by_id[item.work_unit_id].stage == "synthetic_threshold_calibration"
    )
    if len(calibration_dependencies) != 1:
        raise WorkUnitGuardError(
            "pooled_calibration_barrier_missing",
            "E8 intervention test requires exactly one pooled calibration dependency.",
        )
    binding = calibration_dependencies[0]
    path = Path(binding.output_root).resolve() / "threshold_artifact.json"
    try:
        threshold = evidence_guard.DictionaryFamilyThresholdArtifact.deserialize(
            path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise WorkUnitGuardError(
            "pooled_calibration_barrier_invalid",
            f"Cannot authenticate pooled threshold artifact: {error}",
        ) from error
    if (
        not threshold.human_gate_snapshot
        or threshold.protocol_sha256 != plan.protocol_sha256
        or threshold.validation_split_sha256 != plan.composition_split_sha256
    ):
        raise WorkUnitGuardError(
            "pooled_calibration_barrier_binding_mismatch",
            "Pooled threshold artifact is not bound to this approved protocol and split.",
        )
    return threshold


def _truth_binding(
    unit: execution_plan.WorkUnit,
) -> tuple[
    Optional[path_universe.EquivalenceClass],
    Optional[path_universe.PathRecord],
]:
    if unit.composition_class_id is None:
        return None, None
    matches = tuple(
        item
        for item in path_universe.enumerate_equivalence_classes()
        if item.class_id == unit.composition_class_id
    )
    if len(matches) != 1 or not matches[0].members:
        raise WorkUnitGuardError(
            "truth_class_registry_binding_failed",
            "Synthetic composition is absent from the immutable equivalence registry.",
        )
    truth_class = matches[0]
    truth_path = truth_class.members[0]
    if truth_path.class_id != truth_class.class_id:
        raise WorkUnitGuardError(
            "truth_first_member_binding_failed",
            "The immutable class's first raw member has an invalid class binding.",
        )
    return truth_class, truth_path


def _build_intervention_binding(
    plan: execution_plan.ExecutionPlan,
    unit: execution_plan.WorkUnit,
    dependencies: Sequence[DependencyBinding],
    *,
    truth_class: Optional[path_universe.EquivalenceClass],
) -> Optional[intervention_registry.InterventionRegistry]:
    if unit.stage != "synthetic_intervention_test":
        return None
    if truth_class is None or unit.optimization_seed is None:
        raise WorkUnitGuardError(
            "intervention_registry_binding_missing",
            "E8 intervention unit lacks a truth class or optimization seed.",
        )
    by_id = {item.work_unit_id: item for item in plan.units}
    fit_bindings = tuple(
        item
        for item in dependencies
        if by_id[item.work_unit_id].stage == "synthetic_fit_select"
    )
    if len(fit_bindings) != 1:
        raise WorkUnitGuardError(
            "intervention_fit_dependency_invalid",
            "E8 intervention requires exactly one proposed fit dependency.",
        )
    exported_path = (
        Path(fit_bindings[0].output_root).resolve() / "exported_paths.jsonl"
    )
    try:
        selected = _validation_modal_path_record(exported_path.read_bytes())
    except OSError as error:
        raise WorkUnitGuardError(
            "exported_paths_unreadable",
            f"Cannot read dependency exported paths: {error}",
        ) from error
    try:
        registry = intervention_registry.build_intervention_registry(
            truth_class,
            selected,
            unit.optimization_seed,
        )
        intervention_registry.validate_intervention_registry(registry)
    except Exception as error:
        raise WorkUnitGuardError(
            "intervention_registry_build_failed",
            f"The frozen E8 intervention registry could not be built: {error}",
        ) from error
    return registry


def _validation_modal_path_record(payload: bytes) -> path_universe.PathRecord:
    """Select the seed-2203 modal export with registry-order tie breaking."""

    exported = _validated_checkpoint_export_ids(payload)
    registry_order = {
        item.raw_path_id: index
        for index, item in enumerate(path_universe.enumerate_path_records())
    }
    counts = {identifier: exported.count(identifier) for identifier in set(exported)}
    selected_id = min(
        counts,
        key=lambda identifier: (-counts[identifier], registry_order[identifier]),
    )
    selected = _path_record_by_id(selected_id)
    if selected is None:
        raise WorkUnitGuardError(
            "exported_path_unregistered",
            "Exported selected path is absent from the frozen 216-path registry.",
        )
    return selected


def _stage_access_contract(
    unit: execution_plan.WorkUnit,
) -> tuple[
    tuple[DataRole, ...],
    tuple[synthetic_generator.NuisanceCell, ...],
    tuple[int, ...],
]:
    base = tuple(
        cell
        for cell in synthetic_generator.NUISANCE_CELLS
        if cell.snr_db is None and cell.scale == 1.0 and cell.circular_shift == 0
    )
    if len(base) != 1:
        raise RuntimeError("Frozen base nuisance cell is not unique.")
    sentinels = tuple(
        cell
        for target in ((None, 1.0, 0), (20, 0.5, -32), (10, 2.0, 32))
        for cell in synthetic_generator.NUISANCE_CELLS
        if (cell.snr_db, cell.scale, cell.circular_shift) == target
    )
    if len(sentinels) != 3:
        raise RuntimeError("Frozen E8 sentinel cells are incomplete.")

    if unit.stage == "synthetic_fit_select":
        if unit.arm_id == "full_216_discrete_search":
            return (
                "fit",
                "validation_checkpoint_selection",
            ), base, (1103, 1109, 2203)
        return (
            "fit",
            "validation_checkpoint_selection",
        ), base, (1103, 1109, 2203)
    if unit.stage == "synthetic_threshold_calibration":
        return ("validation_threshold_calibration",), base, (2207,)
    if unit.stage == "synthetic_confirmatory_test":
        return ("confirmatory_test",), tuple(synthetic_generator.NUISANCE_CELLS), (
            3301,
            3307,
        )
    if unit.stage == "synthetic_intervention_test":
        return ("confirmatory_test",), sentinels, (3301, 3307)
    if unit.stage == "cwru_fit_select":
        return ("train", "validation_checkpoint_selection"), (), ()
    if unit.stage == "cwru_confirmatory_test":
        return ("confirmatory_test",), (), ()
    if unit.stage == "dirg_fit_select":
        return ("train", "validation_checkpoint_selection"), (), ()
    if unit.stage == "dirg_confirmatory_test":
        return ("confirmatory_test",), (), ()
    raise WorkUnitGuardError(
        "unsupported_work_stage",
        f"No data-role contract exists for stage {unit.stage!r}.",
    )


def _expected_consumed_optimization_seeds(
    unit: execution_plan.WorkUnit,
) -> tuple[int, ...]:
    if unit.stage == "synthetic_threshold_calibration":
        return FROZEN_OPTIMIZATION_SEEDS
    if unit.optimization_seed is None:
        return ()
    return (unit.optimization_seed,)


def _build_and_validate_cwru_manifest(
    sources: CWRUSourcePaths,
    config: Mapping[str, Any],
) -> cwru_manifest.CWRUManifest:
    if not isinstance(sources, CWRUSourcePaths):
        raise TypeError("cwru_sources must be CWRUSourcePaths.")
    metadata = _absolute_path(sources.metadata_path, "metadata_path")
    raw = _absolute_path(sources.raw_dir, "raw_dir")
    reader = _absolute_path(sources.reader_source_path, "reader_source_path")
    preprocessing = _absolute_path(
        sources.preprocessing_source_path,
        "preprocessing_source_path",
    )
    for label, path in {
        "metadata_path": metadata,
        "raw_dir": raw,
        "reader_source_path": reader,
        "preprocessing_source_path": preprocessing,
    }.items():
        _reject_rotor_simulation_path(path, label)
    if raw.name != cwru_manifest.DATASET_NAME:
        raise WorkUnitGuardError(
            "cwru_raw_directory_invalid",
            f"raw_dir must name the frozen dataset {cwru_manifest.DATASET_NAME}.",
        )
    try:
        manifest = cwru_manifest.build_cwru_manifest(
            metadata_path=metadata,
            raw_dir=raw,
            reader_source_path=reader,
            preprocessing_source_path=preprocessing,
        )
    except Exception as error:
        raise WorkUnitGuardError(
            "cwru_manifest_build_failed",
            f"Read-only CWRU manifest authentication failed: {error}",
        ) from error
    declared = _mapping(config.get("cwru"), "cwru")
    observed = {
        "root_sha256": manifest.root_sha256,
        "metadata_subset_sha256": manifest.metadata_subset_sha256,
        "reader_source_sha256": manifest.reader_source_sha256,
        "preprocessing_source_sha256": manifest.preprocessing_source_sha256,
    }
    if any(declared.get(key) != value for key, value in observed.items()):
        raise WorkUnitGuardError(
            "cwru_dependency_hash_drift",
            "CWRU manifest or source hashes do not match the frozen config.",
        )
    return manifest


def _make_cwru_access_view(
    manifest: cwru_manifest.CWRUManifest,
    sources: CWRUSourcePaths,
    unit: execution_plan.WorkUnit,
) -> CWRUAccessView:
    rotations = {
        "D1": ("007", "014", "021"),
        "D2": ("014", "021", "007"),
        "D3": ("021", "007", "014"),
    }
    expected_rotation = rotations.get(unit.fold_id)
    if expected_rotation is None:
        raise WorkUnitGuardError(
            "cwru_fold_binding_failed",
            f"Unknown execution-plan CWRU fold alias {unit.fold_id!r}.",
        )
    folds = tuple(
        item
        for item in manifest.folds
        if (
            item.train_diameter_code,
            item.validation_diameter_code,
            item.test_diameter_code,
        )
        == expected_rotation
    )
    if len(folds) != 1:
        raise WorkUnitGuardError(
            "cwru_fold_binding_failed",
            f"Manifest does not contain exactly one fold {unit.fold_id!r}.",
        )
    fold = folds[0]
    if unit.stage == "cwru_fit_select":
        split_keys = {
            "train": fold.train_specimen_keys,
            "validation_checkpoint_selection": fold.validation_specimen_keys,
        }
    else:
        split_keys = {"confirmatory_test": fold.test_specimen_keys}
    by_key = {item.specimen_key: item for item in manifest.specimens}
    ordered_keys = tuple(key for keys in split_keys.values() for key in keys)
    specimens = tuple(by_key[key] for key in ordered_keys)
    split_by_key = {
        key: role for role, keys in split_keys.items() for key in keys
    }
    return CWRUAccessView(
        fold_id=unit.fold_id,
        manifest_fold_id=fold.fold_id,
        manifest_root_sha256=manifest.root_sha256,
        allowed_data_roles=tuple(split_keys),
        specimens=specimens,
        split_by_specimen_key=MappingProxyType(split_by_key),
        reader_source_path=Path(sources.reader_source_path).resolve(),
        preprocessing_source_path=Path(sources.preprocessing_source_path).resolve(),
        _raw_dir=Path(sources.raw_dir).resolve(),
    )


def _build_and_validate_dirg_manifest(
    sources: DIRGSourcePaths,
    config: Mapping[str, Any],
) -> dirg_manifest.DIRGManifest:
    if not isinstance(sources, DIRGSourcePaths):
        raise TypeError("dirg_sources must be DIRGSourcePaths.")
    metadata = _absolute_path(sources.metadata_path, "dirg_metadata_path")
    raw = _absolute_path(sources.raw_dir, "dirg_raw_dir")
    reader = _absolute_path(
        sources.reader_source_path,
        "dirg_reader_source_path",
    )
    preprocessing = _absolute_path(
        sources.preprocessing_source_path,
        "dirg_preprocessing_source_path",
    )
    for label, path in {
        "dirg_metadata_path": metadata,
        "dirg_raw_dir": raw,
        "dirg_reader_source_path": reader,
        "dirg_preprocessing_source_path": preprocessing,
    }.items():
        _reject_rotor_simulation_path(path, label)
    if raw.name != dirg_manifest.DATASET_NAME:
        raise WorkUnitGuardError(
            "dirg_raw_directory_invalid",
            f"raw_dir must name the frozen dataset {dirg_manifest.DATASET_NAME}.",
        )
    try:
        manifest = dirg_manifest.build_dirg_manifest(
            metadata_path=metadata,
            raw_dir=raw,
            reader_source_path=reader,
            preprocessing_source_path=preprocessing,
        )
    except Exception as error:
        raise WorkUnitGuardError(
            "dirg_manifest_build_failed",
            f"Read-only DIRG manifest authentication failed: {error}",
        ) from error
    declared = config.get("dirg")
    if not isinstance(declared, Mapping):
        raise WorkUnitGuardError(
            "dirg_dependency_hashes_missing",
            "DIRG execution requires a config mapping of frozen source hashes.",
        )
    observed = {
        "root_sha256": manifest.root_sha256,
        "metadata_file_sha256": manifest.metadata_file_sha256,
        "metadata_name_subset_sha256": manifest.metadata_name_subset_sha256,
        "metadata_selected_subset_sha256": manifest.metadata_selected_subset_sha256,
        "raw_inventory_name_size_sha256": (
            manifest.raw_inventory_name_size_sha256
        ),
        "reader_source_sha256": manifest.reader_source_sha256,
        "preprocessing_source_sha256": manifest.preprocessing_source_sha256,
    }
    if any(declared.get(key) != value for key, value in observed.items()):
        raise WorkUnitGuardError(
            "dirg_dependency_hash_drift",
            "DIRG manifest, metadata, inventory, reader, or preprocessing hashes "
            "do not match the frozen config.",
        )
    return manifest


def _make_dirg_access_view(
    manifest: dirg_manifest.DIRGManifest,
    sources: DIRGSourcePaths,
    unit: execution_plan.WorkUnit,
) -> DIRGAccessView:
    rotations = dict(zip(execution_plan.DIRG_FOLDS, dirg_manifest.ROTATIONS))
    expected_rotation = rotations.get(unit.fold_id)
    if expected_rotation is None:
        raise WorkUnitGuardError(
            "dirg_fold_binding_failed",
            f"Unknown execution-plan DIRG fold alias {unit.fold_id!r}.",
        )
    folds = tuple(
        item
        for item in manifest.folds
        if (
            item.train_severity,
            item.validation_severity,
            item.test_severity,
        )
        == expected_rotation
    )
    if len(folds) != 1:
        raise WorkUnitGuardError(
            "dirg_fold_binding_failed",
            f"Manifest does not contain exactly one fold {unit.fold_id!r}.",
        )
    fold = folds[0]
    if unit.stage == "dirg_fit_select":
        split_keys = {
            "train": fold.train_specimen_keys,
            "validation_checkpoint_selection": fold.validation_specimen_keys,
        }
    else:
        split_keys = {"confirmatory_test": fold.test_specimen_keys}
    by_key = {item.specimen_key: item for item in manifest.specimens}
    ordered_keys = tuple(key for keys in split_keys.values() for key in keys)
    specimens = tuple(by_key[key] for key in ordered_keys)
    split_by_key = {
        key: role for role, keys in split_keys.items() for key in keys
    }
    return DIRGAccessView(
        fold_id=unit.fold_id,
        manifest_fold_id=fold.fold_id,
        manifest_root_sha256=manifest.root_sha256,
        metadata_file_sha256=manifest.metadata_file_sha256,
        metadata_name_subset_sha256=manifest.metadata_name_subset_sha256,
        metadata_selected_subset_sha256=manifest.metadata_selected_subset_sha256,
        raw_inventory_name_size_sha256=manifest.raw_inventory_name_size_sha256,
        reader_source_sha256=manifest.reader_source_sha256,
        preprocessing_source_sha256=manifest.preprocessing_source_sha256,
        allowed_data_roles=tuple(split_keys),
        specimens=specimens,
        split_by_specimen_key=MappingProxyType(split_by_key),
        reader_source_path=Path(sources.reader_source_path).resolve(),
        preprocessing_source_path=Path(sources.preprocessing_source_path).resolve(),
        _raw_dir=Path(sources.raw_dir).resolve(),
    )


def _dispatch_backend(
    backend: WorkUnitBackend,
    context: WorkUnitContext,
) -> BackendExecution:
    handler = getattr(backend, context.work_unit.stage, None)
    if handler is None or not callable(handler):
        raise WorkUnitGuardError(
            "backend_stage_not_implemented",
            f"Backend does not implement {context.work_unit.stage}().",
        )
    result = handler(context)
    if not isinstance(result, BackendExecution):
        raise WorkUnitGuardError(
            "backend_result_type_invalid",
            "Backend stage method must return BackendExecution.",
        )
    return result


def _validate_backend_execution(
    prepared: _Prepared,
    result: BackendExecution,
) -> None:
    context = prepared.context
    unit = prepared.unit
    if tuple(result.accessed_data_roles) != context.allowed_data_roles:
        raise WorkUnitGuardError(
            "data_role_separation_violated",
            "Backend data roles do not match the stage-specific access contract.",
        )
    expected_cells = tuple(cell.cell_id for cell in context.allowed_nuisance_cells)
    if tuple(result.accessed_nuisance_cell_ids) != expected_cells:
        raise WorkUnitGuardError(
            "nuisance_scope_violated",
            "Fit/validation must use only base nuisance; test grids must match exactly.",
        )
    if tuple(result.accessed_generator_seeds) != context.allowed_generator_seeds:
        raise WorkUnitGuardError(
            "generator_seed_role_separation_violated",
            "Generator seed access mixed checkpoint selection, calibration, or test roles.",
        )
    if tuple(result.consumed_optimization_seeds) != (
        context.expected_consumed_optimization_seeds
    ):
        raise WorkUnitGuardError(
            "optimization_seed_consumption_invalid",
            "Backend seed consumption violates exact-25 or deterministic-once policy.",
        )

    if context.truth_path is None:
        if result.truth_raw_path_id is not None or result.exported_raw_path_ids:
            raise WorkUnitGuardError(
                "unexpected_synthetic_path_binding",
                "A non-composition work unit may not declare synthetic truth/export paths.",
            )
    else:
        if result.truth_raw_path_id != context.truth_path.raw_path_id:
            raise WorkUnitGuardError(
                "truth_executor_not_first_raw_member",
                "Synthetic truth executor must be the immutable class registry's first raw member.",
            )
        if not result.exported_raw_path_ids:
            raise WorkUnitGuardError(
                "exported_paths_missing",
                "Synthetic composition execution must report exported raw path IDs.",
            )
        for identifier in result.exported_raw_path_ids:
            record = _path_record_by_id(identifier)
            if record is None:
                raise WorkUnitGuardError(
                    "exported_path_unregistered",
                    "An exported path is absent from the frozen 216-path registry.",
                )

    if context.intervention is None:
        if result.intervention_registry_sha256 is not None:
            raise WorkUnitGuardError(
                "unexpected_intervention_registry",
                "Only E8 intervention work may declare an intervention registry.",
            )
    elif result.intervention_registry_sha256 != context.intervention.manifest_sha256:
        raise WorkUnitGuardError(
            "intervention_registry_hash_mismatch",
            "Backend did not consume the executor-validated intervention registry.",
        )

    if not isinstance(result.input_sha256s, Mapping):
        raise TypeError("Backend input_sha256s must be a mapping.")
    for name, value in result.input_sha256s.items():
        _nonempty_text(name, "backend input hash name")
        if not _is_sha256(value):
            raise WorkUnitGuardError(
                "backend_input_hash_invalid",
                f"Backend input hash {name!r} is not canonical SHA-256.",
            )

    artifacts = tuple(result.artifacts)
    if any(not isinstance(item, BackendArtifact) for item in artifacts):
        raise TypeError("Backend artifacts must contain only BackendArtifact objects.")
    paths = tuple(item.relative_path for item in artifacts)
    if len(set(paths)) != len(paths):
        raise WorkUnitGuardError(
            "backend_output_duplicate",
            "Backend returned duplicate artifact paths.",
        )
    expected_backend_paths = set(unit.required_outputs).difference({"run_meta.yaml"})
    if set(paths) != expected_backend_paths:
        raise WorkUnitGuardError(
            "backend_output_contract_incomplete",
            "Backend outputs must exactly match required outputs except executor-owned run_meta.yaml.",
        )
    for item in artifacts:
        _validate_relative_artifact_path(item.relative_path)
        _nonempty_text(item.role, "backend artifact role")
        if not isinstance(item.payload, bytes) or not item.payload:
            raise WorkUnitGuardError(
                "backend_output_empty_or_invalid",
                f"Backend artifact {item.relative_path!r} must be nonempty bytes.",
            )

    by_path = {item.relative_path: item for item in artifacts}
    if unit.stage == "synthetic_fit_select":
        normalization_item = by_path.get("normalization_artifact.json")
        if normalization_item is None:
            raise WorkUnitGuardError(
                "normalization_artifact_missing",
                "Synthetic fit/select must return normalization_artifact.json.",
            )
        try:
            synthetic_generator.load_normalization_artifact(
                normalization_item.payload
            )
        except Exception as error:
            raise WorkUnitGuardError(
                "normalization_artifact_invalid",
                f"Synthetic fit/select returned invalid fitted normalization state: {error}",
            ) from error
    if "exported_paths.jsonl" in by_path:
        observed_ids = _parse_exported_path_bytes(
            by_path["exported_paths.jsonl"].payload
        )
        if (
            unit.stage == "synthetic_fit_select"
            and unit.arm_id != "full_216_discrete_search"
        ):
            observed_ids = _validated_checkpoint_export_ids(
                by_path["exported_paths.jsonl"].payload
            )
        if observed_ids != tuple(result.exported_raw_path_ids):
            raise WorkUnitGuardError(
                "exported_path_declaration_mismatch",
                "exported_paths.jsonl does not match BackendExecution.exported_raw_path_ids.",
            )
    if unit.stage == "synthetic_threshold_calibration":
        threshold_item = by_path.get("threshold_artifact.json")
        if threshold_item is None:
            raise WorkUnitGuardError(
                "threshold_artifact_missing",
                "Pooled calibration did not return threshold_artifact.json.",
            )
        try:
            threshold = evidence_guard.DictionaryFamilyThresholdArtifact.deserialize(
                threshold_item.payload.decode("utf-8")
            )
        except Exception as error:
            raise WorkUnitGuardError(
                "threshold_artifact_invalid",
                f"Pooled calibration returned an invalid threshold artifact: {error}",
            ) from error
        if (
            not threshold.human_gate_snapshot
            or threshold.protocol_sha256 != prepared.request.plan.protocol_sha256
            or threshold.validation_split_sha256
            != prepared.request.plan.composition_split_sha256
            or threshold.resolved_config_sha256
            != prepared.config.resolved_sha256
        ):
            raise WorkUnitGuardError(
                "threshold_artifact_binding_mismatch",
                "Pooled calibration artifact is not bound to this protocol/config/split.",
            )


def _seal_success(
    prepared: _Prepared,
    backend_result: BackendExecution,
    *,
    backend_source_sha256s: Mapping[str, str],
) -> WorkUnitResult:
    request = prepared.request
    unit = prepared.unit
    if request.output_root is None:  # guarded earlier; keeps type narrowing explicit
        raise AssertionError("execute=True reached sealing without output_root")
    root = _absolute_path(request.output_root, "output_root")
    backend_artifacts = {item.relative_path: item for item in backend_result.artifacts}
    output_hashes = {
        path: hashlib.sha256(item.payload).hexdigest()
        for path, item in backend_artifacts.items()
    }
    combined_inputs = dict(prepared.input_sha256s)
    combined_sources = dict(prepared.source_sha256s)
    combined_sources.update(backend_source_sha256s)
    for name, value in sorted(backend_result.input_sha256s.items()):
        combined_inputs[f"backend_{name}"] = value

    run_meta_bytes: Optional[bytes] = None
    if "run_meta.yaml" in unit.required_outputs:
        run_meta = {
            "schema_version": EXECUTOR_SCHEMA_VERSION,
            "domain": EXECUTOR_DOMAIN,
            "status": "backend_complete_pending_store_seal",
            "claim_evidence": False,
            "evidence_state": "not_evidence",
            "work_unit_id": unit.work_unit_id,
            "stage": unit.stage,
            "arm_id": unit.arm_id,
            "composition_class_id": unit.composition_class_id,
            "fold_id": unit.fold_id,
            "optimization_seed": unit.optimization_seed,
            "protocol_sha256": prepared.request.plan.protocol_sha256,
            "plan_sha256": prepared.request.plan.plan_sha256,
            "resolved_config_sha256": prepared.config.resolved_sha256,
            "runtime_commit": request.runtime_commit,
            "source_sha256s": combined_sources,
            "input_sha256s": combined_inputs,
            "command": list(request.command),
        }
        run_meta_bytes = (_canonical_json(run_meta) + "\n").encode("utf-8")
        output_hashes["run_meta.yaml"] = hashlib.sha256(run_meta_bytes).hexdigest()

    record = {
        "schema_version": EXECUTOR_SCHEMA_VERSION,
        "domain": EXECUTOR_DOMAIN,
        "status": "complete",
        "claim_evidence": False,
        "evidence_state": "not_evidence",
        "work_unit_id": unit.work_unit_id,
        "stage": unit.stage,
        "arm_id": unit.arm_id,
        "composition_class_id": unit.composition_class_id,
        "fold_id": unit.fold_id,
        "optimization_seed": unit.optimization_seed,
        "paired_optimization_seeds": list(FROZEN_OPTIMIZATION_SEEDS),
        "consumed_optimization_seeds": list(
            backend_result.consumed_optimization_seeds
        ),
        "protocol_sha256": request.plan.protocol_sha256,
        "approved_protocol_sha256": request.approved_protocol_sha256,
        "plan_sha256": request.plan.plan_sha256,
        "resolved_config_sha256": prepared.config.resolved_sha256,
        "config_source_sha256": prepared.config.source_sha256,
        "runtime_commit": request.runtime_commit,
        "source_sha256s": combined_sources,
        "input_sha256s": combined_inputs,
        "output_sha256s": dict(sorted(output_hashes.items())),
        "dependency_bindings": [item.to_payload() for item in prepared.dependencies],
        "data_access": {
            "roles": list(backend_result.accessed_data_roles),
            "generator_seeds": list(backend_result.accessed_generator_seeds),
            "nuisance_cell_ids": list(backend_result.accessed_nuisance_cell_ids),
            "raw_access": "read_only",
        },
        "protocol_assertions": {
            "truth_raw_path_id": backend_result.truth_raw_path_id,
            "exported_raw_path_ids": list(backend_result.exported_raw_path_ids),
            "intervention_registry_sha256": (
                backend_result.intervention_registry_sha256
            ),
        },
        "hardware": asdict(request.hardware),
        "command": list(request.command),
    }
    bindings = {
        "work_unit_id": unit.work_unit_id,
        "stage": unit.stage,
        "plan_sha256": request.plan.plan_sha256,
        "protocol_sha256": request.plan.protocol_sha256,
        "approved_protocol_sha256": request.approved_protocol_sha256,
        "resolved_config_sha256": prepared.config.resolved_sha256,
        "runtime_commit": request.runtime_commit,
        "status": "complete",
        "evidence_state": "not_evidence",
    }
    store = artifact_store.DerivedArtifactStore(
        root,
        run_id=unit.work_unit_id,
        protocol_id=PROTOCOL_ID,
        immutable_source_roots=prepared.immutable_source_roots,
        bindings=bindings,
    )
    for relative_path in sorted(backend_artifacts):
        item = backend_artifacts[relative_path]
        store.write_bytes(relative_path, item.payload, role=item.role)
    if run_meta_bytes is not None:
        store.write_bytes(
            "run_meta.yaml",
            run_meta_bytes,
            role="executor_run_provenance",
        )
    execution_digest = store.write_canonical_json(
        EXECUTION_RECORD_NAME,
        record,
        role="executor_execution_record",
    )
    output_hashes[EXECUTION_RECORD_NAME] = execution_digest.sha256
    inventory = store.finalize(
        required_artifacts=tuple(unit.required_outputs) + (EXECUTION_RECORD_NAME,)
    )
    return WorkUnitResult(
        state="complete",
        work_unit_id=unit.work_unit_id,
        stage=unit.stage,
        backend_invoked=True,
        evidence_state="not_evidence",
        reason_codes=(),
        message="Work unit completed and was sealed as a non-promoted candidate artifact.",
        output_root=root,
        failure_record_root=None,
        artifact_index_sha256=inventory.artifact_index_sha256,
        completion_marker_sha256=inventory.completion_marker_sha256,
        output_sha256s=MappingProxyType(dict(sorted(output_hashes.items()))),
        provenance=_result_provenance(
            prepared,
            backend_source_sha256s=backend_source_sha256s,
        ),
    )


def _write_failure_record(
    request: WorkUnitRequest,
    *,
    prepared: Optional[_Prepared],
    backend_invoked: bool,
    reason_code: str,
    error: Exception,
    backend_source_sha256s: Mapping[str, str],
) -> tuple[Optional[Path], Optional[str]]:
    if request.output_root is None:
        return None, "output_root was not supplied"
    try:
        output_root = _absolute_path(request.output_root, "output_root")
        failure_root = output_root.with_name(output_root.name + ".failure")
        _reject_rotor_simulation_path(failure_root, "failure output_root")
        unit = prepared.unit if prepared is not None else _best_effort_unit(request)
        provenance = (
            _result_provenance(prepared)
            if prepared is not None
            else _best_effort_provenance(request)
        )
        roots = (
            prepared.immutable_source_roots
            if prepared is not None
            else _best_effort_immutable_roots(request)
        )
        record = {
            "schema_version": EXECUTOR_SCHEMA_VERSION,
            "domain": EXECUTOR_DOMAIN,
            "status": "failed",
            "claim_evidence": False,
            "evidence_state": "not_evidence",
            "work_unit_id": request.work_unit_id,
            "stage": None if unit is None else unit.stage,
            "arm_id": None if unit is None else unit.arm_id,
            "optimization_seed": None if unit is None else unit.optimization_seed,
            "reason_code": reason_code,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "backend_invoked": backend_invoked,
            "runtime_commit": request.runtime_commit,
            "command": list(request.command),
            "source_sha256s": {
                **dict(provenance.get("source_sha256s", {})),
                **dict(backend_source_sha256s),
            },
            "input_sha256s": dict(provenance.get("input_sha256s", {})),
            "output_sha256s": {},
            "created_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        store = artifact_store.DerivedArtifactStore(
            failure_root,
            run_id=request.work_unit_id + ".failure",
            protocol_id=PROTOCOL_ID,
            immutable_source_roots=roots,
            bindings={
                "work_unit_id": request.work_unit_id,
                "plan_sha256": getattr(request.plan, "plan_sha256", None),
                "protocol_sha256": getattr(request.plan, "protocol_sha256", None),
                "status": "failed",
                "evidence_state": "not_evidence",
                "reason_code": reason_code,
            },
        )
        store.write_canonical_json(
            FAILURE_RECORD_NAME,
            record,
            role="executor_failure_record_not_evidence",
        )
        store.finalize(required_artifacts=(FAILURE_RECORD_NAME,))
        return failure_root, None
    except Exception as failure_error:
        return None, f"{type(failure_error).__name__}: {failure_error}"


def _verify_cwru_sources_unchanged(prepared: _Prepared) -> None:
    sources = prepared.request.cwru_sources
    before = prepared.cwru_manifest_value
    if sources is None or before is None:
        raise AssertionError("CWRU integrity check lacks its pre-execution manifest.")
    after = _build_and_validate_cwru_manifest(sources, prepared.config.value)
    if after != before:
        raise WorkUnitGuardError(
            "cwru_source_mutation_detected",
            "CWRU source bytes or source-code bindings changed during execution.",
        )


def _verify_dirg_sources_unchanged(prepared: _Prepared) -> None:
    sources = prepared.request.dirg_sources
    before = prepared.dirg_manifest_value
    if sources is None or before is None:
        raise AssertionError("DIRG integrity check lacks its pre-execution manifest.")
    try:
        dirg_manifest.verify_dirg_source_bindings(
            before,
            metadata_path=sources.metadata_path,
            raw_dir=sources.raw_dir,
            reader_source_path=sources.reader_source_path,
            preprocessing_source_path=sources.preprocessing_source_path,
        )
    except Exception as error:
        raise WorkUnitGuardError(
            "dirg_source_mutation_detected",
            f"DIRG source bytes or source-code bindings changed during execution: {error}",
        ) from error


def _immutable_source_roots(
    request: WorkUnitRequest,
    config_path: Path,
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in request.immutable_source_roots:
        path = _absolute_path(value, "immutable_source_root")
        _reject_rotor_simulation_path(path, "immutable_source_root")
        roots.append(path)
    roots.append(config_path)
    if request.cwru_sources is not None:
        roots.extend(
            (
                _absolute_path(request.cwru_sources.raw_dir, "raw_dir"),
                _absolute_path(request.cwru_sources.metadata_path, "metadata_path"),
                _absolute_path(
                    request.cwru_sources.reader_source_path,
                    "reader_source_path",
                ),
                _absolute_path(
                    request.cwru_sources.preprocessing_source_path,
                    "preprocessing_source_path",
                ),
            )
        )
    if request.dirg_sources is not None:
        roots.extend(
            (
                _absolute_path(request.dirg_sources.raw_dir, "dirg_raw_dir"),
                _absolute_path(
                    request.dirg_sources.metadata_path,
                    "dirg_metadata_path",
                ),
                _absolute_path(
                    request.dirg_sources.reader_source_path,
                    "dirg_reader_source_path",
                ),
                _absolute_path(
                    request.dirg_sources.preprocessing_source_path,
                    "dirg_preprocessing_source_path",
                ),
            )
        )
    deduplicated = tuple(dict.fromkeys(roots))
    if not deduplicated:
        raise WorkUnitGuardError(
            "immutable_source_roots_missing",
            "At least one immutable source root is required.",
        )
    return deduplicated


def _best_effort_immutable_roots(request: WorkUnitRequest) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if request.cwru_sources is not None:
        candidates.append(Path(request.cwru_sources.raw_dir).resolve(strict=False))
    if request.dirg_sources is not None:
        candidates.extend(
            Path(value).resolve(strict=False)
            for value in (
                request.dirg_sources.raw_dir,
                request.dirg_sources.metadata_path,
                request.dirg_sources.reader_source_path,
                request.dirg_sources.preprocessing_source_path,
            )
        )
    candidates.extend(
        Path(value).resolve(strict=False) for value in request.immutable_source_roots
    )
    candidates.append(Path(request.config_path).resolve(strict=False))
    return tuple(dict.fromkeys(candidates))


def _all_thresholds_approved(config: Mapping[str, Any]) -> bool:
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        return False
    for record in thresholds.values():
        if not isinstance(record, Mapping) or record.get("approved") is not True:
            return False
        value = record.get("value", record.get("values"))
        if isinstance(value, Mapping):
            numeric = tuple(value.values())
        else:
            numeric = (value,)
        if not numeric or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in numeric
        ):
            return False
    return True


def _source_sha256s() -> Mapping[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "execution_plan",
        "experiment_runner",
        "artifact_store",
        "evidence_guard",
        "cwru_manifest",
        "cwru_preprocessing",
        "dirg_manifest",
        "dirg_preprocessing",
        "intervention_registry",
        "statistics_engine",
        "path_universe",
        "synthetic_generator",
        "work_unit_executor",
    )
    return MappingProxyType(
        {name: _sha256_file(directory / f"{name}.py") for name in names}
    )


def _backend_source_sha256s(
    backend: WorkUnitBackend,
    stage: str,
) -> Mapping[str, str]:
    """Hash the exact Python source file that implements the dispatched method."""

    handler = getattr(backend, stage, None)
    if handler is None or not callable(handler):
        # The dispatcher will issue the stage-specific error; do not obscure it here.
        return MappingProxyType({})
    try:
        source_name = inspect.getsourcefile(handler) or inspect.getfile(handler)
    except (TypeError, OSError) as error:
        raise WorkUnitGuardError(
            "backend_source_unavailable",
            f"Cannot resolve backend source for {stage}: {error}",
        ) from error
    source_path = Path(source_name).resolve(strict=False)
    if not source_path.is_file():
        raise WorkUnitGuardError(
            "backend_source_unavailable",
            f"Backend source is not a readable file: {source_path}",
        )
    key = (
        "backend_"
        + type(backend).__module__.replace(".", "_")
        + "_"
        + type(backend).__qualname__.replace(".", "_")
    )
    return MappingProxyType({key: _sha256_file(source_path)})


def _result_provenance(
    prepared: _Prepared,
    *,
    backend_source_sha256s: Mapping[str, str] = MappingProxyType({}),
) -> Mapping[str, Any]:
    sources = dict(prepared.source_sha256s)
    sources.update(backend_source_sha256s)
    return MappingProxyType(
        {
            "protocol_sha256": prepared.request.plan.protocol_sha256,
            "plan_sha256": prepared.request.plan.plan_sha256,
            "resolved_config_sha256": prepared.config.resolved_sha256,
            "config_source_sha256": prepared.config.source_sha256,
            "runtime_commit": prepared.request.runtime_commit,
            "source_sha256s": sources,
            "input_sha256s": dict(prepared.input_sha256s),
            "optimization_seed": prepared.unit.optimization_seed,
            "paired_optimization_seeds": list(FROZEN_OPTIMIZATION_SEEDS),
            "command": list(prepared.request.command),
            "hardware": asdict(prepared.request.hardware),
        }
    )


def _best_effort_provenance(request: WorkUnitRequest) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "runtime_commit": request.runtime_commit,
        "command": list(request.command),
        "source_sha256s": {},
        "input_sha256s": {},
    }
    try:
        result["protocol_sha256"] = request.plan.protocol_sha256
        result["plan_sha256"] = request.plan.plan_sha256
    except Exception:
        pass
    try:
        result["source_sha256s"] = dict(_source_sha256s())
    except Exception:
        pass
    try:
        config = _load_config(request.config_path)
        result["resolved_config_sha256"] = config.resolved_sha256
        result["config_source_sha256"] = config.source_sha256
        result["input_sha256s"] = {
            "config_source_sha256": config.source_sha256,
            "resolved_config_sha256": config.resolved_sha256,
        }
    except Exception:
        pass
    return MappingProxyType(result)


def _blocked_result(
    request: WorkUnitRequest,
    *,
    unit: Optional[execution_plan.WorkUnit],
    reason_codes: Sequence[str],
    message: str,
    provenance: Mapping[str, Any],
) -> WorkUnitResult:
    return WorkUnitResult(
        state="dry_run_blocked",
        work_unit_id=request.work_unit_id,
        stage=None if unit is None else unit.stage,
        backend_invoked=False,
        evidence_state="not_evidence",
        reason_codes=tuple(reason_codes),
        message=message,
        output_root=request.output_root,
        failure_record_root=None,
        artifact_index_sha256=None,
        completion_marker_sha256=None,
        output_sha256s=MappingProxyType({}),
        provenance=provenance,
    )


def _best_effort_unit(
    request: WorkUnitRequest,
) -> Optional[execution_plan.WorkUnit]:
    try:
        return next(
            unit
            for unit in request.plan.units
            if unit.work_unit_id == request.work_unit_id
        )
    except Exception:
        return None


def _read_exported_path_ids(path: Path) -> tuple[str, ...]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise WorkUnitGuardError(
            "exported_paths_unreadable",
            f"Cannot read dependency exported paths: {error}",
        ) from error
    return _parse_exported_path_bytes(payload)


def _validated_checkpoint_export_ids(payload: bytes) -> tuple[str, ...]:
    """Authenticate the complete seed-2203 validation export cohort."""

    records = _parse_exported_path_records(payload)
    expected_ids = tuple(
        path_universe.make_sample_id("validation", 2203, index)
        for index in range(
            path_universe.SAMPLES_PER_GENERATOR_SEED["validation"]
        )
    )
    observed_ids: list[str] = []
    observed_samples: list[str] = []
    for value in records:
        if (
            value.get("generator_seed") != 2203
            or value.get("role") != "validation_checkpoint_selection"
        ):
            raise WorkUnitGuardError(
                "checkpoint_export_role_invalid",
                "Learned fit exports must come only from seed-2203 checkpoint validation.",
            )
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str):
            raise WorkUnitGuardError(
                "checkpoint_export_sample_id_invalid",
                "Every learned fit export requires its registered validation sample ID.",
            )
        observed_samples.append(sample_id)
        observed_ids.append(value["raw_path_id"])
    if tuple(observed_samples) != expected_ids:
        raise WorkUnitGuardError(
            "checkpoint_export_cohort_incomplete_or_reordered",
            "Learned fit exports must cover the exact ordered 128-sample seed-2203 cohort.",
        )
    return tuple(observed_ids)


def _parse_exported_path_bytes(payload: bytes) -> tuple[str, ...]:
    return tuple(
        value["raw_path_id"] for value in _parse_exported_path_records(payload)
    )


def _parse_exported_path_records(payload: bytes) -> tuple[Mapping[str, Any], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise WorkUnitGuardError(
            "exported_paths_invalid",
            "exported_paths.jsonl must be UTF-8.",
        ) from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise WorkUnitGuardError(
            "exported_paths_invalid",
            "exported_paths.jsonl must contain nonempty JSON records.",
        )
    records: list[Mapping[str, Any]] = []
    for line in lines:
        try:
            value = _strict_json_loads(line)
        except ValueError as error:
            raise WorkUnitGuardError(
                "exported_paths_invalid",
                f"Invalid exported-path JSON line: {error}",
            ) from error
        if not isinstance(value, Mapping):
            raise WorkUnitGuardError(
                "exported_paths_invalid",
                "Every exported-path line must be a JSON object.",
            )
        identifier = value.get("raw_path_id")
        if not isinstance(identifier, str) or _path_record_by_id(identifier) is None:
            raise WorkUnitGuardError(
                "exported_paths_invalid",
                "Every exported-path record requires a registered raw_path_id.",
            )
        records.append(value)
    return tuple(records)


def _path_record_by_id(identifier: str) -> Optional[path_universe.PathRecord]:
    return next(
        (
            item
            for item in path_universe.enumerate_path_records()
            if item.raw_path_id == identifier
        ),
        None,
    )


def _load_config(path: Path) -> _LoadedConfig:
    config_path = _absolute_path(path, "config_path")
    _reject_rotor_simulation_path(config_path, "config_path")
    try:
        raw = config_path.read_bytes()
        loaded = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise WorkUnitGuardError(
            "config_load_failed",
            f"Cannot load strict protocol config: {error}",
        ) from error
    if not isinstance(loaded, dict):
        raise WorkUnitGuardError(
            "config_not_mapping",
            "Protocol config must be a YAML mapping.",
        )
    resolved = _canonical_json_value(loaded, "resolved config")
    return _LoadedConfig(
        path=config_path,
        value=MappingProxyType(resolved),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        resolved_sha256=hashlib.sha256(
            _canonical_json(resolved).encode("utf-8")
        ).hexdigest(),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise WorkUnitGuardError(
            "dependency_json_invalid",
            f"Cannot load strict JSON object {path.name}: {error}",
        ) from error
    if not isinstance(value, dict):
        raise WorkUnitGuardError(
            "dependency_json_invalid",
            f"{path.name} must contain a JSON object.",
        )
    return value


def _strict_json_loads(serialized: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        serialized,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkUnitGuardError(
            "config_mapping_missing",
            f"{label} must be a mapping.",
        )
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_json_value(value: Any, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise WorkUnitGuardError(
            "noncanonical_json_value",
            f"{label} must contain finite canonical-JSON values.",
        ) from error


def _validate_relative_artifact_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise WorkUnitGuardError(
            "artifact_path_invalid",
            "Artifact path must be a nonempty normalized POSIX-relative path.",
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or value in {
            artifact_store.ARTIFACT_INDEX_NAME,
            artifact_store.COMPLETION_MARKER_NAME,
            EXECUTION_RECORD_NAME,
            FAILURE_RECORD_NAME,
        }
    ):
        raise WorkUnitGuardError(
            "artifact_path_invalid",
            f"Artifact path is unsafe or reserved: {value!r}.",
        )


def _absolute_path(value: Path | str, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be path-like.")
    path = Path(value)
    if not path.is_absolute():
        raise WorkUnitGuardError(
            "path_not_absolute",
            f"{label} must be absolute.",
        )
    return path.resolve(strict=False)


def _reject_rotor_simulation_path(path: Path, label: str) -> None:
    parts = path.resolve(strict=False).parts
    forbidden = any(
        left == "data" and right == "Rotor_simulation"
        for left, right in zip(parts, parts[1:])
    )
    if forbidden:
        raise WorkUnitGuardError(
            "rotor_simulation_path_forbidden",
            f"{label} may not touch data/Rotor_simulation: {path}",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise WorkUnitGuardError(
            "source_file_unreadable",
            f"Cannot hash source file {path}: {error}",
        ) from error
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text.")
    return value.strip()


__all__ = [
    "BackendArtifact",
    "BackendExecution",
    "CWRUAccessView",
    "CWRUSourcePaths",
    "DIRGAccessView",
    "DIRGSourcePaths",
    "DependencyBinding",
    "EXECUTION_RECORD_NAME",
    "FAILURE_RECORD_NAME",
    "FROZEN_OPTIMIZATION_SEEDS",
    "HardwareRequest",
    "WorkUnitBackend",
    "WorkUnitContext",
    "WorkUnitGuardError",
    "WorkUnitRequest",
    "WorkUnitResult",
    "load_protocol_config",
    "run_work_unit",
]

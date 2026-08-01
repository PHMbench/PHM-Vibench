from __future__ import annotations

import csv
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pytest
import yaml

import src.utils.p07_protocol.work_unit_executor as executor_module
from scripts.p07_execute_work_unit import main as cli_main
from src.utils.p07_protocol import dirg_manifest, path_universe, synthetic_generator
from src.utils.p07_protocol.artifact_store import audit_finalized_store
from src.utils.p07_protocol.execution_plan import (
    ExecutionPlan,
    WorkUnit,
    build_execution_plan,
)
from src.utils.p07_protocol.cwru_manifest import (
    OFFICIAL_12K_DRIVE_END_SPECIMENS,
    WINDOW_COUNT,
    WINDOW_SIZE,
    build_cwru_manifest,
)
from src.utils.p07_protocol.work_unit_executor import (
    BackendArtifact,
    BackendExecution,
    CWRUSourcePaths,
    DIRGSourcePaths,
    DependencyBinding,
    HardwareRequest,
    WorkUnitContext,
    WorkUnitGuardError,
    WorkUnitRequest,
    run_work_unit,
)


PROTOCOL_SHA = "a" * 64
RUNTIME_COMMIT = "c" * 40
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "p07_xoan_operator_attention"
    / "g040_protocol.yaml"
)
DIRG_METADATA_FIELDS = (
    "Id",
    "Dataset_id",
    "Name",
    "TYPE",
    "File",
    "Label",
    "Label_Description",
    "Fault_level",
    "Domain_id",
    "Domain_description",
    "Sample_rate",
    "Sample_lenth",
    "Channel",
)


@lru_cache(maxsize=1)
def _normalization_artifact_bytes() -> bytes:
    sample_ids = tuple(
        path_universe.make_sample_id("fit", 1103, index) for index in range(8)
    )
    return synthetic_generator.estimate_normalization_artifact(
        sample_ids
    ).to_json().encode("utf-8")


def _plan(*, approved: bool) -> ExecutionPlan:
    return build_execution_plan(
        protocol_sha256=PROTOCOL_SHA,
        human_gate_snapshot=approved,
        thresholds_approved_snapshot=approved,
    )


def _approved_config(tmp_path: Path) -> Path:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["protocol_sha256"]["value"] = PROTOCOL_SHA
    config["approval"].update(
        {
            "experiment_protocol_approved": True,
            "thresholds_approved": True,
            "evidence_execution_allowed": True,
            "approved_protocol_sha256": PROTOCOL_SHA,
        }
    )
    for threshold in config["thresholds"].values():
        threshold["approved"] = True
    path = (tmp_path / "approved_protocol.yaml").resolve()
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _cwru_fixture(
    tmp_path: Path,
    config_path: Path,
) -> tuple[CWRUSourcePaths, object]:
    raw_dir = (tmp_path / "cwru" / "raw" / "RM_001_CWRU").resolve()
    raw_dir.mkdir(parents=True)
    metadata_path = (tmp_path / "cwru" / "metadata.csv").resolve()
    fields = (
        "Id",
        "Dataset_id",
        "Name",
        "File",
        "Label",
        "Fault_level",
        "Domain_id",
        "Load_hp",
        "Sample_rate",
        "Sample_lenth",
        "Channel",
    )
    rows = []
    for metadata_id, specimen in enumerate(
        OFFICIAL_12K_DRIVE_END_SPECIMENS,
        start=1001,
    ):
        rows.append(
            {
                "Id": metadata_id,
                "Dataset_id": 1,
                "Name": "RM_001_CWRU",
                "File": specimen.file_name,
                "Label": specimen.label,
                "Fault_level": specimen.fault_level,
                "Domain_id": specimen.domain_id,
                "Load_hp": specimen.load_hp,
                "Sample_rate": 12000,
                "Sample_lenth": WINDOW_COUNT * WINDOW_SIZE + metadata_id * 31,
                "Channel": 2,
            }
        )
        (raw_dir / specimen.file_name).write_bytes(
            b"read-only-cwru-fixture\x00" + specimen.file_name.encode("ascii")
        )
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    reader = (tmp_path / "cwru" / "reader.py").resolve()
    preprocessing = (tmp_path / "cwru" / "preprocessing.py").resolve()
    reader.write_text("def read(path): return path\n", encoding="utf-8")
    preprocessing.write_text("WINDOW_SIZE = 4096\n", encoding="utf-8")
    sources = CWRUSourcePaths(
        metadata_path=metadata_path,
        raw_dir=raw_dir,
        reader_source_path=reader,
        preprocessing_source_path=preprocessing,
    )
    manifest = build_cwru_manifest(
        metadata_path=metadata_path,
        raw_dir=raw_dir,
        reader_source_path=reader,
        preprocessing_source_path=preprocessing,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["cwru"].update(
        {
            "root_sha256": manifest.root_sha256,
            "metadata_subset_sha256": manifest.metadata_subset_sha256,
            "reader_source_sha256": manifest.reader_source_sha256,
            "preprocessing_source_sha256": manifest.preprocessing_source_sha256,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return sources, manifest


def _dirg_fixture(
    tmp_path: Path,
    config_path: Path,
) -> tuple[DIRGSourcePaths, dirg_manifest.DIRGManifest]:
    root = (tmp_path / "dirg").resolve()
    raw_dir = root / "raw" / dirg_manifest.DATASET_NAME
    raw_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    metadata_id = 1_000

    def add_row(
        *,
        file_name: str,
        label: int,
        description: str,
        severity: int,
        domain_id: int,
        sample_rate: int = dirg_manifest.EXPECTED_SAMPLE_RATE_HZ,
        sample_length: int = dirg_manifest.EXPECTED_SAMPLE_LENGTH,
    ) -> None:
        nonlocal metadata_id
        rows.append(
            {
                "Id": metadata_id,
                "Dataset_id": 916,
                "Name": dirg_manifest.DATASET_NAME,
                "TYPE": "Vibration",
                "File": file_name,
                "Label": label,
                "Label_Description": description,
                "Fault_level": severity,
                "Domain_id": domain_id,
                "Domain_description": f"operating condition {domain_id}",
                "Sample_rate": sample_rate,
                "Sample_lenth": sample_length,
                "Channel": dirg_manifest.EXPECTED_CHANNELS,
            }
        )
        metadata_id += 1

    for domain_id in range(1, 18):
        add_row(
            file_name=f"C0A_D{domain_id:02d}.mat",
            label=0,
            description="Healthy bearing (0A)",
            severity=0,
            domain_id=domain_id,
        )
    for condition_id in dirg_manifest.CONDITION_IDS:
        domains = (
            dirg_manifest.DOMAIN_IDS
            if condition_id == "C3"
            else tuple(range(1, 18))
        )
        _, class_name, observed_label = dirg_manifest.CLASS_BY_CONDITION[
            condition_id
        ]
        description = (
            "Inner ring defect, synthetic indentation"
            if class_name == "inner_ring"
            else "Roller defect, synthetic indentation"
        )
        for domain_id in domains:
            add_row(
                file_name=f"{condition_id}A_D{domain_id:02d}.mat",
                label=observed_label,
                description=description,
                severity=dirg_manifest.SEVERITY_BY_CONDITION[condition_id],
                domain_id=domain_id,
            )
    for index in range(65):
        add_row(
            file_name=f"E4A{index:03d}.mat",
            label=2,
            description="Roller defect endurance evolution",
            severity=3,
            domain_id=12,
            sample_rate=102_400,
            sample_length=819_600,
        )
    assert len(rows) == 180

    for row in rows:
        file_name = str(row["File"])
        (raw_dir / file_name).write_bytes(
            b"read-only-dirg-fixture\x00" + file_name.encode("ascii")
        )
    metadata_path = root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIRG_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    reader = root / "RM_020_DIRG.py"
    preprocessing = root / "dirg_preprocessing.py"
    reader.write_text("def read(path): return path\n", encoding="utf-8")
    preprocessing.write_text(
        "WINDOW_ALGORITHM_ID = 'p07-evenly-distributed-nonoverlap-v1'\n",
        encoding="utf-8",
    )
    sources = DIRGSourcePaths(
        metadata_path=metadata_path.resolve(),
        raw_dir=raw_dir.resolve(),
        reader_source_path=reader.resolve(),
        preprocessing_source_path=preprocessing.resolve(),
    )
    manifest = dirg_manifest.build_dirg_manifest(
        metadata_path=sources.metadata_path,
        raw_dir=sources.raw_dir,
        reader_source_path=sources.reader_source_path,
        preprocessing_source_path=sources.preprocessing_source_path,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dirg"] = {
        "dataset_name": dirg_manifest.DATASET_NAME,
        "subset_id": dirg_manifest.SUBSET_ID,
        "root_sha256": manifest.root_sha256,
        "metadata_file_sha256": manifest.metadata_file_sha256,
        "metadata_name_subset_sha256": manifest.metadata_name_subset_sha256,
        "metadata_selected_subset_sha256": (
            manifest.metadata_selected_subset_sha256
        ),
        "raw_inventory_name_size_sha256": (
            manifest.raw_inventory_name_size_sha256
        ),
        "reader_source_sha256": manifest.reader_source_sha256,
        "preprocessing_source_sha256": manifest.preprocessing_source_sha256,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return sources, manifest


def _request(
    *,
    plan: ExecutionPlan,
    unit: WorkUnit,
    config_path: Path,
    execute: bool = False,
    output_root: Path | None = None,
    dependencies: tuple[DependencyBinding, ...] = (),
    hardware: HardwareRequest = HardwareRequest(),
    cwru_sources: CWRUSourcePaths | None = None,
    dirg_sources: DIRGSourcePaths | None = None,
    immutable_source_roots: tuple[Path, ...] = (),
) -> WorkUnitRequest:
    return WorkUnitRequest(
        plan=plan,
        work_unit_id=unit.work_unit_id,
        config_path=config_path.resolve(),
        approved_protocol_sha256=(PROTOCOL_SHA if plan.human_gate_snapshot else None),
        runtime_commit=RUNTIME_COMMIT,
        command=("python", "p07_execute_work_unit.py", "--unit-id", unit.work_unit_id),
        execute=execute,
        output_root=None if output_root is None else output_root.resolve(),
        dependencies=dependencies,
        immutable_source_roots=immutable_source_roots,
        cwru_sources=cwru_sources,
        dirg_sources=dirg_sources,
        hardware=hardware,
    )


class RecordingBackend:
    def __init__(
        self,
        mutate: Callable[[WorkUnitContext, BackendExecution], BackendExecution]
        | None = None,
    ) -> None:
        self.contexts: list[WorkUnitContext] = []
        self._mutate = mutate

    def _run(self, context: WorkUnitContext) -> BackendExecution:
        self.contexts.append(context)
        truth_id = None if context.truth_path is None else context.truth_path.raw_path_id
        learned_synthetic_fit = (
            context.work_unit.stage == "synthetic_fit_select"
            and context.work_unit.arm_id != "full_216_discrete_search"
        )
        exported = (
            ()
            if truth_id is None
            else ((truth_id,) * 128 if learned_synthetic_fit else (truth_id,))
        )
        artifacts: list[BackendArtifact] = []
        for relative_path in context.work_unit.required_outputs:
            if relative_path == "run_meta.yaml":
                continue
            if relative_path == "exported_paths.jsonl":
                if learned_synthetic_fit:
                    payload = "".join(
                        json.dumps(
                            {
                                "generator_seed": 2203,
                                "raw_path_id": truth_id,
                                "role": "validation_checkpoint_selection",
                                "sample_id": path_universe.make_sample_id(
                                    "validation", 2203, index
                                ),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                        for index in range(128)
                    ).encode("utf-8")
                else:
                    payload = (
                        json.dumps(
                            {"raw_path_id": truth_id},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
            elif relative_path == "normalization_artifact.json":
                payload = _normalization_artifact_bytes()
            elif relative_path.endswith(".json"):
                payload = b"{}\n"
            else:
                payload = (relative_path + "\n").encode("utf-8")
            artifacts.append(
                BackendArtifact(
                    relative_path=relative_path,
                    payload=payload,
                    role="test_backend_output",
                )
            )
        result = BackendExecution(
            artifacts=tuple(artifacts),
            accessed_data_roles=context.allowed_data_roles,
            accessed_nuisance_cell_ids=tuple(
                item.cell_id for item in context.allowed_nuisance_cells
            ),
            accessed_generator_seeds=context.allowed_generator_seeds,
            consumed_optimization_seeds=context.expected_consumed_optimization_seeds,
            input_sha256s={"fixture_input": "d" * 64},
            truth_raw_path_id=truth_id,
            exported_raw_path_ids=exported,
            intervention_registry_sha256=(
                None
                if context.intervention is None
                else context.intervention.manifest_sha256
            ),
        )
        return result if self._mutate is None else self._mutate(context, result)

    synthetic_fit_select = _run
    synthetic_confirmatory_test = _run
    synthetic_threshold_calibration = _run
    synthetic_intervention_test = _run
    cwru_fit_select = _run
    cwru_confirmatory_test = _run
    dirg_fit_select = _run
    dirg_confirmatory_test = _run


def _first_unit(plan: ExecutionPlan, *, stage: str, arm: str | None = None) -> WorkUnit:
    return next(
        unit
        for unit in plan.units
        if unit.stage == stage and (arm is None or unit.arm_id == arm)
    )


def _dependency(result_root: Path, result: object, unit: WorkUnit) -> DependencyBinding:
    return DependencyBinding(
        work_unit_id=unit.work_unit_id,
        output_root=result_root.resolve(),
        artifact_index_sha256=getattr(result, "artifact_index_sha256"),
        completion_marker_sha256=getattr(result, "completion_marker_sha256"),
    )


def _checkpoint_export_payload(
    raw_path_ids: tuple[str, ...],
    *,
    role: str = "validation_checkpoint_selection",
    generator_seed: int = 2203,
) -> bytes:
    assert len(raw_path_ids) == 128
    return "".join(
        json.dumps(
            {
                "generator_seed": generator_seed,
                "raw_path_id": raw_path_id,
                "role": role,
                "sample_id": path_universe.make_sample_id(
                    "validation", 2203, index
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for index, raw_path_id in enumerate(raw_path_ids)
    ).encode("utf-8")


def test_seed_2203_modal_export_uses_registry_tie_and_rejects_role_mixing() -> None:
    records = path_universe.enumerate_path_records()
    first_id = records[0].raw_path_id
    second_id = records[1].raw_path_id
    tied = (second_id, first_id) * 64

    selected = executor_module._validation_modal_path_record(
        _checkpoint_export_payload(tied)
    )
    assert selected.raw_path_id == first_id

    with pytest.raises(
        WorkUnitGuardError,
        match="seed-2203 checkpoint validation",
    ):
        executor_module._validation_modal_path_record(
            _checkpoint_export_payload(
                tied,
                role="validation_threshold_calibration",
                generator_seed=2207,
            )
        )


def test_default_dry_run_is_read_only_and_false_gate_never_calls_backend(
    tmp_path: Path,
) -> None:
    plan = _plan(approved=False)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    output = tmp_path / "would-be-output"
    backend = RecordingBackend()

    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=DEFAULT_CONFIG,
            output_root=output,
        ),
        backend=backend,
    )

    assert result.state == "dry_run_blocked"
    assert "human_gate_not_approved" in result.reason_codes
    assert result.backend_invoked is False
    assert backend.contexts == []
    assert not output.exists()
    assert not output.with_name(output.name + ".failure").exists()


def test_approved_fit_executes_with_first_truth_member_base_nuisance_and_2203_selection(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    raw = (tmp_path / "immutable-source").resolve()
    raw.mkdir()
    output = (tmp_path / "fit-output").resolve()
    backend = RecordingBackend()

    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            execute=True,
            output_root=output,
            immutable_source_roots=(raw,),
        ),
        backend=backend,
    )

    assert result.state == "complete", (result.reason_codes, result.message)
    assert result.evidence_state == "not_evidence"
    assert len(backend.contexts) == 1
    context = backend.contexts[0]
    assert context.truth_path == context.truth_class.members[0]
    assert context.allowed_data_roles == (
        "fit",
        "validation_checkpoint_selection",
    )
    assert context.allowed_generator_seeds == (1103, 1109, 2203)
    assert 2207 not in context.allowed_generator_seeds
    assert len(context.allowed_nuisance_cells) == 1
    base = context.allowed_nuisance_cells[0]
    assert (base.snr_db, base.scale, base.circular_shift) == (None, 1.0, 0)

    audited = audit_finalized_store(output)
    assert audited.artifact_index_sha256 == result.artifact_index_sha256
    execution_record = json.loads(
        (output / "execution_record.json").read_text(encoding="utf-8")
    )
    assert execution_record["status"] == "complete"
    assert execution_record["claim_evidence"] is False
    assert execution_record["evidence_state"] == "not_evidence"
    assert execution_record["runtime_commit"] == RUNTIME_COMMIT
    assert execution_record["command"] == list(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
        ).command
    )
    assert execution_record["data_access"]["generator_seeds"] == [1103, 1109, 2203]
    assert execution_record["protocol_assertions"]["truth_raw_path_id"] == (
        context.truth_class.members[0].raw_path_id
    )
    assert execution_record["source_sha256s"]
    assert execution_record["input_sha256s"]
    assert execution_record["output_sha256s"]
    normalization_path = output / "normalization_artifact.json"
    assert normalization_path.is_file()
    synthetic_generator.load_normalization_artifact(
        normalization_path.read_bytes()
    )


def test_seed_role_violation_writes_separate_not_evidence_failure(
    tmp_path: Path,
) -> None:
    reason = "generator_seed_role_separation_violated"
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")

    def mutator(
        _context: WorkUnitContext, result: BackendExecution
    ) -> BackendExecution:
        return replace(
            result,
            accessed_generator_seeds=(1103, 1109, 2203, 2207),
        )

    output = (tmp_path / f"bad-{reason}").resolve()
    immutable = (tmp_path / f"raw-{reason}").resolve()
    immutable.mkdir()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            execute=True,
            output_root=output,
            immutable_source_roots=(immutable,),
        ),
        backend=RecordingBackend(mutator),
    )

    assert result.state == "failed"
    assert reason in result.reason_codes
    assert not output.exists()
    assert result.failure_record_root == output.with_name(output.name + ".failure")
    audited = audit_finalized_store(result.failure_record_root)
    assert [item.relative_path for item in audited.artifacts] == ["failure_record.json"]
    failure = json.loads(
        (result.failure_record_root / "failure_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["status"] == "failed"
    assert failure["evidence_state"] == "not_evidence"
    assert failure["output_sha256s"] == {}


@pytest.mark.parametrize(
    ("mode", "reason"),
    (
        ("missing", "backend_output_contract_incomplete"),
        ("invalid", "normalization_artifact_invalid"),
    ),
)
def test_synthetic_fit_normalization_artifact_is_mandatory_and_self_hashed(
    tmp_path: Path,
    mode: str,
    reason: str,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")

    def break_normalization(
        _context: WorkUnitContext,
        result: BackendExecution,
    ) -> BackendExecution:
        if mode == "missing":
            artifacts = tuple(
                item
                for item in result.artifacts
                if item.relative_path != "normalization_artifact.json"
            )
        else:
            artifacts = tuple(
                replace(item, payload=b"{}")
                if item.relative_path == "normalization_artifact.json"
                else item
                for item in result.artifacts
            )
        return replace(result, artifacts=artifacts)

    immutable = (tmp_path / f"immutable-normalization-{mode}").resolve()
    immutable.mkdir()
    output = (tmp_path / f"normalization-{mode}").resolve()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            execute=True,
            output_root=output,
            immutable_source_roots=(immutable,),
        ),
        backend=RecordingBackend(break_normalization),
    )

    assert result.state == "failed"
    assert reason in result.reason_codes
    assert result.backend_invoked is True
    assert not output.exists()
    assert result.failure_record_root is not None


def test_registered_wrong_recovery_is_retained_as_a_complete_run(
    tmp_path: Path,
) -> None:
    from src.utils.p07_protocol.path_universe import enumerate_path_records

    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    wrong_id = next(
        item.raw_path_id
        for item in enumerate_path_records()
        if item.class_id != unit.composition_class_id
    )

    def retain_wrong_export(
        context: WorkUnitContext, result: BackendExecution
    ) -> BackendExecution:
        payload = "".join(
            json.dumps(
                {
                    "generator_seed": 2203,
                    "raw_path_id": wrong_id,
                    "role": "validation_checkpoint_selection",
                    "sample_id": path_universe.make_sample_id(
                        "validation", 2203, index
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for index in range(128)
        ).encode("utf-8")
        artifacts = tuple(
            BackendArtifact(
                relative_path=item.relative_path,
                payload=(
                    payload
                    if item.relative_path == "exported_paths.jsonl"
                    else item.payload
                ),
                role=item.role,
            )
            for item in result.artifacts
        )
        return replace(
            result,
            artifacts=artifacts,
            truth_raw_path_id=context.truth_path.raw_path_id,
            exported_raw_path_ids=(wrong_id,) * 128,
        )

    immutable = (tmp_path / "immutable-wrong-recovery").resolve()
    immutable.mkdir()
    output = (tmp_path / "retained-wrong-recovery").resolve()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            execute=True,
            output_root=output,
            immutable_source_roots=(immutable,),
        ),
        backend=RecordingBackend(retain_wrong_export),
    )

    assert result.state == "complete", (result.reason_codes, result.message)
    assert result.evidence_state == "not_evidence"
    assert output.exists()
    assert result.failure_record_root is None


def test_unapproved_confirmatory_unit_is_never_dispatched(tmp_path: Path) -> None:
    plan = _plan(approved=False)
    unit = _first_unit(plan, stage="synthetic_confirmatory_test", arm="proposed")
    output = (tmp_path / "blocked-confirmatory").resolve()
    backend = RecordingBackend()

    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=DEFAULT_CONFIG,
            execute=True,
            output_root=output,
        ),
        backend=backend,
    )

    assert result.state == "failed"
    assert "human_gate_not_approved" in result.reason_codes
    assert backend.contexts == []
    assert not output.exists()
    assert result.failure_record_root is not None


def test_dependency_hash_mismatch_fails_before_confirmatory_dispatch(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    fit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    immutable = (tmp_path / "immutable").resolve()
    immutable.mkdir()
    fit_root = (tmp_path / "fit").resolve()
    fit_result = run_work_unit(
        _request(
            plan=plan,
            unit=fit,
            config_path=config_path,
            execute=True,
            output_root=fit_root,
            immutable_source_roots=(immutable,),
        ),
        backend=RecordingBackend(),
    )
    assert fit_result.state == "complete"
    confirmatory = next(
        item
        for item in plan.units
        if item.stage == "synthetic_confirmatory_test"
        and item.depends_on == (fit.work_unit_id,)
    )
    bad_dependency = DependencyBinding(
        work_unit_id=fit.work_unit_id,
        output_root=fit_root,
        artifact_index_sha256="f" * 64,
        completion_marker_sha256=fit_result.completion_marker_sha256,
    )
    backend = RecordingBackend()
    output = (tmp_path / "confirmatory").resolve()

    result = run_work_unit(
        _request(
            plan=plan,
            unit=confirmatory,
            config_path=config_path,
            execute=True,
            output_root=output,
            dependencies=(bad_dependency,),
            immutable_source_roots=(immutable,),
        ),
        backend=backend,
    )

    assert result.state == "failed"
    assert "dependency_hash_mismatch" in result.reason_codes
    assert backend.contexts == []
    assert not output.exists()


def test_approved_confirmatory_dispatch_requires_and_accepts_exact_sealed_dependency(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    fit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    immutable = (tmp_path / "immutable-success").resolve()
    immutable.mkdir()
    fit_root = (tmp_path / "fit-success").resolve()
    fit_result = run_work_unit(
        _request(
            plan=plan,
            unit=fit,
            config_path=config_path,
            execute=True,
            output_root=fit_root,
            immutable_source_roots=(immutable,),
        ),
        backend=RecordingBackend(),
    )
    assert fit_result.state == "complete"
    confirmatory = next(
        item
        for item in plan.units
        if item.stage == "synthetic_confirmatory_test"
        and item.depends_on == (fit.work_unit_id,)
    )
    backend = RecordingBackend()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=confirmatory,
            config_path=config_path,
            execute=True,
            output_root=(tmp_path / "confirmatory-success").resolve(),
            dependencies=(_dependency(fit_root, fit_result, fit),),
            immutable_source_roots=(immutable,),
        ),
        backend=backend,
    )

    assert result.state == "complete"
    assert len(backend.contexts) == 1
    context = backend.contexts[0]
    assert context.allowed_data_roles == ("confirmatory_test",)
    assert context.allowed_generator_seeds == (3301, 3307)
    assert len(context.allowed_nuisance_cells) == 27


def test_full216_is_seed_invariant_once_per_composition_and_consumes_no_fake_seed(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    full = _first_unit(
        plan,
        stage="synthetic_fit_select",
        arm="full_216_discrete_search",
    )
    assert sum(
        unit.stage == "synthetic_fit_select"
        and unit.arm_id == "full_216_discrete_search"
        for unit in plan.units
    ) == 18
    assert sum(
        unit.stage == "cwru_fit_select"
        and unit.arm_id == "full_216_discrete_search"
        for unit in plan.units
    ) == 3
    assert sum(
        unit.stage == "dirg_fit_select"
        and unit.arm_id == "full_216_discrete_search"
        for unit in plan.units
    ) == 3
    assert len(plan.units) == 3_799
    backend = RecordingBackend()
    immutable = (tmp_path / "immutable").resolve()
    immutable.mkdir()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=full,
            config_path=config_path,
            execute=True,
            output_root=(tmp_path / "full216").resolve(),
            immutable_source_roots=(immutable,),
        ),
        backend=backend,
    )

    assert result.state == "complete"
    context = backend.contexts[0]
    assert full.optimization_seed is None
    assert context.expected_consumed_optimization_seeds == ()
    assert context.allowed_generator_seeds == (1103, 1109, 2203)
    assert context.allowed_data_roles == (
        "fit",
        "validation_checkpoint_selection",
    )
    assert context.deterministic_bookkeeping_seed is not None


def test_e8_intervention_requires_exact_dependency_closure_and_one_barrier(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_intervention_test")
    backend = RecordingBackend()

    result = run_work_unit(
        _request(plan=plan, unit=unit, config_path=config_path),
        backend=backend,
    )

    assert result.state == "dry_run_blocked"
    assert "dependency_closure_incomplete" in result.reason_codes
    assert backend.contexts == []


@pytest.mark.parametrize(
    "hardware",
    (
        HardwareRequest(device="cuda", physical_gpu_index=2),
        HardwareRequest(device="cuda", physical_gpu_index=0, world_size=2),
        HardwareRequest(
            device="cuda",
            physical_gpu_index=0,
            distributed_backend="nccl",
        ),
    ),
)
def test_gpu2_multigpu_and_ddp_fail_closed(
    tmp_path: Path,
    hardware: HardwareRequest,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            hardware=hardware,
        )
    )
    assert result.state == "dry_run_blocked"
    assert set(result.reason_codes).intersection(
        {"physical_gpu_2_forbidden", "multi_gpu_or_ddp_forbidden"}
    )


def test_cwru_rotor_simulation_path_is_rejected_without_touching_backend(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="cwru_fit_select", arm="proposed")
    forbidden = (tmp_path / "data" / "Rotor_simulation").resolve()
    backend = RecordingBackend()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            cwru_sources=CWRUSourcePaths(
                metadata_path=forbidden / "metadata.csv",
                raw_dir=forbidden / "RM_001_CWRU",
                reader_source_path=forbidden / "reader.py",
                preprocessing_source_path=forbidden / "preprocessing.py",
            ),
        ),
        backend=backend,
    )
    assert result.state == "dry_run_blocked"
    assert "rotor_simulation_path_forbidden" in result.reason_codes
    assert backend.contexts == []


def test_cwru_executor_exposes_only_fit_validation_files_and_preserves_raw_bytes(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    sources, manifest = _cwru_fixture(tmp_path, config_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="cwru_fit_select", arm="proposed")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sources.raw_dir.iterdir()
        if path.is_file()
    }
    backend = RecordingBackend()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            execute=True,
            output_root=(tmp_path / "cwru-fit-output").resolve(),
            cwru_sources=sources,
        ),
        backend=backend,
    )

    assert result.state == "complete", (result.reason_codes, result.message)
    context = backend.contexts[0]
    assert context.cwru_access is not None
    assert context.cwru_access.allowed_data_roles == (
        "train",
        "validation_checkpoint_selection",
    )
    assert len(context.cwru_access.specimens) == 24
    assert context.cwru_access.fold_id == unit.fold_id
    fold = next(
        item
        for item in manifest.folds
        if item.fold_id == context.cwru_access.manifest_fold_id
    )
    assert not set(fold.test_specimen_keys).intersection(
        item.specimen_key for item in context.cwru_access.specimens
    )
    allowed = context.cwru_access.specimens[0]
    assert context.cwru_access.raw_path(allowed.specimen_key) == (
        sources.raw_dir / allowed.file_name
    )
    with pytest.raises(WorkUnitGuardError, match="outside.*authorized fold view"):
        context.cwru_access.raw_path(fold.test_specimen_keys[0])
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sources.raw_dir.iterdir()
        if path.is_file()
    }
    assert after == before


def test_cwru_and_dirg_source_capabilities_cannot_cross_datasets(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    cwru_unit = _first_unit(plan, stage="cwru_fit_select", arm="proposed")
    dirg_unit = _first_unit(plan, stage="dirg_fit_select", arm="proposed")
    absent = (tmp_path / "must-not-be-opened").resolve()
    cwru_sources = CWRUSourcePaths(
        metadata_path=absent / "cwru.csv",
        raw_dir=absent / "RM_001_CWRU",
        reader_source_path=absent / "cwru_reader.py",
        preprocessing_source_path=absent / "cwru_preprocessing.py",
    )
    dirg_sources = DIRGSourcePaths(
        metadata_path=absent / "dirg.csv",
        raw_dir=absent / dirg_manifest.DATASET_NAME,
        reader_source_path=absent / "dirg_reader.py",
        preprocessing_source_path=absent / "dirg_preprocessing.py",
    )
    backend = RecordingBackend()

    wrong_for_dirg = run_work_unit(
        _request(
            plan=plan,
            unit=dirg_unit,
            config_path=config_path,
            cwru_sources=cwru_sources,
        ),
        backend=backend,
    )
    wrong_for_cwru = run_work_unit(
        _request(
            plan=plan,
            unit=cwru_unit,
            config_path=config_path,
            dirg_sources=dirg_sources,
        ),
        backend=backend,
    )

    assert wrong_for_dirg.state == wrong_for_cwru.state == "dry_run_blocked"
    assert wrong_for_dirg.reason_codes == ("unexpected_cwru_sources",)
    assert wrong_for_cwru.reason_codes == ("unexpected_dirg_sources",)
    assert backend.contexts == []
    assert not absent.exists()


def test_dirg_rotor_simulation_path_is_rejected_without_touching_backend(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="dirg_fit_select", arm="proposed")
    forbidden = (tmp_path / "data" / "Rotor_simulation").resolve()
    backend = RecordingBackend()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            dirg_sources=DIRGSourcePaths(
                metadata_path=forbidden / "metadata.csv",
                raw_dir=forbidden / dirg_manifest.DATASET_NAME,
                reader_source_path=forbidden / "reader.py",
                preprocessing_source_path=forbidden / "preprocessing.py",
            ),
        ),
        backend=backend,
    )
    assert result.state == "dry_run_blocked"
    assert "rotor_simulation_path_forbidden" in result.reason_codes
    assert backend.contexts == []
    assert not forbidden.exists()


def test_dirg_fit_and_confirmatory_dispatch_are_fold_scoped_hashed_and_read_only(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    sources, manifest = _dirg_fixture(tmp_path, config_path)
    plan = _plan(approved=True)
    fit = next(
        unit
        for unit in plan.units
        if unit.stage == "dirg_fit_select"
        and unit.arm_id == "proposed"
        and unit.fold_id == "S1"
    )
    source_files = (
        sources.metadata_path,
        sources.reader_source_path,
        sources.preprocessing_source_path,
        *tuple(sorted(path for path in sources.raw_dir.iterdir() if path.is_file())),
    )
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source_files
    }
    fit_root = (tmp_path / "dirg-fit-output").resolve()
    fit_backend = RecordingBackend()
    fit_result = run_work_unit(
        _request(
            plan=plan,
            unit=fit,
            config_path=config_path,
            execute=True,
            output_root=fit_root,
            dirg_sources=sources,
        ),
        backend=fit_backend,
    )

    assert fit_result.state == "complete", (
        fit_result.reason_codes,
        fit_result.message,
    )
    fit_context = fit_backend.contexts[0]
    assert fit_context.cwru_access is None
    assert fit_context.dirg_access is not None
    fit_access = fit_context.dirg_access
    assert fit_access.fold_id == "S1"
    assert fit_access.allowed_data_roles == (
        "train",
        "validation_checkpoint_selection",
    )
    assert len(fit_access.specimens) == 2 * dirg_manifest.FILES_PER_SPLIT
    manifest_fold = next(
        item for item in manifest.folds if item.fold_id == fit_access.manifest_fold_id
    )
    assert not set(manifest_fold.test_specimen_keys).intersection(
        item.specimen_key for item in fit_access.specimens
    )
    allowed = fit_access.specimens[0]
    assert fit_access.raw_path(allowed.specimen_key) == (
        sources.raw_dir / allowed.file_name
    )
    with pytest.raises(WorkUnitGuardError, match="outside.*authorized fold view"):
        fit_access.raw_path(manifest_fold.test_specimen_keys[0])
    expected_hashes = {
        "dirg_manifest_sha256": manifest.root_sha256,
        "dirg_metadata_file_sha256": manifest.metadata_file_sha256,
        "dirg_metadata_name_subset_sha256": manifest.metadata_name_subset_sha256,
        "dirg_metadata_selected_subset_sha256": (
            manifest.metadata_selected_subset_sha256
        ),
        "dirg_raw_inventory_name_size_sha256": (
            manifest.raw_inventory_name_size_sha256
        ),
        "dirg_reader_source_sha256": manifest.reader_source_sha256,
        "dirg_preprocessing_source_sha256": manifest.preprocessing_source_sha256,
    }
    assert {
        key: fit_result.provenance["input_sha256s"][key]
        for key in expected_hashes
    } == expected_hashes
    assert {
        "dirg_manifest",
        "dirg_preprocessing",
        "work_unit_executor",
    }.issubset(fit_result.provenance["source_sha256s"])

    confirmatory = next(
        unit
        for unit in plan.units
        if unit.stage == "dirg_confirmatory_test"
        and unit.depends_on == (fit.work_unit_id,)
    )
    confirmatory_backend = RecordingBackend()
    confirmatory_result = run_work_unit(
        _request(
            plan=plan,
            unit=confirmatory,
            config_path=config_path,
            execute=True,
            output_root=(tmp_path / "dirg-confirmatory-output").resolve(),
            dependencies=(_dependency(fit_root, fit_result, fit),),
            dirg_sources=sources,
        ),
        backend=confirmatory_backend,
    )
    assert confirmatory_result.state == "complete", (
        confirmatory_result.reason_codes,
        confirmatory_result.message,
    )
    confirmatory_access = confirmatory_backend.contexts[0].dirg_access
    assert confirmatory_access is not None
    assert confirmatory_access.allowed_data_roles == ("confirmatory_test",)
    assert len(confirmatory_access.specimens) == dirg_manifest.FILES_PER_SPLIT
    assert {item.specimen_key for item in confirmatory_access.specimens} == set(
        manifest_fold.test_specimen_keys
    )
    with pytest.raises(WorkUnitGuardError, match="outside.*authorized fold view"):
        confirmatory_access.raw_path(manifest_fold.train_specimen_keys[0])

    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source_files
    }
    assert after == before


def test_dirg_config_hash_drift_fails_before_backend_dispatch(
    tmp_path: Path,
) -> None:
    config_path = _approved_config(tmp_path)
    sources, _manifest = _dirg_fixture(tmp_path, config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dirg"]["reader_source_sha256"] = "f" * 64
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="dirg_fit_select", arm="proposed")
    backend = RecordingBackend()
    result = run_work_unit(
        _request(
            plan=plan,
            unit=unit,
            config_path=config_path,
            dirg_sources=sources,
        ),
        backend=backend,
    )
    assert result.state == "dry_run_blocked"
    assert "dirg_dependency_hash_drift" in result.reason_codes
    assert backend.contexts == []


def test_seed_cohort_drift_is_rejected_before_dispatch(tmp_path: Path) -> None:
    config_path = _approved_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seeds"]["optimization"] = config["seeds"]["optimization"][:-1]
    config["seeds"]["optimization_count"] = 24
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    plan = _plan(approved=True)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    backend = RecordingBackend()
    result = run_work_unit(
        _request(plan=plan, unit=unit, config_path=config_path), backend=backend
    )
    assert result.state == "dry_run_blocked"
    assert "seed_cohort_not_exactly_frozen_25" in result.reason_codes
    assert backend.contexts == []


def test_cli_defaults_to_read_only_and_reports_gate_block(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _plan(approved=False)
    unit = _first_unit(plan, stage="synthetic_fit_select", arm="proposed")
    output = (tmp_path / "cli-output").resolve()
    code = cli_main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--protocol-sha256",
            PROTOCOL_SHA,
            "--unit-id",
            unit.work_unit_id,
            "--runtime-commit",
            RUNTIME_COMMIT,
            "--output-root",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert payload["state"] == "dry_run_blocked"
    assert payload["backend_invoked"] is False
    assert payload["evidence_state"] == "not_evidence"
    assert not output.exists()


def test_cli_dirg_arguments_are_dataset_specific_read_only_and_all_or_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _plan(approved=False)
    unit = _first_unit(plan, stage="dirg_fit_select", arm="proposed")
    absent = (tmp_path / "dirg-not-opened").resolve()
    common = [
        "--config",
        str(DEFAULT_CONFIG),
        "--protocol-sha256",
        PROTOCOL_SHA,
        "--unit-id",
        unit.work_unit_id,
        "--runtime-commit",
        RUNTIME_COMMIT,
    ]
    code = cli_main(
        [
            *common,
            "--dirg-metadata-path",
            str(absent / "metadata.csv"),
            "--dirg-raw-dir",
            str(absent / dirg_manifest.DATASET_NAME),
            "--dirg-reader-source-path",
            str(absent / "reader.py"),
            "--dirg-preprocessing-source-path",
            str(absent / "preprocessing.py"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["state"] == "dry_run_blocked"
    assert payload["stage"] == "dirg_fit_select"
    assert payload["backend_invoked"] is False
    assert payload["evidence_state"] == "not_evidence"
    assert not absent.exists()

    partial_code = cli_main(
        [
            *common,
            "--dirg-metadata-path",
            str(absent / "metadata.csv"),
        ]
    )
    partial_payload = json.loads(capsys.readouterr().out)
    assert partial_code == 2
    assert partial_payload["state"] == "cli_error"
    assert "DIRG execution requires --dirg-metadata-path" in partial_payload[
        "error_message"
    ]
    assert partial_payload["evidence_state"] == "not_evidence"

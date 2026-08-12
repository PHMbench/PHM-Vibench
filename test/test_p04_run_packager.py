from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.p04.package_decisive_run import (
    REQUIRED_ARTIFACTS,
    RunPackagingError,
    build_parser,
    package_decisive_run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Any]:
    inputs = tmp_path / "inputs"
    evaluator = tmp_path / "evaluator"
    inputs.mkdir()
    evaluator.mkdir()
    config = inputs / "resolved.yaml"
    config.write_text("model:\n  name: M_04_RoleConstrainedMoE\n", encoding="utf-8")
    checkpoint = inputs / "model.ckpt"
    checkpoint.write_bytes(b"frozen-checkpoint-bytes")
    metadata_sha256 = "4" * 64
    generator_sha256 = "5" * 64
    identification_ids = [100, 101, 102, 103]
    intervention_ids = [200, 201, 202, 203]
    manifest = inputs / "partition_manifest.json"
    _write_json(
        manifest,
        {
            "schema_id": "p04.synthetic-partitions.v1",
            "schema_version": 1,
            "seed": 240401,
            "runtime_random_resplit_forbidden": True,
            "offline_partition": "identification",
            "partition_map": {
                "train": "train",
                "val": "optimization_validation",
                "test": "intervention",
            },
            "metadata_file_sha256": metadata_sha256,
            "partitions": {
                "identification": {
                    "ids": identification_ids,
                    "sample_count": len(identification_ids),
                },
                "intervention": {
                    "ids": intervention_ids,
                    "sample_count": len(intervention_ids),
                },
            },
        },
    )
    config_hash = _sha256(config)
    checkpoint_hash = _sha256(checkpoint)
    manifest_hash = _sha256(manifest)
    sample_ids = np.asarray(identification_ids + intervention_ids, dtype=np.int64)
    partition = np.asarray(
        ["identification"] * 4 + ["intervention"] * 4, dtype=np.str_
    )
    labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    mechanisms = np.asarray(
        ["low_frequency", "harmonic", "impulsive_envelope", "aperiodic_residual"]
        * 2,
        dtype=np.str_,
    )
    logits = np.asarray(
        [
            [4.0, 1.0, 0.0, -1.0],
            [0.0, 5.0, 1.0, -1.0],
            [-1.0, 0.0, 4.0, 1.0],
            [1.0, 0.0, -1.0, 4.0],
        ]
        * 2,
        dtype=np.float32,
    )
    routing = np.asarray(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.2],
            [0.2, 0.1, 0.4, 0.3],
            [0.3, 0.2, 0.1, 0.4],
        ]
        * 2,
        dtype=np.float32,
    )
    trace = inputs / "collector_trace.npz"
    assignment_seal_sha256 = "6" * 64
    np.savez(
        trace,
        schema=np.asarray("p04.mechanism-evaluator-input.v1"),
        schema_id=np.asarray("p04.mechanism-evaluator-input.v1"),
        arm=np.asarray("FULL"),
        seed=np.asarray(42, dtype=np.int64),
        sample_id=sample_ids,
        partition=partition,
        label=labels,
        mechanism=mechanisms,
        diagnosis=labels.copy(),
        nuisance_cell=np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        draw=np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        logits=logits,
        routing_weights=routing,
        config_sha256=np.asarray(config_hash),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        manifest_sha256=np.asarray(manifest_hash),
        partition_manifest_sha256=np.asarray(manifest_hash),
        generator_manifest_sha256=np.asarray(generator_sha256),
        metadata_sha256=np.asarray(metadata_sha256),
        assignment_seal_sha256=np.asarray(assignment_seal_sha256),
    )
    trace_hash = _sha256(trace)
    evaluator_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "p04"
        / "evaluate_role_identification.py"
    )
    evaluator_source_sha256 = _sha256(evaluator_source)
    traces = []
    for index, (arm, seed) in enumerate(
        (arm, seed)
        for arm in ("FULL", "HOMO", "RAND")
        for seed in (42, 123, 456, 789, 1024)
    ):
        record = {
            "arm": arm,
            "seed": seed,
            "trace_sha256": f"{index + 1:064x}",
            "assignment_seal_sha256": f"{index + 101:064x}",
        }
        if (arm, seed) == ("FULL", 42):
            record["trace_sha256"] = trace_hash
            record["assignment_seal_sha256"] = assignment_seal_sha256
        traces.append(record)
    correction = inputs / "evaluation_correction.yaml"
    _write_json(
        correction,
        {
            "schema_id": "p04.evaluation-correction.v1",
            "schema_version": "1.0.0",
            "evaluation_correction_id": "P04-G050-EVAL-C2",
            "status": "registered",
            "supersedes_evaluator_sha256": (
                "9848399cae54c1941e52cbb40ca31af5"
                "08cd770199fb11021399fbec826d9950"
            ),
            "evaluator_source_sha256": evaluator_source_sha256,
            "verification_dtype": "float32",
            "fixed_mass_rtol": 1.0e-5,
            "fixed_mass_atol": 1.0e-6,
            "estimand_changed": False,
            "thresholds_changed": False,
            "discovery_boundary": "no_aggregate_or_claim_decision",
            "traces": traces,
        },
    )
    correction_hash = _sha256(correction)
    provenance = {
        "seed": 42,
        "arm": "FULL",
        "unified_trace_sha256": trace_hash,
        "partition_manifest_sha256": manifest_hash,
        "generator_manifest_sha256": generator_sha256,
        "assignment_seal_sha256": assignment_seal_sha256,
        "evaluation_correction_id": "P04-G050-EVAL-C2",
        "evaluator_source_sha256": evaluator_source_sha256,
        "supersedes_evaluator_sha256": (
            "9848399cae54c1941e52cbb40ca31af5"
            "08cd770199fb11021399fbec826d9950"
        ),
        "correction_manifest_sha256": correction_hash,
        "verification_dtype": "float32",
        "fixed_mass_rtol": 1.0e-5,
        "fixed_mass_atol": 1.0e-6,
    }
    _write_json(
        evaluator / "behavior_signatures.json",
        {
            "schema_id": "p04.behavior-signatures.v1",
            "schema_version": "1.0.0",
            "provenance": provenance,
        },
    )
    _write_json(
        evaluator / "role_assignment.json",
        {
            "schema_id": "p04.role-assignment.v1",
            "schema_version": "1.0.0",
            "provenance": provenance,
        },
    )
    metrics = {
        "schema_id": "p04.mechanism-metrics.v1",
        "schema_version": "1.0.0",
        "role_recovery_count": 4,
        "role_recovery_accuracy": 1.0,
        "per_role_correctness": {
            "low_frequency": True,
            "harmonic": True,
            "impulsive_envelope": True,
            "aperiodic_residual": True,
        },
        "primary_deletion_interaction_I": 0.5,
        "intact_task_competence": {
            "balanced_accuracy": 1.0,
            "label_recalls": [1.0, 1.0, 1.0, 1.0],
            "every_label_recall_positive": True,
        },
        "intervention": {
            "primary_deletion": {"interaction": 0.5},
            "fixed_mass_output_substitution": {"estimand_J": 0.25},
        },
        "provenance": provenance,
    }
    _write_json(evaluator / "metrics.json", metrics)
    np.savez(
        evaluator / "deletion_losses.npz",
        schema_id=np.asarray("p04.deletion-losses.v1"),
        seed=np.asarray(42, dtype=np.int64),
        arm=np.asarray("FULL"),
        sample_ids=np.asarray(intervention_ids, dtype=np.str_),
        baseline_loss=np.asarray([0.1, 0.2, 0.3, 0.4]),
        unified_trace_sha256=np.asarray(trace_hash),
        assignment_seal_sha256=np.asarray(assignment_seal_sha256),
        evaluation_correction_id=np.asarray("P04-G050-EVAL-C2"),
        evaluator_source_sha256=np.asarray(evaluator_source_sha256),
        supersedes_evaluator_sha256=np.asarray(
            "9848399cae54c1941e52cbb40ca31af5"
            "08cd770199fb11021399fbec826d9950"
        ),
        correction_manifest_sha256=np.asarray(correction_hash),
        verification_dtype=np.asarray("float32"),
        fixed_mass_rtol=np.asarray(1.0e-5),
        fixed_mass_atol=np.asarray(1.0e-6),
    )
    run_meta = inputs / "run_meta.json"
    meta = {
        "run_id": "P04-E-MINDEC-FULL-42-attempt1",
        "experiment_id": "E-MINDEC",
        "dataset": "P04_SYNTHETIC",
        "arm": "FULL",
        "status": "completed",
        "conda_environment": "LQ_signal",
        "command": "conda run -n LQ_signal python main.py --config decisive_full.yaml",
        "working_directory": "/workspace/P04/src/vibench",
        "physical_gpu_indices": [0],
        "cuda_visible_devices": "0",
        "multi_gpu": False,
        "gpu_model": "fixture-gpu",
        "gpu_count": 1,
        "precision": 32,
        "started_at": "2026-08-01T10:00:00+08:00",
        "ended_at": "2026-08-01T10:01:00+08:00",
        "runtime_seconds": 60.0,
        "exit_code": 0,
        "oom_or_failure_reason": None,
        "fallback_used": False,
        "git_commit": "a" * 40,
        "git_diff_sha256": "b" * 64,
        "resolved_config_sha256": config_hash,
        "source_metadata_sha256": generator_sha256,
        "derived_metadata_sha256": metadata_sha256,
        "split_manifest_sha256": manifest_hash,
        "code_artifact_sha256": "c" * 64,
        "checkpoint_sha256": checkpoint_hash,
        "training_seed": 42,
        "split_seed": 240401,
    }
    _write_json(run_meta, meta)
    return {
        "inputs": inputs,
        "evaluator": evaluator,
        "config": config,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "trace": trace,
        "correction": correction,
        "run_meta": run_meta,
        "meta": meta,
        "intervention_ids": intervention_ids,
        "intervention_logits": logits[4:],
        "intervention_routing": routing[4:],
    }


def _package(fixture: dict[str, Any], bundle: Path) -> dict[str, Any]:
    return package_decisive_run(
        bundle_dir=bundle,
        resolved_config=fixture["config"],
        split_manifest=fixture["manifest"],
        checkpoint=fixture["checkpoint"],
        collector_trace=fixture["trace"],
        evaluator_dir=fixture["evaluator"],
        run_meta=fixture["run_meta"],
        correction_manifest=fixture["correction"],
    )


def _parse_ledger(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        records[relative] = digest
    return records


def test_happy_path_exact_hash_inventory_and_deterministic_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = tmp_path / "bundles" / "seed_42_a"
    second = tmp_path / "bundles" / "seed_42_b"
    result = _package(fixture, first)
    _package(fixture, second)

    assert result["prediction_rows"] == 4
    expected_inventory = set(REQUIRED_ARTIFACTS) | {"artifact_hashes.sha256"}
    assert {path.name for path in first.iterdir()} == expected_inventory
    ledger = _parse_ledger(first / "artifact_hashes.sha256")
    assert set(ledger) == set(REQUIRED_ARTIFACTS)
    for relative, digest in ledger.items():
        assert digest == _sha256(first / relative)
    assert (first / "resolved_config.yaml").read_bytes() == fixture["config"].read_bytes()
    assert (first / "split_manifest.json").read_bytes() == fixture["manifest"].read_bytes()
    assert (first / "checkpoint.ckpt").read_bytes() == fixture["checkpoint"].read_bytes()
    assert (first / "routing_trace.npz").read_bytes() == fixture["trace"].read_bytes()
    assert (first / "evaluation_correction.yaml").read_bytes() == fixture[
        "correction"
    ].read_bytes()
    for name in REQUIRED_ARTIFACTS:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "artifact_hashes.sha256").read_bytes() == (
        second / "artifact_hashes.sha256"
    ).read_bytes()


def test_predictions_parquet_contains_exact_intervention_rows(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    _package(fixture, bundle)
    table = pq.read_table(bundle / "predictions.parquet")
    assert table.column_names == [
        "sample_id",
        "partition",
        "label",
        "mechanism",
        "diagnosis",
        "nuisance_cell",
        "draw",
        "logit_0",
        "logit_1",
        "logit_2",
        "logit_3",
        "predicted_label",
        "route_0",
        "route_1",
        "route_2",
        "route_3",
    ]
    assert table.num_rows == 4
    assert table["sample_id"].to_pylist() == fixture["intervention_ids"]
    assert table["partition"].to_pylist() == ["intervention"] * 4
    np.testing.assert_allclose(
        np.column_stack([table[f"logit_{index}"].to_numpy() for index in range(4)]),
        fixture["intervention_logits"],
    )
    np.testing.assert_allclose(
        np.column_stack([table[f"route_{index}"].to_numpy() for index in range(4)]),
        fixture["intervention_routing"],
    )
    assert table["predicted_label"].to_pylist() == [0, 1, 2, 3]


def test_refuses_overwrite_without_touching_existing_bundle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    marker = bundle / "owned.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _package(fixture, bundle)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_tampered_input_hash_fails_and_cleans_only_staging(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["config"].write_text("model:\n  name: tampered\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    with pytest.raises(RunPackagingError, match="trace config_sha256"):
        _package(fixture, bundle)
    assert not bundle.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))
    assert fixture["checkpoint"].read_bytes() == b"frozen-checkpoint-bytes"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physical_gpu_indices", [2], "singleton"),
        ("multi_gpu", True, "multi_gpu"),
        ("cuda_visible_devices", "1", "cuda_visible_devices"),
    ],
)
def test_rejects_forbidden_or_inconsistent_device_metadata(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    fixture = _fixture(tmp_path)
    meta = dict(fixture["meta"])
    meta[field] = value
    fixture["run_meta"].write_text(
        yaml.safe_dump(meta, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(RunPackagingError, match=message):
        _package(fixture, tmp_path / "bundle")
    assert not list(tmp_path.glob(".bundle.staging-*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_field", "fields must be exact"),
        ("wrong_trace", "trace hash disagrees"),
        ("wrong_seal", "assignment seal disagrees"),
        ("wrong_evaluator", "evaluator_source_sha256 must equal"),
        ("numeric_false", "estimand_changed must equal False"),
    ],
)
def test_rejects_nonexact_or_unbound_correction_manifest(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["correction"].read_text(encoding="utf-8"))
    if mutation == "extra_field":
        manifest["notes"] = "not permitted"
    elif mutation == "wrong_trace":
        manifest["traces"][0]["trace_sha256"] = "a" * 64
    elif mutation == "wrong_seal":
        manifest["traces"][0]["assignment_seal_sha256"] = "a" * 64
    elif mutation == "wrong_evaluator":
        manifest["evaluator_source_sha256"] = "a" * 64
    else:
        manifest["estimand_changed"] = 0
    _write_json(fixture["correction"], manifest)

    with pytest.raises(RunPackagingError, match=message):
        _package(fixture, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        (
            "behavior_signatures.json",
            "missing_correction_id",
            "evaluation_correction_id",
        ),
        (
            "metrics.json",
            "mixed_dtype",
            "verification_dtype disagrees",
        ),
    ],
)
def test_rejects_old_or_mixed_json_evaluator_provenance(
    tmp_path: Path, artifact: str, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["evaluator"] / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_correction_id":
        del payload["provenance"]["evaluation_correction_id"]
    else:
        payload["provenance"]["verification_dtype"] = "float64"
    _write_json(path, payload)

    with pytest.raises(RunPackagingError, match=message):
        _package(fixture, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_rejects_missing_deletion_correction_provenance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    deletion_path = fixture["evaluator"] / "deletion_losses.npz"
    with np.load(deletion_path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
            if name != "correction_manifest_sha256"
        }
    np.savez(deletion_path, **arrays)

    with pytest.raises(RunPackagingError, match="correction_manifest_sha256"):
        _package(fixture, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_rejects_non_json_correction_manifest_text(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["correction"].read_text(encoding="utf-8"))
    fixture["correction"].write_text(
        yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(RunPackagingError, match="cannot parse evaluation correction"):
        _package(fixture, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_cli_requires_correction_manifest_argument() -> None:
    arguments = [
        "--bundle-dir",
        "bundle",
        "--resolved-config",
        "config",
        "--split-manifest",
        "split",
        "--checkpoint",
        "checkpoint",
        "--collector-trace",
        "trace",
        "--evaluator-dir",
        "evaluator",
        "--run-meta",
        "meta",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
    parsed = build_parser().parse_args(
        [*arguments, "--correction-manifest", "evaluation_correction.yaml"]
    )
    assert parsed.correction_manifest == Path("evaluation_correction.yaml")

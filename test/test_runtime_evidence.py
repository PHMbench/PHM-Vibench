from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import cli
from phmfactory.config import ResolvedConfig
from phmfactory.runtime import (
    AttestationError,
    CompiledRunSpec,
    ExecutionEnvelope,
    RunAttestation,
)
from phmfactory.runtime.evidence import register_pipeline_result_evidence
from src.runtime.classification import _register_iteration_evidence


def _spec(tmp_path: Path, pipeline: str = "Pipeline_01_Fault_Diagnosis") -> CompiledRunSpec:
    return CompiledRunSpec.compile(
        ResolvedConfig(
            requested="evidence-test",
            path=tmp_path / "config.yaml",
            data={
                "pipeline": pipeline,
                "environment": {"output_dir": str(tmp_path / "outputs")},
            },
            pipeline=pipeline,
            overrides={},
        )
    )


def _attestation(tmp_path: Path, pipeline: str = "Pipeline_01_Fault_Diagnosis"):
    spec = _spec(tmp_path, pipeline)
    envelope = ExecutionEnvelope(spec=spec, pipeline_module=f"src.{pipeline}")
    return spec, envelope, RunAttestation.prepare(spec, f"src.{pipeline}", envelope)


def test_artifact_registry_is_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    _, _, attestation = _attestation(tmp_path)
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nacc,1\n", encoding="utf-8")

    first = attestation.register_artifact(
        role="metrics",
        path=artifact,
        metadata={"iteration": 0},
    )
    second = attestation.register_artifact(
        role="metrics",
        path=artifact,
        metadata={"iteration": 0},
    )
    assert first == second
    assert len(attestation.artifacts) == 1

    with pytest.raises(AttestationError, match="conflicting artifact"):
        attestation.register_artifact(
            role="metrics",
            path=artifact,
            metadata={"iteration": 1},
        )


def test_artifact_registry_rejects_missing_files_and_bad_hashes(tmp_path: Path) -> None:
    _, _, attestation = _attestation(tmp_path)
    with pytest.raises(AttestationError, match="does not exist"):
        attestation.register_artifact(role="missing", path=tmp_path / "missing.csv")

    artifact = tmp_path / "metrics.csv"
    artifact.write_text("ok", encoding="utf-8")
    with pytest.raises(AttestationError, match="64 lowercase hex"):
        attestation.register_artifact(role="metrics", path=artifact, sha256="ABC")


def test_evidence_sections_reject_conflicting_overwrite(tmp_path: Path) -> None:
    _, _, attestation = _attestation(tmp_path)
    attestation.set_evidence("protocol", {"name": "dg"})
    attestation.set_evidence("protocol", {"name": "dg"})
    with pytest.raises(AttestationError, match="conflicting evidence"):
        attestation.set_evidence("protocol", {"name": "cddg"})

    attestation.append_evidence("iterations", {"index": 0})
    attestation.append_evidence("iterations", {"index": 1})
    assert attestation.evidence["iterations"] == [{"index": 0}, {"index": 1}]


def test_classification_metrics_are_registered_without_new_output_files(
    tmp_path: Path,
) -> None:
    _, _, attestation = _attestation(tmp_path)
    metrics = tmp_path / "run" / "test_result_0.csv"
    metrics.parent.mkdir()
    metrics.write_text("acc\n1\n", encoding="utf-8")
    args = SimpleNamespace(run_attestation=attestation)

    _register_iteration_evidence(
        args,
        iteration=0,
        seed=7,
        path=metrics.parent,
        metrics_path=metrics,
    )

    assert attestation.artifacts[0]["role"] == "classification_test_metrics"
    assert attestation.evidence["classification_iterations"][0]["seed"] == 7


def test_pipeline06_result_indexes_existing_stage_artifacts(tmp_path: Path) -> None:
    spec, _, attestation = _attestation(tmp_path, "Pipeline_06_Generative_Modeling")
    output_root = Path(spec.config["environment"]["output_dir"])
    output_root.mkdir(parents=True)
    ledger = output_root / "stage_ledger.json"
    ledger.write_text('{"stages": {}}\n', encoding="utf-8")
    run_dir = tmp_path / "stage"
    run_dir.mkdir()
    metrics = run_dir / "generative_eval_metrics.csv"
    metrics.write_text("metric,value\n", encoding="utf-8")
    evaluation = run_dir / "evaluation_evidence_manifest.json"
    evaluation.write_text("{}\n", encoding="utf-8")

    result = [
        {
            "stage": "eval",
            "status": "completed",
            "run_dir": str(run_dir),
            "metrics": {"path": str(metrics)},
            "evaluation_manifest": {"path": str(evaluation)},
            "metric_summary": {},
        }
    ]
    register_pipeline_result_evidence(attestation, spec, result)

    roles = {artifact["role"] for artifact in attestation.artifacts}
    assert roles == {
        "generative_eval_metrics",
        "generative_eval_evaluation_manifest",
        "generative_stage_ledger",
    }
    assert attestation.evidence["generative_stages"][0]["stage"] == "eval"


def test_pipeline06_missing_ledger_invalidates_evidence(tmp_path: Path) -> None:
    spec, _, attestation = _attestation(tmp_path, "Pipeline_06_Generative_Modeling")
    with pytest.raises(AttestationError, match="stage ledger is missing"):
        register_pipeline_result_evidence(
            attestation,
            spec,
            [{"stage": "train", "status": "completed", "run_dir": str(tmp_path)}],
        )


def test_cli_records_evidence_finalize_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = "Pipeline_06_Generative_Modeling"
    resolved = ResolvedConfig(
        requested="generative",
        path=tmp_path / "generative.yaml",
        data={
            "pipeline": pipeline,
            "environment": {"output_dir": str(tmp_path / "outputs")},
        },
        pipeline=pipeline,
        overrides={},
    )
    monkeypatch.setattr(cli, "resolve_config", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            pipeline=lambda args: [
                {"stage": "train", "status": "completed", "run_dir": str(tmp_path)}
            ]
        ),
    )
    args = argparse.Namespace(
        config="generative",
        config_path=None,
        notes="",
        override=None,
        allow_experimental=False,
    )

    with pytest.raises(AttestationError, match="stage ledger is missing"):
        cli.run(args)

    payload = json.loads(Path(args.run_manifest_path).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure"]["stage"] == "evidence_finalize"

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.explain_factory.run_artifacts as run_artifacts_module
from src.explain_factory.run_artifacts import (
    rewrite_manifest_after_test_result,
    write_explain_eligibility,
    write_run_artifact_sidecars,
)
from src.trainer_factory.extensions import write_run_manifest


def _write_required_run_files(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text("pipeline: Pipeline_01_default\n", encoding="utf-8")
    (run_dir / "test_result_0.csv").write_text("acc\n1.0\n", encoding="utf-8")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "data_metadata_snapshot.json").write_text('{"source":"test"}\n', encoding="utf-8")


def test_run_manifest_contract_contains_parent_consumable_fields(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_required_run_files(run_dir)

    manifest_path = write_run_manifest(
        run_dir,
        stage="test",
        run_id="demo",
        seed=7,
        paper_id="paper",
        preset_version="v1",
        required=True,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in [
        "run_id",
        "stage",
        "config_snapshot",
        "metrics_path",
        "run_dir",
        "timestamp",
        "seed",
        "git_sha",
        "data_metadata_snapshot",
    ]:
        assert key in payload

    assert payload["run_id"] == "demo"
    assert payload["stage"] == "test"
    assert payload["seed"] == 7
    assert payload["metrics_path"].endswith("test_result_0.csv")


def test_run_manifest_required_mode_rejects_missing_metrics(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_required_run_files(run_dir)
    (run_dir / "test_result_0.csv").unlink()

    with pytest.raises(RuntimeError, match="metrics_path"):
        write_run_manifest(run_dir, stage="test", run_id="demo", seed=7, required=True)


def test_run_manifest_accepts_legacy_test_result_csv(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_required_run_files(run_dir)
    (run_dir / "test_result_0.csv").unlink()
    (run_dir / "test_result.csv").write_text("acc\n1.0\n", encoding="utf-8")

    manifest_path = write_run_manifest(run_dir, stage="test", run_id="demo", seed=7, required=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["metrics_path"].endswith("test_result.csv")


def test_run_manifest_allows_optional_explain_outputs_to_be_absent(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_required_run_files(run_dir)

    manifest_path = write_run_manifest(
        run_dir,
        stage="test",
        run_id="demo",
        seed=7,
        required=True,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["explain_dir"] == ""
    assert payload["eligibility"] == ""


def test_run_artifact_helpers_write_sidecars_and_manifest(tmp_path) -> None:
    class FakeDataFactory:
        def get_dataloader(self, split):
            assert split == "test"
            batch = {
                "x": SimpleNamespace(shape=(2, 3)),
                "y": SimpleNamespace(shape=(2,)),
                "meta": {"sampling_rate": 12000},
            }
            return iter([batch])

    run_dir = tmp_path / "run"
    args_trainer = SimpleNamespace(
        logger_name="demo",
        paper_id="paper",
        preset_version="v1",
        extensions=SimpleNamespace(
            explain=SimpleNamespace(enable=False),
            report=SimpleNamespace(enable=True, manifest=True),
        ),
    )

    write_run_artifact_sidecars(
        run_dir=run_dir,
        cfg={"pipeline": "Pipeline_01_default"},
        args_trainer=args_trainer,
        data_factory=FakeDataFactory(),
    )
    (run_dir / "test_result_0.csv").write_text("acc\n1.0\n", encoding="utf-8")

    manifest_path = rewrite_manifest_after_test_result(
        run_dir=run_dir,
        args_trainer=args_trainer,
        trainer=SimpleNamespace(callback_metrics={}),
        seed=7,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["config_snapshot"].endswith("config_snapshot.yaml")
    assert payload["data_metadata_snapshot"].endswith("data_metadata_snapshot.json")
    assert payload["metrics_path"].endswith("test_result_0.csv")


def test_run_artifact_helpers_write_enabled_explain_eligibility(tmp_path) -> None:
    class FakeDataFactory:
        def get_dataloader(self, split):
            assert split == "test"
            batch = {
                "x": SimpleNamespace(shape=(2, 3)),
                "y": SimpleNamespace(shape=(2,)),
                "meta": {"sampling_rate": 12000},
            }
            return iter([batch])

    run_dir = tmp_path / "run"
    args_trainer = SimpleNamespace(
        logger_name="demo",
        extensions=SimpleNamespace(
            explain=SimpleNamespace(enable=True, explainer="timefreq"),
            report=SimpleNamespace(enable=True, manifest=True),
        ),
    )

    write_run_artifact_sidecars(
        run_dir=run_dir,
        cfg={"pipeline": "Pipeline_01_default"},
        args_trainer=args_trainer,
        data_factory=FakeDataFactory(),
    )
    (run_dir / "test_result_0.csv").write_text("acc\n1.0\n", encoding="utf-8")
    manifest_path = rewrite_manifest_after_test_result(
        run_dir=run_dir,
        args_trainer=args_trainer,
        trainer=SimpleNamespace(callback_metrics={}),
        seed=7,
    )

    eligibility_path = run_dir / "artifacts" / "explain" / "eligibility.json"
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert eligibility["ok"] is True
    assert eligibility["explainer_id"] == "timefreq"
    assert manifest["eligibility"].endswith("artifacts/explain/eligibility.json")
    assert manifest["explain_dir"].endswith("artifacts/explain")


def test_enabled_explain_eligibility_write_failures_are_loud(tmp_path, monkeypatch) -> None:
    def fail_write(*args, **kwargs):
        raise OSError("cannot write eligibility")

    monkeypatch.setattr(run_artifacts_module, "write_eligibility", fail_write)

    with pytest.raises(OSError, match="cannot write eligibility"):
        write_explain_eligibility(
            run_dir=tmp_path / "run",
            explainer_id="timefreq",
            meta={"sampling_rate": 12000},
            meta_source="batch",
            degraded=False,
            required_meta_keys=["sampling_rate"],
        )

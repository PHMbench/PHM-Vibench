from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.trainer_factory.extensions.manifest import ManifestWriterCallback


def test_manifest_json_required_keys_and_optional_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run_0"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_snapshot.yaml").write_text("dummy: true\n", encoding="utf-8")
    (run_dir / "test_result_0.csv").write_text("metric,value\nacc,1.0\n", encoding="utf-8")

    trainer = SimpleNamespace(callback_metrics={"test_loss": torch.tensor(0.1)})
    cb = ManifestWriterCallback(
        run_dir=str(run_dir),
        paper_id="Paper_fuzzy_XFD",
        preset_version="vibench-min-v1",
        run_id="run_0",
        enabled=True,
        is_main_process=True,
    )

    cb.on_test_end(trainer, pl_module=None)
    manifest_path = artifacts_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_keys = [
        "paper_id",
        "preset_version",
        "run_id",
        "run_dir",
        "stage",
        "timestamp",
        "config_snapshot",
        "metrics_path",
        "metrics_csv_logger",
        "figures_dir",
        "predictions_path",
        "data_metadata_snapshot",
        "eligibility",
        "explain_dir",
        "explain_summary",
        "distilled_dir",
    ]
    for k in required_keys:
        assert k in manifest

    assert manifest["run_dir"] == str(run_dir)
    assert manifest["run_id"] == "run_0"
    assert manifest["stage"] == "test"
    assert manifest["timestamp"]
    assert manifest["config_snapshot"].endswith("config_snapshot.yaml")
    assert manifest["metrics_path"].endswith("test_result_0.csv")

    # Optional fields are present but may be empty when artifacts are disabled / absent.
    optional_paths = [
        "metrics_csv_logger",
        "figures_dir",
        "predictions_path",
        "data_metadata_snapshot",
        "eligibility",
        "explain_dir",
        "explain_summary",
        "distilled_dir",
    ]
    for k in optional_paths:
        assert manifest[k] == ""


def test_manifest_json_schema_and_optional_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run_0"
    artifacts_dir = run_dir / "artifacts"
    explain_dir = artifacts_dir / "explain"
    distilled_dir = artifacts_dir / "distilled"
    logs_dir = run_dir / "logs" / "csv"
    explain_dir.mkdir(parents=True, exist_ok=True)
    distilled_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_snapshot.yaml").write_text("dummy: true\n", encoding="utf-8")
    (logs_dir / "metrics.csv").write_text("step,metric\n0,0.0\n", encoding="utf-8")
    (run_dir / "test_result_0.csv").write_text("metric,value\nacc,1.0\n", encoding="utf-8")

    np.savez(artifacts_dir / "predictions.npz", preds=np.zeros((2, 3)), labels=np.zeros((2,)))
    (explain_dir / "eligibility.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    trainer = SimpleNamespace(callback_metrics={"test_loss": torch.tensor(0.1)})
    cb = ManifestWriterCallback(
        run_dir=str(run_dir),
        paper_id="Paper_fuzzy_XFD",
        preset_version="vibench-min-v1",
        run_id="run_0",
        enabled=True,
        is_main_process=True,
    )

    cb.on_test_end(trainer, pl_module=None)
    manifest_path = artifacts_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_keys = [
        "paper_id",
        "preset_version",
        "run_id",
        "run_dir",
        "stage",
        "timestamp",
        "config_snapshot",
        "metrics_path",
        "metrics_csv_logger",
        "figures_dir",
        "predictions_path",
        "data_metadata_snapshot",
        "eligibility",
        "explain_dir",
        "explain_summary",
        "distilled_dir",
    ]
    for k in required_keys:
        assert k in manifest

    assert manifest["run_dir"] == str(run_dir)
    assert manifest["run_id"] == "run_0"
    assert manifest["stage"] == "test"
    assert manifest["timestamp"]
    assert manifest["config_snapshot"].endswith("config_snapshot.yaml")

    # metrics_path prefers test_result_*.csv over logs/**/metrics.csv
    assert manifest["metrics_path"].endswith("test_result_0.csv")

    # Optional fields become non-empty only when artifacts exist
    assert manifest["predictions_path"].endswith("predictions.npz")
    assert Path(manifest["predictions_path"]).exists()
    assert manifest["eligibility"].endswith("eligibility.json")
    assert Path(manifest["eligibility"]).exists()
    assert manifest["explain_dir"].endswith("artifacts/explain")
    assert manifest["distilled_dir"].endswith("artifacts/distilled")

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


CONFIG = "configs/demo/10_generative/dummy_generative_cfm.yaml"
REQUIRED_METRICS = {
    "time_domain_statistics_distance",
    "spectral_distance",
    "condition_consistency_distance",
    "nearest_neighbor_leakage_l2",
    "duplicate_rate",
    "downstream_classifier_utility",
    "fid_like_embedding_distance",
    "training_wall_clock_seconds",
}


def _run(*overrides: str) -> None:
    command = [sys.executable, "main.py", "--config", CONFIG]
    for override in overrides:
        command.extend(("--override", override))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"command failed: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _ledger(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_maintained_cfm_demo_completes_cpu_e_chain(tmp_path: Path) -> None:
    output = tmp_path / "cfm"
    ledger_path = output / "stage_ledger.json"
    common = (
        f"environment.output_dir={output}",
        "environment.seed=0",
        "trainer.device=cpu",
        "trainer.gpus=1",
        "trainer.num_epochs=1",
        "data.num_workers=0",
        f"task.generative.stage_ledger_path={ledger_path}",
    )

    _run(*common, "task.generative.mode=train")
    train = _ledger(ledger_path)["stages"]["train"]
    assert train["status"] == "completed"
    assert train["checkpoint"]["strict"] is True
    assert train["normalization"]["source_split"] == "train"

    _run(
        *common,
        "task.generative.mode=sample",
        f"task.generative.checkpoint_path={train['checkpoint']['path']}",
        f"task.generative.normalization_path={train['normalization']['path']}",
        f"task.generative.normalization_sha256={train['normalization']['sha256']}",
        f"task.generative.protocol_path={train['protocol']['path']}",
    )
    sample = _ledger(ledger_path)["stages"]["sample"]
    assert sample["status"] == "completed"
    assert sample["checkpoint"]["sha256"] == train["checkpoint"]["sha256"]
    assert sample["normalization"]["sha256"] == train["normalization"]["sha256"]

    sample_payload = torch.load(
        sample["samples"]["path"], map_location="cpu", weights_only=True
    )
    assert torch.isfinite(sample_payload["samples"]).all()
    synthetic_manifest = json.loads(
        Path(sample["synthetic_manifest"]["path"]).read_text(encoding="utf-8")
    )
    assert set(synthetic_manifest["conditions"]["direct_keys"]) == {
        "fault_label",
        "domain_id",
    }
    assert synthetic_manifest["validity"]["scientific_status"] == "exploratory"
    assert synthetic_manifest["validity"]["benchmark_valid"] is False
    assert synthetic_manifest["validity"]["paper_ready"] is False

    _run(
        *common,
        "task.generative.mode=eval",
        f"task.generative.generated_path={sample['samples']['path']}",
        (
            "task.generative.synthetic_manifest_path="
            f"{sample['synthetic_manifest']['path']}"
        ),
        "task.generative.eval_split=train",
    )
    evaluation = _ledger(ledger_path)["stages"]["eval"]
    assert evaluation["status"] == "completed"
    assert set(evaluation["metric_summary"]["required"]) == REQUIRED_METRICS
    assert evaluation["metric_summary"]["failed"] == 0
    assert evaluation["metric_summary"]["not_computable"] == 1

    with Path(evaluation["metrics"]["path"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metric_rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert set(metric_rows) == REQUIRED_METRICS
    for metric, row in metric_rows.items():
        assert row["status"] != "failed", metric
        if row["status"] == "ok":
            assert math.isfinite(float(row["value"])), metric
        else:
            assert row["status"] == "not_computable", metric
            assert row["reason"].strip(), metric

    manifest = json.loads(
        Path(evaluation["evaluation_manifest"]["path"]).read_text(encoding="utf-8")
    )
    assert manifest["metric_summary"]["missing"] == 0
    assert manifest["metric_summary"]["failed"] == 0
    assert manifest["promotion"]["runtime_smoke_eligible"] is True
    assert manifest["promotion"]["paper_smoke_metric_eligible"] is False
    assert manifest["promotion"]["sanity_ok"] is False
    assert manifest["promotion"]["paper_smoke_ready"] is False
    assert manifest["promotion"]["benchmark_valid"] is False

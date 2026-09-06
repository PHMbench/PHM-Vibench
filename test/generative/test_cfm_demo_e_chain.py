from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


CONFIG = "configs/demo/10_generative/dummy_generative_cfm.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY_RAW_FILES = ("dummy1.csv", "dummy2.csv")
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
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert completed.returncode == 0, (
        f"command failed: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _ledger(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_h5_cache_paths() -> set[Path]:
    return {
        path.resolve()
        for path in (REPO_ROOT / "data").rglob("*.h5")
        if path.is_file()
    }


def _copy_dummy_fixture(destination: Path) -> None:
    raw_destination = destination / "raw" / "Dummy_Data"
    raw_destination.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "data" / "metadata_dummy.csv", destination)
    for file_name in DUMMY_RAW_FILES:
        shutil.copy2(
            REPO_ROOT / "data" / "raw" / "Dummy_Data" / file_name,
            raw_destination / file_name,
        )


def test_candidate_cfm_demo_completes_cpu_e_chain(tmp_path: Path) -> None:
    repo_h5_before = _repo_h5_cache_paths()
    fixture_data = tmp_path / "data"
    _copy_dummy_fixture(fixture_data)
    output = tmp_path / "cfm"
    ledger_path = output / "stage_ledger.json"
    common = (
        f"environment.output_dir={output}",
        "environment.seed=0",
        "trainer.device=cpu",
        "trainer.gpus=1",
        "trainer.num_epochs=1",
        "data.num_workers=0",
        f"data.data_dir={fixture_data}",
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
    assert manifest["promotion"]["benchmark_valid"] is False
    assert (fixture_data / "Dummy_Data.h5").is_file()
    assert (fixture_data / "cache.h5").is_file()
    assert _repo_h5_cache_paths() == repo_h5_before

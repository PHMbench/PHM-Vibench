from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.paperpack_generative import build_paperpack
from src.Pipeline_06_generative import _update_stage_ledger, _write_eval_evidence_manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_stage_ledger_updates_preserve_existing_stage_entries(tmp_path: Path) -> None:
    ledger_path = tmp_path / "stage_ledger.json"

    _update_stage_ledger(
        ledger_path,
        mode="sample",
        values={
            "run_dir": tmp_path / "sample",
            "samples_path": tmp_path / "sample" / "synthetic" / "samples.pt",
        },
    )
    _update_stage_ledger(
        ledger_path,
        mode="eval",
        values={"metrics_path": tmp_path / "eval" / "generative_eval_metrics.csv"},
    )

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["schema_version"] == "0.3.0"
    assert ledger["stages"]["sample"]["samples_path"].endswith("samples.pt")
    assert ledger["stages"]["eval"]["metrics_path"].endswith("generative_eval_metrics.csv")


def test_eval_evidence_manifest_records_status_summary_and_sample_manifest(
    tmp_path: Path,
) -> None:
    generated_path = tmp_path / "sample" / "synthetic" / "samples.pt"
    generated_path.parent.mkdir(parents=True)
    generated_path.write_text("placeholder", encoding="utf-8")
    sample_manifest = generated_path.with_name("synthetic_data_manifest.json")
    sample_manifest.write_text("{}", encoding="utf-8")
    metrics_path = tmp_path / "eval" / "generative_eval_metrics.csv"
    evidence_path = tmp_path / "eval" / "eval_evidence_manifest.json"

    manifest = _write_eval_evidence_manifest(
        evidence_path,
        generated_path=generated_path,
        metrics_path=metrics_path,
        reference_split="train",
        allow_test_reference_eval=False,
        metrics={
            "temporal_l1": 1.0,
            "temporal_l1_status": "ok",
            "tstr_nearest_centroid_accuracy": float("nan"),
            "tstr_nearest_centroid_accuracy_status": "not_computable",
        },
    )

    assert evidence_path.exists()
    assert manifest["synthetic_manifest_path"] == str(sample_manifest)
    assert manifest["metric_status_summary"] == {"ok": 1, "not_computable": 1}
    assert manifest["promotion"]["eligible"] is False
    assert "metric_status_ok" in manifest["promotion"]["missing"]


def test_paperpack_stage_ledger_includes_sibling_sample_manifest(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample" / "synthetic"
    eval_dir = tmp_path / "eval"
    sample_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    manifest_path = sample_dir / "synthetic_data_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "synthetic_dataset_id": "synthetic-seed-0",
                "sampling": {"seed": 0},
                "normalization": {"params_recorded": True},
                "validity": {
                    "status": "exploratory",
                    "benchmark_valid": False,
                    "missing_evidence": ["metric_status_reason_recorded"],
                    "evidence": {"metric_status_reason_recorded": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "generative_eval_metrics.csv").write_text(
        "temporal_l1,temporal_l1_status\n1.0,ok\n",
        encoding="utf-8",
    )
    ledger_path = tmp_path / "stage_ledger.json"
    _update_stage_ledger(
        ledger_path,
        mode="sample",
        values={"synthetic_manifest_path": manifest_path},
    )

    paperpack = build_paperpack(eval_dir, stage_ledger=ledger_path)

    rows = _read_csv(paperpack / "appendix" / "manifest_completeness.csv")
    assert rows[0]["manifest_path"] == str(manifest_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["stages"]["paperpack"]["paperpack_dir"] == str(paperpack)


def test_paperpack_missing_stage_ledger_records_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "eval"
    run_dir.mkdir()
    (run_dir / "generative_eval_metrics.csv").write_text(
        "temporal_l1,temporal_l1_status\n1.0,ok\n",
        encoding="utf-8",
    )

    paperpack = build_paperpack(run_dir, stage_ledger=tmp_path / "missing.json")

    rows = _read_csv(paperpack / "appendix" / "run_index.csv")
    warning = next(row for row in rows if row["source_type"] == "warning")
    assert "stage_ledger missing" in warning["utility_reference_split"]

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generative_benchmark_effect import (
    build_effect_report,
    build_run_plan,
    load_matrix,
    validate_matrix_inputs,
    write_run_plan,
)


def _write_matrix(path: Path, data_dir: Path) -> None:
    path.write_text(
        f"""
benchmark:
  id: "fixture_benchmark"
  dataset: "CWRU_domain_shift"
  output_dir: "{path.parent / 'out'}"
  baseline_method: "cfm_grid"
  python: "python"
  seeds: [0, 1]
  data_check:
    data_dir: "{data_dir}"
    metadata_file: "metadata.xlsx"
  overrides:
    data.data_dir: "{data_dir}"
    data.metadata_file: "metadata.xlsx"
    task.target_system_id: [1]
    task.source_domain_id: [0, 1, 2]
    task.target_domain_id: [3]
methods:
  - method: "cfm_grid"
    label: "CFM"
    train_config: "configs/demo/10_generative/dummy_generative_cfm.yaml"
    condition_sampling_policy: "grid"
  - method: "rectified_flow_grid"
    label: "Rectified Flow"
    train_config: "configs/demo/10_generative/dummy_generative_rectified_flow.yaml"
    condition_sampling_policy: "grid"
  - method: "ddpm_train_distribution"
    label: "DDPM"
    train_config: "configs/demo/10_generative/dummy_generative_ddpm.yaml"
    condition_sampling_policy: "train_distribution"
""".strip(),
        encoding="utf-8",
    )


def _write_metrics(path: Path, *, temporal: str, tstr: str, trts: str, reason: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "temporal_l1",
        "temporal_l1_status",
        "tstr_accuracy",
        "tstr_accuracy_status",
        "tstr_accuracy_reason",
        "trts_accuracy",
        "trts_accuracy_status",
        "trts_accuracy_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "temporal_l1": temporal,
                "temporal_l1_status": "ok",
                "tstr_accuracy": tstr,
                "tstr_accuracy_status": "not_computable" if reason else "ok",
                "tstr_accuracy_reason": reason,
                "trts_accuracy": trts,
                "trts_accuracy_status": "not_computable" if reason else "ok",
                "trts_accuracy_reason": reason,
            }
        )


def _write_manifest(path: Path, *, method: str, seed: int, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [] if valid else ["normalization_params"]
    path.write_text(
        json.dumps(
            {
                "synthetic_dataset_id": f"CWRU_domain_shift_{method}_seed{seed}",
                "sampling": {"seed": seed},
                "validity": {
                    "status": "benchmark-valid" if valid else "exploratory",
                    "benchmark_valid": valid,
                    "missing_evidence": missing,
                },
            }
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_dry_run_plan_uses_real_phm_matrix_and_keeps_stages(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)

    matrix = load_matrix(matrix_path)
    assert validate_matrix_inputs(matrix, allow_missing_data=False) == []
    rows = build_run_plan(matrix, ["trainer.num_epochs=1"])
    write_run_plan(tmp_path / "run_plan.csv", rows)

    assert len(rows) == 3 * 2 * 4
    stages = {row["stage"] for row in rows}
    assert stages == {"train", "sample", "eval", "paperpack"}
    assert any("task.generative.mode=eval" in row["command"] for row in rows)
    assert any("trainer.num_epochs=1" in row["command"] for row in rows)
    assert (tmp_path / "run_plan.csv").exists()


def test_effect_report_aggregates_quality_utility_delta_and_missing_reasons(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    matrix = load_matrix(matrix_path)
    root = tmp_path / "runs"

    values = {
        ("cfm_grid", 0): ("1.0", "0.5", "0.4", "", True),
        ("cfm_grid", 1): ("3.0", "0.7", "0.6", "", True),
        ("rectified_flow_grid", 0): ("1.0", "0.8", "0.7", "", True),
        ("rectified_flow_grid", 1): ("1.0", "0.9", "0.8", "", True),
        (
            "ddpm_train_distribution",
            0,
        ): ("nan", "nan", "nan", "labels unavailable for utility probe", False),
        (
            "ddpm_train_distribution",
            1,
        ): ("nan", "nan", "nan", "labels unavailable for utility probe", False),
    }
    for (method, seed), (temporal, tstr, trts, reason, valid) in values.items():
        run_dir = root / method / f"seed_{seed}" / "eval"
        _write_metrics(run_dir / "generative_eval_metrics.csv", temporal=temporal, tstr=tstr, trts=trts, reason=reason)
        _write_manifest(run_dir / "synthetic" / "synthetic_data_manifest.json", method=method, seed=seed, valid=valid)

    out = build_effect_report(matrix, [root], tmp_path / "effect")
    summary = _read_csv(out / "benchmark_effect_summary.csv")

    rf_temporal = next(row for row in summary if row["method"] == "rectified_flow_grid" and row["metric"] == "temporal_l1")
    assert float(rf_temporal["mean"]) == 1.0
    assert float(rf_temporal["baseline_mean"]) == 2.0
    assert float(rf_temporal["delta_vs_baseline"]) == -1.0
    assert rf_temporal["rank"] == "1"
    assert "generative_eval_metrics.csv" in rf_temporal["metric_source_paths"]
    assert "synthetic_data_manifest.json" in rf_temporal["manifest_paths"]

    rf_tstr = next(row for row in summary if row["method"] == "rectified_flow_grid" and row["metric"] == "tstr_accuracy")
    assert round(float(rf_tstr["mean"]), 4) == 0.85
    assert round(float(rf_tstr["delta_vs_baseline"]), 4) == 0.25
    assert rf_tstr["rank"] == "1"

    ddpm_tstr = next(row for row in summary if row["method"] == "ddpm_train_distribution" and row["metric"] == "tstr_accuracy")
    assert ddpm_tstr["n"] == "0"
    assert ddpm_tstr["missing_count"] == "2"
    assert "labels unavailable" in ddpm_tstr["missing_reasons"]
    assert ddpm_tstr["benchmark_status"] == "exploratory"

    missing = (out / "missing_metrics.md").read_text(encoding="utf-8")
    assert "labels unavailable for utility probe" in missing
    assert (out / "benchmark_effect_report.md").exists()
    assert (out / "benchmark_effect_manifest.json").exists()


def test_matrix_validation_fails_missing_real_phm_data_without_explicit_allow(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, tmp_path / "missing_data")
    matrix = load_matrix(matrix_path)

    errors = validate_matrix_inputs(matrix, allow_missing_data=False)
    assert errors
    assert "missing PHM metadata" in errors[0]
    assert validate_matrix_inputs(matrix, allow_missing_data=True) == []

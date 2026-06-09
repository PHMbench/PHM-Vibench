from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.paperpack_generative import build_paperpack


def _write_manifest(path: Path, *, seed: int, missing_evidence: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = missing_evidence or []
    evidence = {
        "protocol_hash": "protocol_hash" not in missing,
        "config_hash": "config_hash" not in missing,
        "dependency_lock_hash": "dependency_lock_hash" not in missing,
        "normalization_params": "normalization_params" not in missing,
        "leakage_checks": "leakage_checks" not in missing,
        "condition_sampling_policy": "condition_sampling_policy" not in missing,
        "condition_counts": "condition_counts" not in missing,
        "metric_status_reason_recorded": "metric_status_reason_recorded" not in missing,
    }
    path.write_text(
        json.dumps(
            {
                "synthetic_dataset_id": f"synthetic-seed-{seed}",
                "config": {"config_path": "config.yaml", "config_hash": "cfg"},
                "protocol": {"protocol_path": "protocol.json", "protocol_hash": "proto"},
                "environment": {"python": "3.11", "torch": "test", "dependency_lock_hash": "deps"},
                "sampling": {"seed": seed, "sampler_id": "euler_ode", "num_steps": 8, "shape": [2, 1, 8]},
                "normalization": {"params_recorded": evidence["normalization_params"]},
                "validity": {
                    "status": "benchmark-valid" if not missing else "exploratory",
                    "benchmark_valid": not missing,
                    "missing_evidence": missing,
                    "evidence": evidence,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "seed",
        "temporal_l1",
        "temporal_l1_status",
        "temporal_l1_reason",
        "spectral_fft_l1",
        "tstr_accuracy",
        "tstr_accuracy_status",
        "tstr_accuracy_reason",
        "parameter_count",
        "sampling_nfe",
        "leakage_nearest_neighbor_l2",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_paperpack_aggregates_multi_seed_metrics_and_missing_reasons(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir / "seed0" / "synthetic" / "synthetic_data_manifest.json", seed=0)
    _write_manifest(
        run_dir / "seed1" / "synthetic" / "synthetic_data_manifest.json",
        seed=1,
        missing_evidence=["metric_status_reason_recorded"],
    )
    _write_metrics(
        run_dir / "seed0" / "generative_eval_metrics.csv",
        [
            {
                "dataset": "CWRU",
                "method": "cfm_grid",
                "seed": "0",
                "temporal_l1": "1.0",
                "temporal_l1_status": "ok",
                "spectral_fft_l1": "2.0",
                "tstr_accuracy": "nan",
                "tstr_accuracy_status": "not_computable",
                "tstr_accuracy_reason": "real_labels and fake_labels are required",
                "parameter_count": "10",
                "sampling_nfe": "8",
                "leakage_nearest_neighbor_l2": "0.5",
            }
        ],
    )
    _write_metrics(
        run_dir / "seed1" / "generative_eval_metrics.csv",
        [
            {
                "dataset": "CWRU",
                "method": "cfm_grid",
                "seed": "1",
                "temporal_l1": "3.0",
                "temporal_l1_status": "ok",
                "spectral_fft_l1": "4.0",
                "tstr_accuracy": "0.75",
                "tstr_accuracy_status": "ok",
                "parameter_count": "12",
                "sampling_nfe": "8",
                "leakage_nearest_neighbor_l2": "0.7",
            }
        ],
    )

    paperpack = build_paperpack(run_dir)

    quality_rows = _read_csv(paperpack / "tables" / "table_quality_mean_std.csv")
    temporal_l1 = next(row for row in quality_rows if row["metric"] == "temporal_l1")
    assert float(temporal_l1["mean"]) == 2.0
    assert temporal_l1["n"] == "2"
    assert "seed0/generative_eval_metrics.csv" in temporal_l1["source_paths"]
    assert "seed1/generative_eval_metrics.csv" in temporal_l1["source_paths"]

    utility_rows = _read_csv(paperpack / "tables" / "table_utility_mean_std.csv")
    tstr = next(row for row in utility_rows if row["metric"] == "tstr_accuracy")
    assert tstr["n"] == "1"
    assert tstr["missing_count"] == "1"
    assert "real_labels and fake_labels" in tstr["missing_reasons"]

    missing_md = (paperpack / "appendix" / "missing_metrics.md").read_text(encoding="utf-8")
    assert "`tstr_accuracy`" in missing_md
    assert (paperpack / "appendix" / "run_index.csv").exists()
    assert (paperpack / "appendix" / "manifest_completeness.csv").exists()
    manifest_index = json.loads(
        (paperpack / "figure_sources" / "manifest_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest_index["synthetic_manifest_paths"]) == 2
    assert len(manifest_index["metric_source_paths"]) == 2
    assert all(
        path.endswith("generative_eval_metrics.csv")
        for path in manifest_index["metric_source_paths"]
    )
    assert (paperpack / "figure_sources" / "spectra_overlay.csv").exists()
    assert (paperpack / "figure_sources" / "temporal_overlay.csv").exists()
    assert (paperpack / "figure_sources" / "metric_barplot.csv").exists()
    spectra_rows = _read_csv(paperpack / "figure_sources" / "spectra_overlay.csv")
    assert all(row["source_path"] for row in spectra_rows)
    temporal_rows = _read_csv(paperpack / "figure_sources" / "temporal_overlay.csv")
    assert all(row["source_path"] for row in temporal_rows)
    barplot_rows = _read_csv(paperpack / "figure_sources" / "metric_barplot.csv")
    barplot_temporal = next(row for row in barplot_rows if row["metric"] == "temporal_l1")
    assert "seed0/generative_eval_metrics.csv" in barplot_temporal["source_paths"]
    assert "seed1/generative_eval_metrics.csv" in barplot_temporal["source_paths"]
    heatmap_rows = _read_csv(paperpack / "figure_sources" / "dataset_method_heatmap.csv")
    heatmap_temporal = next(row for row in heatmap_rows if row["metric"] == "temporal_l1")
    assert heatmap_temporal["dataset"] == "CWRU"
    assert heatmap_temporal["method"] == "cfm_grid"
    assert heatmap_temporal["n"] == "2"
    assert "seed0/generative_eval_metrics.csv" in heatmap_temporal["source_paths"]
    assert "seed1/generative_eval_metrics.csv" in heatmap_temporal["source_paths"]
    audit_rows = _read_csv(paperpack / "figure_sources" / "missing_metric_audit.csv")
    audit_tstr = next(row for row in audit_rows if row["metric"] == "tstr_accuracy")
    assert audit_tstr["dataset"] == "CWRU"
    assert audit_tstr["method"] == "cfm_grid"
    assert audit_tstr["seed"] == "0"
    assert "real_labels and fake_labels" in audit_tstr["reason"]


def test_paperpack_single_run_without_manifest_still_writes_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "single"
    _write_metrics(
        run_dir / "generative_eval_metrics.csv",
        [
            {
                "temporal_l1": "1.0",
                "temporal_l1_status": "ok",
                "spectral_fft_l1": "2.0",
                "tstr_accuracy": "0.5",
                "tstr_accuracy_status": "ok",
                "parameter_count": "10",
                "sampling_nfe": "8",
                "leakage_nearest_neighbor_l2": "0.5",
            }
        ],
    )

    paperpack = build_paperpack(run_dir)

    assert (paperpack / "reproducibility_statement.md").exists()
    assert _read_csv(paperpack / "tables" / "table_quality_mean_std.csv")
    assert _read_csv(paperpack / "appendix" / "run_index.csv")

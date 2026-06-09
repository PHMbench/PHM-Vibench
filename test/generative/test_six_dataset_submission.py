from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generative_benchmark_effect import (
    build_effect_report,
    build_run_plan,
    load_matrix,
    validate_matrix_inputs,
)
from scripts.generative_submission_draft import (
    assert_no_placeholders,
    build_draft,
    main as draft_main,
    readiness,
)


def _write_multi_dataset_matrix(path: Path, data_dir: Path) -> None:
    path.write_text(
        f"""
benchmark:
  id: "fixture_six_dataset"
  output_dir: "{path.parent / 'out'}"
  baseline_method: "cfm_grid"
  min_datasets: 2
  python: "python"
  seeds: [0, 1]
  resource:
    gpu_ids: [6, 7]
    max_parallel_runs: 2
    require_cuda: true
  data_check:
    data_dir: "{data_dir}"
    metadata_file: "metadata.xlsx"
  overrides:
    data.data_dir: "{data_dir}"
    data.metadata_file: "metadata.xlsx"
    trainer.device: "cuda"
    trainer.gpus: 1
datasets:
  - dataset: "RM_001_CWRU"
    dataset_id: 1
    name: "CWRU"
    overrides:
      task.target_system_id: [1]
      task.source_domain_id: [0, 1, 2]
      task.target_domain_id: [3]
  - dataset: "RM_027_PU"
    dataset_id: 20
    name: "PU"
    overrides:
      task.target_system_id: [20]
      task.source_domain_id: [0, 1, 2]
      task.target_domain_id: [3]
methods:
  - method: "cfm_grid"
    label: "CFM"
    train_config: "configs/demo/10_generative/dummy_generative_cfm.yaml"
    condition_sampling_policy: "grid"
""".strip(),
        encoding="utf-8",
    )


def _write_six_dataset_effect_matrix(path: Path) -> None:
    datasets_yaml = "\n".join(
        [
            f'''  - dataset: "D{i}"
    dataset_id: {i}
    name: "Dataset {i}"'''
            for i in range(1, 7)
        ]
    )
    path.write_text(
        f"""
benchmark:
  id: "fixture_six_dataset_effect"
  output_dir: "{path.parent / 'out'}"
  baseline_method: "cfm_grid"
  min_datasets: 6
  python: "python"
  seeds: [0, 1]
datasets:
{datasets_yaml}
methods:
  - method: "cfm_grid"
    label: "CFM"
    train_config: "configs/demo/10_generative/dummy_generative_cfm.yaml"
  - method: "rectified_flow_grid"
    label: "Rectified Flow"
    train_config: "configs/demo/10_generative/dummy_generative_rectified_flow.yaml"
""".strip(),
        encoding="utf-8",
    )


def _write_eval_metrics(path: Path, *, temporal: str, tstr: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "temporal_l1",
                "temporal_l1_status",
                "tstr_accuracy",
                "tstr_accuracy_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "temporal_l1": temporal,
                "temporal_l1_status": "ok",
                "tstr_accuracy": tstr,
                "tstr_accuracy_status": "ok",
            }
        )


def _write_valid_synthetic_manifest(
    path: Path, *, dataset: str, method: str, seed: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "synthetic_dataset_id": f"{dataset}_{method}_seed{seed}",
                "dataset": dataset,
                "sampling": {"seed": seed},
                "validity": {
                    "status": "benchmark-valid",
                    "benchmark_valid": True,
                    "missing_evidence": [],
                },
            }
        ),
        encoding="utf-8",
    )


def _ready_summary() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        rows.append(
            {
                "dataset": dataset,
                "method": "cfm_grid",
                "metric": "temporal_l1",
                "category": "quality",
                "mean": "1.0",
                "n": "2",
                "missing_count": "0",
                "rank": "1",
                "delta_vs_baseline": "0.0",
                "benchmark_status": "benchmark-valid",
                "metric_source_paths": f"runs/{dataset}/metrics.csv",
                "manifest_paths": f"runs/{dataset}/manifest.json",
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "method": "cfm_grid",
                "metric": "tstr_accuracy",
                "category": "utility",
                "mean": "0.8",
                "n": "2",
                "missing_count": "0",
                "rank": "1",
                "delta_vs_baseline": "0.0",
                "benchmark_status": "benchmark-valid",
                "metric_source_paths": f"runs/{dataset}/metrics.csv",
                "manifest_paths": f"runs/{dataset}/manifest.json",
            }
        )
    return rows


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "metric",
        "category",
        "mean",
        "n",
        "missing_count",
        "rank",
        "delta_vs_baseline",
        "benchmark_status",
        "metric_source_paths",
        "manifest_paths",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_multi_dataset_matrix_assigns_gpu_resources_and_dataset_overrides(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_multi_dataset_matrix(matrix_path, data_dir)

    matrix = load_matrix(matrix_path)
    assert validate_matrix_inputs(matrix, allow_missing_data=False) == []
    rows = build_run_plan(matrix, [])

    assert len(rows) == 2 * 1 * 2 * 4
    assert {row["dataset"] for row in rows} == {"RM_001_CWRU", "RM_027_PU"}
    assert {row["gpu_id"] for row in rows} == {"6", "7"}
    assert rows[0]["command"].startswith("env CUDA_VISIBLE_DEVICES=6 python main.py")
    assert rows[1]["command"].startswith("env CUDA_VISIBLE_DEVICES=7 python main.py")
    assert (
        "task.target_system_id='[1]'" in rows[0]["command"]
        or "task.target_system_id=[1]" in rows[0]["command"]
    )
    assert any("RM_027_PU" in row["command"] for row in rows)


def test_repository_six_dataset_matrix_builds_complete_run_plan() -> None:
    matrix = load_matrix(
        Path("configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml")
    )
    rows = build_run_plan(matrix, [])

    assert len(rows) == 6 * 3 * 2 * 4
    assert {row["dataset"] for row in rows} == {
        "RM_001_CWRU",
        "RM_002_XJTU",
        "RM_003_FEMTO",
        "RM_008_UNSW",
        "RM_024_JUST",
        "RM_027_PU",
    }
    assert {row["method"] for row in rows} == {
        "cfm_grid",
        "rectified_flow_grid",
        "ddpm_train_distribution",
    }
    assert {row["seed"] for row in rows} == {0, 1}
    assert {row["stage"] for row in rows} == {"train", "sample", "eval", "paperpack"}
    assert {row["gpu_id"] for row in rows} == {"6", "7"}
    main_rows = [row for row in rows if row["stage"] != "paperpack"]
    paperpack_rows = [row for row in rows if row["stage"] == "paperpack"]
    assert all("trainer.device=cuda" in row["command"] for row in main_rows)
    assert all("trainer.gpus=1" in row["command"] for row in main_rows)
    assert all("python -m scripts.paperpack_generative" in row["command"] for row in paperpack_rows)


def test_six_dataset_fixture_aggregation_writes_paper_evidence(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_six_dataset_effect_matrix(matrix_path)
    matrix = load_matrix(matrix_path)
    root = tmp_path / "runs"
    for dataset_index, dataset in enumerate([f"D{i}" for i in range(1, 7)], start=1):
        for method in ["cfm_grid", "rectified_flow_grid"]:
            for seed in [0, 1]:
                run_dir = root / dataset / method / f"seed_{seed}" / "eval"
                temporal = str(float(dataset_index + seed))
                tstr = str(0.70 + dataset_index * 0.01 + seed * 0.001)
                _write_eval_metrics(
                    run_dir / "generative_eval_metrics.csv",
                    temporal=temporal,
                    tstr=tstr,
                )
                _write_valid_synthetic_manifest(
                    run_dir / "synthetic" / "synthetic_data_manifest.json",
                    dataset=dataset,
                    method=method,
                    seed=seed,
                )

    out = build_effect_report(matrix, [root], tmp_path / "effect")
    summary = list(csv.DictReader((out / "benchmark_effect_summary.csv").open()))

    assert (out / "benchmark_effect_report.md").exists()
    assert (out / "benchmark_effect_manifest.json").exists()
    assert (out / "missing_metrics.md").exists()
    manifest = json.loads(
        (out / "benchmark_effect_manifest.json").read_text(encoding="utf-8")
    )
    assert {row["dataset"] for row in summary} == {f"D{i}" for i in range(1, 7)}
    assert {row["category"] for row in summary} == {"quality", "utility"}
    assert all(row["benchmark_status"] == "benchmark-valid" for row in summary)
    assert all(row["manifest_paths"] for row in summary)
    assert all(row["metric_source_paths"] for row in summary)
    assert manifest["configured_dataset_count"] == 6
    assert manifest["observed_dataset_count"] == 6
    assert manifest["observed_configured_dataset_count"] == 6
    assert manifest["observed_datasets"] == [f"D{i}" for i in range(1, 7)]
    assert manifest["observed_configured_datasets"] == [f"D{i}" for i in range(1, 7)]
    assert manifest["missing_datasets"] == []
    assert manifest["unexpected_datasets"] == []
    assert manifest["min_datasets_met"] is True
    assert manifest["input_gaps"] == []


def test_six_dataset_fixture_manifest_records_missing_dataset_evidence(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_six_dataset_effect_matrix(matrix_path)
    matrix = load_matrix(matrix_path)
    root = tmp_path / "runs"
    for dataset in [f"D{i}" for i in range(1, 6)]:
        run_dir = root / dataset / "cfm_grid" / "seed_0" / "eval"
        _write_eval_metrics(
            run_dir / "generative_eval_metrics.csv",
            temporal="1.0",
            tstr="0.8",
        )
        _write_valid_synthetic_manifest(
            run_dir / "synthetic" / "synthetic_data_manifest.json",
            dataset=dataset,
            method="cfm_grid",
            seed=0,
        )

    out = build_effect_report(matrix, [root], tmp_path / "effect")
    manifest = json.loads(
        (out / "benchmark_effect_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["observed_dataset_count"] == 5
    assert manifest["observed_configured_dataset_count"] == 5
    assert manifest["missing_datasets"] == ["D6"]
    assert manifest["unexpected_datasets"] == []
    assert manifest["min_datasets_met"] is False
    assert "missing configured dataset evidence: D6" in manifest["input_gaps"]
    assert any("below required minimum 6" in gap for gap in manifest["input_gaps"])


def test_six_dataset_fixture_manifest_records_unexpected_dataset_evidence(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_six_dataset_effect_matrix(matrix_path)
    matrix = load_matrix(matrix_path)
    root = tmp_path / "runs"
    for dataset in [f"D{i}" for i in range(1, 7)] + ["D7"]:
        run_dir = root / dataset / "cfm_grid" / "seed_0" / "eval"
        _write_eval_metrics(
            run_dir / "generative_eval_metrics.csv",
            temporal="1.0",
            tstr="0.8",
        )
        _write_valid_synthetic_manifest(
            run_dir / "synthetic" / "synthetic_data_manifest.json",
            dataset=dataset,
            method="cfm_grid",
            seed=0,
        )

    out = build_effect_report(matrix, [root], tmp_path / "effect")
    manifest = json.loads(
        (out / "benchmark_effect_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["observed_dataset_count"] == 7
    assert manifest["observed_configured_dataset_count"] == 6
    assert manifest["missing_datasets"] == []
    assert manifest["unexpected_datasets"] == ["D7"]
    assert manifest["min_datasets_met"] is True
    assert "unexpected observed dataset evidence: D7" in manifest["input_gaps"]


def test_six_dataset_fixture_unexpected_dataset_does_not_satisfy_minimum(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_six_dataset_effect_matrix(matrix_path)
    matrix = load_matrix(matrix_path)
    root = tmp_path / "runs"
    for dataset in [f"D{i}" for i in range(1, 6)] + ["D7"]:
        run_dir = root / dataset / "cfm_grid" / "seed_0" / "eval"
        _write_eval_metrics(
            run_dir / "generative_eval_metrics.csv",
            temporal="1.0",
            tstr="0.8",
        )
        _write_valid_synthetic_manifest(
            run_dir / "synthetic" / "synthetic_data_manifest.json",
            dataset=dataset,
            method="cfm_grid",
            seed=0,
        )

    out = build_effect_report(matrix, [root], tmp_path / "effect")
    manifest = json.loads(
        (out / "benchmark_effect_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["observed_dataset_count"] == 6
    assert manifest["observed_configured_dataset_count"] == 5
    assert manifest["missing_datasets"] == ["D6"]
    assert manifest["unexpected_datasets"] == ["D7"]
    assert manifest["min_datasets_met"] is False
    assert any(
        "observed configured 5 dataset(s), below required minimum 6" in gap
        for gap in manifest["input_gaps"]
    )


def test_submission_draft_marks_ready_only_with_six_valid_datasets() -> None:
    manifest = {
        "benchmark_id": "fixture_six_dataset",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
        "min_datasets_met": True,
        "observed_configured_dataset_count": 6,
        "missing_datasets": [],
        "unexpected_datasets": [],
        "input_gaps": [],
        "datasets": [{"dataset": f"D{i}"} for i in range(1, 7)],
    }
    summary = _ready_summary()

    ready, reasons = readiness(summary, manifest)
    assert ready
    assert reasons == []
    draft = build_draft(summary, manifest)
    assert "**Draft status:** `SUBMISSION_READY`" in draft
    assert "D1" in draft
    assert "Metric Source | Manifest Source" in draft
    assert "runs/D1/metrics.csv" in draft
    assert "runs/D1/manifest.json" in draft
    assert_no_placeholders(draft)


def test_submission_draft_cli_writes_ready_sidecars(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "PAPER_DRAFT.md"
    _write_summary(summary_path, _ready_summary())
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_id": "fixture_six_dataset",
                "baseline_method": "cfm_grid",
                "min_datasets": 6,
                "min_datasets_met": True,
                "observed_configured_dataset_count": 6,
                "missing_datasets": [],
                "unexpected_datasets": [],
                "input_gaps": [],
                "datasets": [{"dataset": f"D{i}"} for i in range(1, 7)],
            }
        ),
        encoding="utf-8",
    )

    rc = draft_main(
        [
            "--summary",
            str(summary_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-submission-ready",
        ]
    )

    assert rc == 0
    draft = output_path.read_text(encoding="utf-8")
    evidence_gaps = (tmp_path / "evidence_gaps.md").read_text(encoding="utf-8")
    readiness_note = (tmp_path / "submission_readiness.md").read_text(
        encoding="utf-8"
    )
    assert "**Draft status:** `SUBMISSION_READY`" in draft
    assert "No evidence gaps were reported" in evidence_gaps
    assert "Status: `SUBMISSION_READY`" in readiness_note
    assert "all draft readiness gates passed" in readiness_note
    assert_no_placeholders(draft)


def test_submission_draft_refuses_submission_ready_when_evidence_is_incomplete() -> (
    None
):
    manifest = {
        "benchmark_id": "fixture",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
    }
    summary = _ready_summary()[:4]
    summary[0]["benchmark_status"] = "exploratory"

    ready, reasons = readiness(summary, manifest)
    assert not ready
    assert any("requires at least 6 datasets" in reason for reason in reasons)
    assert any("benchmark-valid" in reason for reason in reasons)
    draft = build_draft(summary, manifest)
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "No numerical claim" in draft
    assert_no_placeholders(draft)


def test_submission_draft_requires_quality_and_utility_for_each_dataset() -> None:
    manifest = {
        "benchmark_id": "fixture",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
    }
    summary = [
        row
        for row in _ready_summary()
        if not (row["dataset"] == "D6" and row["category"] == "utility")
    ]

    ready, reasons = readiness(summary, manifest)

    assert not ready
    assert any("quality and utility evidence" in reason for reason in reasons)
    draft = build_draft(summary, manifest)
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "quality and utility evidence" in draft
    assert_no_placeholders(draft)


def test_submission_draft_requires_metric_and_manifest_source_paths() -> None:
    manifest = {
        "benchmark_id": "fixture_six_dataset",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
        "min_datasets_met": True,
        "observed_configured_dataset_count": 6,
        "missing_datasets": [],
        "unexpected_datasets": [],
        "input_gaps": [],
        "datasets": [{"dataset": f"D{i}"} for i in range(1, 7)],
    }
    summary = _ready_summary()
    summary[0]["metric_source_paths"] = ""
    summary[1]["manifest_paths"] = ""

    ready, reasons = readiness(summary, manifest)

    assert not ready
    assert any("missing metric source paths" in reason for reason in reasons)
    assert any("missing manifest source paths" in reason for reason in reasons)
    draft = build_draft(summary, manifest)
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "missing metric source paths" in draft
    assert "missing manifest source paths" in draft
    assert_no_placeholders(draft)


def test_submission_draft_requires_manifest_coverage_fields_for_ready() -> None:
    manifest = {
        "benchmark_id": "fixture_six_dataset",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
        "datasets": [{"dataset": f"D{i}"} for i in range(1, 7)],
    }
    summary = _ready_summary()

    ready, reasons = readiness(summary, manifest)

    assert not ready
    assert "benchmark-effect manifest missing input_gaps field" in reasons
    assert "benchmark-effect manifest missing missing_datasets field" in reasons
    assert "benchmark-effect manifest missing unexpected_datasets field" in reasons
    assert "benchmark-effect manifest missing min_datasets_met=true" in reasons
    assert (
        "benchmark-effect manifest missing observed_configured_dataset_count field"
        in reasons
    )
    draft = build_draft(summary, manifest)
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "benchmark-effect manifest missing min_datasets_met=true" in draft
    assert_no_placeholders(draft)


def test_submission_draft_requires_observed_configured_dataset_count() -> None:
    manifest = {
        "benchmark_id": "fixture_six_dataset",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
        "min_datasets_met": True,
        "missing_datasets": [],
        "unexpected_datasets": [],
        "input_gaps": [],
        "datasets": [{"dataset": f"D{i}"} for i in range(1, 7)],
    }
    summary = _ready_summary()

    ready, reasons = readiness(summary, manifest)

    assert not ready
    assert (
        "benchmark-effect manifest missing observed_configured_dataset_count field"
        in reasons
    )

    manifest["observed_configured_dataset_count"] = 5
    ready, reasons = readiness(summary, manifest)

    assert not ready
    assert any(
        "observed configured 5 dataset(s), below required minimum 6" in reason
        for reason in reasons
    )


def test_submission_draft_respects_manifest_dataset_coverage_gaps() -> None:
    manifest = {
        "benchmark_id": "fixture_six_dataset",
        "baseline_method": "cfm_grid",
        "min_datasets": 6,
        "observed_dataset_count": 5,
        "observed_configured_dataset_count": 5,
        "missing_datasets": ["D6"],
        "unexpected_datasets": ["D7"],
        "min_datasets_met": False,
        "datasets": [{"dataset": f"D{i}"} for i in range(1, 7)],
    }
    summary = _ready_summary()

    ready, reasons = readiness(summary, manifest)
    assert not ready
    assert "missing configured dataset evidence: D6" in reasons
    assert "unexpected observed dataset evidence: D7" in reasons
    assert any("below required minimum 6" in reason for reason in reasons)
    draft = build_draft(summary, manifest)
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "missing configured dataset evidence: D6" in draft
    assert "unexpected observed dataset evidence: D7" in draft
    assert_no_placeholders(draft)


def test_submission_draft_cli_requires_submission_ready_evidence(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "PAPER_DRAFT.md"
    summary = _ready_summary()[:4]
    summary[0]["benchmark_status"] = "exploratory"
    _write_summary(summary_path, summary)
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_id": "fixture",
                "baseline_method": "cfm_grid",
                "min_datasets": 6,
            }
        ),
        encoding="utf-8",
    )

    rc = draft_main(
        [
            "--summary",
            str(summary_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-submission-ready",
        ]
    )

    assert rc == 2
    draft = output_path.read_text(encoding="utf-8")
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "requires at least 6 datasets" in draft
    assert_no_placeholders(draft)


def test_submission_draft_cli_writes_blocked_draft_when_inputs_are_missing(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "PAPER_DRAFT.md"
    missing_summary = tmp_path / "missing_summary.csv"
    missing_manifest = tmp_path / "missing_manifest.json"

    rc = draft_main(
        [
            "--summary",
            str(missing_summary),
            "--manifest",
            str(missing_manifest),
            "--output",
            str(output_path),
            "--require-submission-ready",
        ]
    )

    assert rc == 2
    draft = output_path.read_text(encoding="utf-8")
    assert "**Draft status:** `NOT_SUBMISSION_READY`" in draft
    assert "required summary file not found" in draft
    assert "manifest file not found" in draft
    assert_no_placeholders(draft)
    evidence_gaps = (tmp_path / "evidence_gaps.md").read_text(encoding="utf-8")
    readiness_note = (tmp_path / "submission_readiness.md").read_text(
        encoding="utf-8"
    )
    assert "required summary file not found" in evidence_gaps
    assert "manifest file not found" in evidence_gaps
    assert "Status: `NOT_SUBMISSION_READY`" in readiness_note
    assert "required summary file not found" in readiness_note


def test_m2_blocked_run_status_ledger_matches_dry_run_plan() -> None:
    plan_path = Path(
        "results/paper/phm_generative/six_dataset_submission_v1/dry_run/run_plan.csv"
    )
    ledger_path = Path(
        "specs/002-phm-genbench-frontier/reviews/codex/"
        "2026-05-11-m2-run-status-ledger.csv"
    )
    assert plan_path.exists()
    assert ledger_path.exists()

    plan_rows = list(csv.DictReader(plan_path.open()))
    ledger_rows = list(csv.DictReader(ledger_path.open()))

    plan_groups = {
        (
            row["dataset"],
            row["dataset_name"],
            row["method"],
            row["method_label"],
            row["seed"],
        )
        for row in plan_rows
    }
    ledger_groups = {
        (
            row["dataset"],
            row["dataset_name"],
            row["method"],
            row["method_label"],
            row["seed"],
        )
        for row in ledger_rows
    }

    assert len(plan_rows) == 144
    assert len(plan_groups) == 36
    assert ledger_groups == plan_groups
    assert {row["status"] for row in ledger_rows} == {"BLOCKED_GPU_PREFLIGHT"}
    assert {row["planned_stages"] for row in ledger_rows} == {
        "train;sample;eval;paperpack"
    }


def test_m2_gpu_preflight_report_is_reviewable_evidence() -> None:
    report_path = Path(
        "specs/002-phm-genbench-frontier/reviews/codex/"
        "2026-05-12-gpu-preflight-report.json"
    )
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["benchmark_id"] == "phm_genbench_six_dataset_submission_v1"
    assert (
        report["matrix_path"]
        == "configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml"
    )
    assert report["require_cuda"] is True
    assert report["gpu_ids"] == ["6", "7"]
    assert report["max_parallel_runs"] == 2
    assert report["source_report"].endswith("gpu_preflight_report.json")
    assert {row["gpu_id"] for row in report["results"]} == {"6", "7"}
    if report["passed"] is True:
        assert {row["status"] for row in report["results"]} == {"passed"}
        assert all(row["returncode"] == 0 for row in report["results"])
        assert all("NVIDIA" in row["stdout_tail"] for row in report["results"])
        assert all(row["error"] == "" for row in report["results"])
    else:
        assert report["passed"] is False
        assert {row["status"] for row in report["results"]} == {"failed"}
        assert all("torch cuda unavailable" in row["error"] for row in report["results"])

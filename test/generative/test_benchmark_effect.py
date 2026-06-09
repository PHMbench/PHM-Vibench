from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts.generative_benchmark_effect import (
    build_effect_report,
    build_run_plan,
    execute_plan,
    load_matrix,
    main as benchmark_effect_main,
    preflight_gpu_resource,
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


def _write_metrics(
    path: Path, *, temporal: str, tstr: str, trts: str, reason: str = ""
) -> None:
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


def test_dry_run_stage_filter_writes_only_requested_stage(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    out_dir = tmp_path / "stage_plan"

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--dry-run",
            "--stages",
            "train",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    rows = _read_csv(out_dir / "run_plan.csv")
    assert len(rows) == 3 * 2
    assert {row["stage"] for row in rows} == {"train"}


def test_dry_run_rejects_unknown_stage_filter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--dry-run",
            "--stages",
            "trian",
            "--output-dir",
            str(tmp_path / "stage_plan"),
        ]
    )

    assert rc == 2
    assert not (tmp_path / "stage_plan" / "run_plan.csv").exists()


def test_execute_requires_gpu_preflight_when_matrix_requires_cuda(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "seeds: [0, 1]",
        "seeds: [0, 1]\n  resource:\n    gpu_ids: [6, 7]\n    require_cuda: true",
    )
    matrix_path.write_text(text, encoding="utf-8")
    out_dir = tmp_path / "stage_plan"

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--execute",
            "--stages",
            "train",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "run_plan.csv").exists()
    assert not (out_dir / "execution_summary.csv").exists()


def test_execute_requires_single_stage_when_matrix_requires_cuda(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "seeds: [0, 1]",
        "seeds: [0, 1]\n  resource:\n    gpu_ids: [6, 7]\n    require_cuda: true",
    )
    matrix_path.write_text(text, encoding="utf-8")
    out_dir = tmp_path / "stage_plan"

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--execute",
            "--preflight-gpu",
            "--stages",
            "train,sample",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "run_plan.csv").exists()
    assert not (out_dir / "execution_summary.csv").exists()


def test_cli_rejects_multiple_primary_modes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    out_dir = tmp_path / "stage_plan"

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--dry-run",
            "--execute",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "run_plan.csv").exists()
    assert not (out_dir / "execution_summary.csv").exists()


def test_from_runs_requires_at_least_one_run_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    out_dir = tmp_path / "effect"

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--from-runs",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "benchmark_effect_summary.csv").exists()
    assert not (out_dir / "benchmark_effect_manifest.json").exists()


def test_from_runs_rejects_run_dir_without_metric_records(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    out_dir = tmp_path / "effect"

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--from-runs",
            str(run_dir),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "benchmark_effect_summary.csv").exists()
    assert not (out_dir / "benchmark_effect_manifest.json").exists()


def test_effect_report_aggregates_quality_utility_delta_and_missing_reasons(
    tmp_path: Path,
) -> None:
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
        _write_metrics(
            run_dir / "generative_eval_metrics.csv",
            temporal=temporal,
            tstr=tstr,
            trts=trts,
            reason=reason,
        )
        _write_manifest(
            run_dir / "synthetic" / "synthetic_data_manifest.json",
            method=method,
            seed=seed,
            valid=valid,
        )

    out = build_effect_report(matrix, [root], tmp_path / "effect")
    summary = _read_csv(out / "benchmark_effect_summary.csv")

    rf_temporal = next(
        row
        for row in summary
        if row["method"] == "rectified_flow_grid" and row["metric"] == "temporal_l1"
    )
    assert float(rf_temporal["mean"]) == 1.0
    assert float(rf_temporal["baseline_mean"]) == 2.0
    assert float(rf_temporal["delta_vs_baseline"]) == -1.0
    assert rf_temporal["rank"] == "1"
    assert "generative_eval_metrics.csv" in rf_temporal["metric_source_paths"]
    assert "synthetic_data_manifest.json" in rf_temporal["manifest_paths"]

    rf_tstr = next(
        row
        for row in summary
        if row["method"] == "rectified_flow_grid" and row["metric"] == "tstr_accuracy"
    )
    assert round(float(rf_tstr["mean"]), 4) == 0.85
    assert round(float(rf_tstr["delta_vs_baseline"]), 4) == 0.25
    assert rf_tstr["rank"] == "1"

    ddpm_tstr = next(
        row
        for row in summary
        if row["method"] == "ddpm_train_distribution"
        and row["metric"] == "tstr_accuracy"
    )
    assert ddpm_tstr["n"] == "0"
    assert ddpm_tstr["missing_count"] == "2"
    assert "labels unavailable" in ddpm_tstr["missing_reasons"]
    assert ddpm_tstr["benchmark_status"] == "exploratory"

    missing = (out / "missing_metrics.md").read_text(encoding="utf-8")
    assert "labels unavailable for utility probe" in missing
    assert (out / "benchmark_effect_report.md").exists()
    assert (out / "benchmark_effect_manifest.json").exists()


def test_matrix_validation_fails_missing_real_phm_data_without_explicit_allow(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, tmp_path / "missing_data")
    matrix = load_matrix(matrix_path)

    errors = validate_matrix_inputs(matrix, allow_missing_data=False)
    assert errors
    assert "missing PHM metadata" in errors[0]
    assert validate_matrix_inputs(matrix, allow_missing_data=True) == []


def test_matrix_validation_rejects_parallel_runs_exceeding_declared_gpus(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "seeds: [0, 1]",
        "seeds: [0, 1]\n  resource:\n    gpu_ids: [6]\n    max_parallel_runs: 2",
    )
    matrix_path.write_text(text, encoding="utf-8")
    matrix = load_matrix(matrix_path)

    errors = validate_matrix_inputs(matrix, allow_missing_data=False)
    assert errors == [
        "resource.max_parallel_runs cannot exceed number of resource.gpu_ids"
    ]


def test_gpu_preflight_checks_each_declared_gpu_and_reports_cuda_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "seeds: [0, 1]",
        "seeds: [0, 1]\n  resource:\n    gpu_ids: [6, 7]\n    require_cuda: true",
    )
    matrix_path.write_text(text, encoding="utf-8")
    matrix = load_matrix(matrix_path)
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output):
        calls.append(list(cmd))
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="AssertionError: torch cuda unavailable\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    errors = preflight_gpu_resource(matrix)

    assert [call[1] for call in calls] == [
        "CUDA_VISIBLE_DEVICES=6",
        "CUDA_VISIBLE_DEVICES=7",
    ]
    assert errors == [
        "GPU 6 failed CUDA preflight: AssertionError: torch cuda unavailable",
        "GPU 7 failed CUDA preflight: AssertionError: torch cuda unavailable",
    ]


def test_gpu_preflight_writes_machine_readable_failure_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "seeds: [0, 1]",
        "seeds: [0, 1]\n  resource:\n    gpu_ids: [6, 7]\n    require_cuda: true",
    )
    matrix_path.write_text(text, encoding="utf-8")
    out_dir = tmp_path / "gpu_preflight"

    def fake_run(cmd, text, capture_output):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="AssertionError: torch cuda unavailable\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--preflight-gpu",
            "--dry-run",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "run_plan.csv").exists()
    ledger = _read_csv(out_dir / "blocked_run_status_ledger.csv")
    assert len(ledger) == 3 * 2
    assert {row["status"] for row in ledger} == {"BLOCKED_GPU_PREFLIGHT"}
    assert {row["planned_stages"] for row in ledger} == {"train;sample;eval;paperpack"}
    assert all("torch cuda unavailable" in row["reason"] for row in ledger)
    report = json.loads((out_dir / "gpu_preflight_report.json").read_text())
    assert report["benchmark_id"] == "fixture_benchmark"
    assert report["matrix_path"] == str(matrix_path)
    assert report["created_at"]
    assert report["python"] == "python"
    assert report["require_cuda"] is True
    assert report["gpu_ids"] == ["6", "7"]
    assert report["max_parallel_runs"] == 2
    assert report["passed"] is False
    assert [row["gpu_id"] for row in report["results"]] == ["6", "7"]
    assert {row["status"] for row in report["results"]} == {"failed"}
    assert all("torch cuda unavailable" in row["error"] for row in report["results"])


def test_execute_preflight_failure_stops_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.xlsx").write_text("placeholder", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, data_dir)
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "seeds: [0, 1]",
        "seeds: [0, 1]\n  resource:\n    gpu_ids: [6, 7]\n    require_cuda: true",
    )
    matrix_path.write_text(text, encoding="utf-8")
    out_dir = tmp_path / "execute_blocked"

    def fake_run(cmd, text, capture_output):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="AssertionError: torch cuda unavailable\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = benchmark_effect_main(
        [
            "--matrix",
            str(matrix_path),
            "--execute",
            "--preflight-gpu",
            "--stages",
            "train",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 2
    assert not (out_dir / "runs").exists()
    assert not (out_dir / "run_plan.csv").exists()
    assert not (out_dir / "execution_summary.csv").exists()
    ledger = _read_csv(out_dir / "blocked_run_status_ledger.csv")
    assert len(ledger) == 3 * 2
    assert {row["planned_stages"] for row in ledger} == {"train"}
    assert {row["status"] for row in ledger} == {"BLOCKED_GPU_PREFLIGHT"}
    report = json.loads((out_dir / "gpu_preflight_report.json").read_text())
    assert report["passed"] is False
    assert [row["gpu_id"] for row in report["results"]] == ["6", "7"]


def test_execute_plan_rejects_unresolved_placeholder_artifact_paths(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "stage": "sample",
            "command": (
                "env CUDA_VISIBLE_DEVICES=7 python main.py --config cfg.yaml "
                "--override task.generative.checkpoint_path=runs/<experiment_name>/best.ckpt"
            ),
        }
    ]

    try:
        execute_plan(rows, tmp_path / "execution_summary.csv")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("execute_plan accepted placeholder artifact paths")

    assert "placeholder base path does not exist" in message
    assert "sample" in message


def test_execute_plan_resolves_existing_placeholder_artifact_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "train"
        / "resolved_experiment"
        / "iter_0"
        / "checkpoints"
        / "best.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    placeholder = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "train"
        / "<experiment_name>"
        / "iter_0"
        / "checkpoints"
        / "best.ckpt"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rows = [
        {
            "benchmark_id": "fixture",
            "dataset": "dataset",
            "dataset_id": "1",
            "dataset_name": "Dataset",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "sample",
            "gpu_id": "7",
            "config": "cfg.yaml",
            "command": (
                "env CUDA_VISIBLE_DEVICES=7 python main.py --config cfg.yaml "
                f"--override task.generative.checkpoint_path={placeholder}"
            ),
        }
    ]

    rc = execute_plan(rows, tmp_path / "execution_summary.csv")

    assert rc == 0
    assert any(str(checkpoint) in arg for arg in calls[0])
    summary = _read_csv(tmp_path / "execution_summary.csv")
    assert "<experiment_name>" not in summary[0]["command"]


def test_execute_plan_resolves_sample_and_paperpack_placeholder_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "sample"
        / "resolved_sample"
        / "iter_0"
        / "synthetic"
        / "samples.pt"
    )
    sample.parent.mkdir(parents=True)
    sample.write_text("samples", encoding="utf-8")
    sample_placeholder = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "sample"
        / "<experiment_name>"
        / "iter_0"
        / "synthetic"
        / "samples.pt"
    )
    eval_iter = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "eval"
        / "resolved_eval"
        / "iter_0"
    )
    eval_iter.mkdir(parents=True)
    eval_placeholder = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "eval"
        / "<experiment_name>"
        / "iter_0"
    )
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rows = [
        {
            "benchmark_id": "fixture",
            "dataset": "dataset",
            "dataset_id": "1",
            "dataset_name": "Dataset",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "eval",
            "gpu_id": "6",
            "config": "cfg.yaml",
            "command": (
                "env CUDA_VISIBLE_DEVICES=6 python main.py --config cfg.yaml "
                f"--override task.generative.generated_path={sample_placeholder}"
            ),
        },
        {
            "benchmark_id": "fixture",
            "dataset": "dataset",
            "dataset_id": "1",
            "dataset_name": "Dataset",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "paperpack",
            "gpu_id": "7",
            "config": "",
            "command": (
                "env CUDA_VISIBLE_DEVICES=7 python -m scripts.paperpack_generative "
                f"--run_dir {eval_placeholder}"
            ),
        },
    ]

    rc = execute_plan(rows, tmp_path / "execution_summary.csv")

    assert rc == 0
    assert any(str(sample) in arg for arg in calls[0])
    assert str(eval_iter) in calls[1]
    summary = _read_csv(tmp_path / "execution_summary.csv")
    assert all("<experiment_name>" not in row["command"] for row in summary)


def test_execute_plan_skip_existing_train_artifact(tmp_path: Path, monkeypatch) -> None:
    train_result = (
        tmp_path
        / "runs"
        / "dataset"
        / "method"
        / "seed_0"
        / "train"
        / "exp"
        / "iter_0"
        / "train_result_0.csv"
    )
    train_result.parent.mkdir(parents=True)
    train_result.write_text("epoch,loss\n0,1.0\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rows = [
        {
            "benchmark_id": "fixture",
            "dataset": "dataset",
            "dataset_id": "1",
            "dataset_name": "Dataset",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "train",
            "gpu_id": "6",
            "config": "cfg.yaml",
            "command": (
                "env CUDA_VISIBLE_DEVICES=6 python main.py --config cfg.yaml "
                f"--override environment.output_dir={tmp_path / 'runs' / 'dataset' / 'method' / 'seed_0' / 'train'}"
            ),
        }
    ]

    rc = execute_plan(rows, tmp_path / "execution_summary.csv", skip_existing=True)

    assert rc == 0
    assert calls == []
    summary = _read_csv(tmp_path / "execution_summary.csv")
    assert summary[0]["returncode"] == "0"
    assert "[SKIP] existing train artifact" in summary[0]["stdout_tail"]


def test_execute_plan_max_runs_limits_non_skipped_commands(
    tmp_path: Path, monkeypatch
) -> None:
    train_result = tmp_path / "runs" / "done" / "exp" / "train_result_0.csv"
    train_result.parent.mkdir(parents=True)
    train_result.write_text("epoch,loss\n0,1.0\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rows = [
        {
            "benchmark_id": "fixture",
            "dataset": "done",
            "dataset_id": "1",
            "dataset_name": "Done",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "train",
            "gpu_id": "6",
            "config": "cfg.yaml",
            "command": (
                "env CUDA_VISIBLE_DEVICES=6 python main.py --config cfg.yaml "
                f"--override environment.output_dir={tmp_path / 'runs' / 'done'}"
            ),
        },
        {
            "benchmark_id": "fixture",
            "dataset": "todo1",
            "dataset_id": "1",
            "dataset_name": "Todo1",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "train",
            "gpu_id": "6",
            "config": "cfg.yaml",
            "command": (
                "env CUDA_VISIBLE_DEVICES=6 python main.py --config cfg.yaml "
                f"--override environment.output_dir={tmp_path / 'runs' / 'todo1'}"
            ),
        },
        {
            "benchmark_id": "fixture",
            "dataset": "todo2",
            "dataset_id": "1",
            "dataset_name": "Todo2",
            "method": "method",
            "method_label": "Method",
            "seed": 0,
            "stage": "train",
            "gpu_id": "6",
            "config": "cfg.yaml",
            "command": (
                "env CUDA_VISIBLE_DEVICES=6 python main.py --config cfg.yaml "
                f"--override environment.output_dir={tmp_path / 'runs' / 'todo2'}"
            ),
        },
    ]

    rc = execute_plan(
        rows,
        tmp_path / "execution_summary.csv",
        skip_existing=True,
        max_runs=1,
    )

    assert rc == 0
    assert len(calls) == 1
    assert any("todo1" in arg for arg in calls[0])
    summary = _read_csv(tmp_path / "execution_summary.csv")
    assert [row["dataset"] for row in summary] == ["done", "todo1"]

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from apps.streamlit.result_service import (
    DiscoveryLimits,
    artifact_groups,
    discover_results,
    load_metric_table,
    parse_direct_results,
    primary_metric_headlines,
)
from apps.streamlit.run_service import RunRecord


CONFIG = '''\
environment:
  seed: 0
  output_dir: results/demo
data:
  data_dir: data
  metadata_file: dummy.csv
model:
  name: Dummy
  type: Dummy
task:
  name: classification
  type: DG
trainer:
  num_epochs: 1
  device: cpu
  test_after_fit: true
'''


def record(
    tmp_path: Path,
    *,
    status: str = "succeeded",
    exit_code: int | None = 0,
    test_after_fit: bool = True,
) -> tuple[Path, RunRecord]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("")
    (repo / "configs").mkdir()
    run_dir = repo / "outputs" / "streamlit" / "run-1"
    run_dir.mkdir(parents=True)
    config = CONFIG.replace(
        "  test_after_fit: true",
        f"  test_after_fit: {'true' if test_after_fit else 'false'}",
    )
    (run_dir / "execution.yaml").write_text(config, encoding="utf-8")
    rec = RunRecord(
        run_id="run-1",
        status=status,
        run_dir=run_dir,
        command=("python", "main.py"),
        output_root="results/demo",
        started_at=datetime.now(timezone.utc).isoformat(),
        exit_code=exit_code,
    )
    return repo, rec


def write_direct_result(
    repo: Path,
    rec: RunRecord,
    *,
    result_name: str = "run-a",
    include_evaluation: bool = True,
) -> Path:
    result_dir = repo / "results" / "demo" / result_name
    result_dir.mkdir(parents=True)
    checkpoint = result_dir / "iter_0" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("checkpoint", encoding="utf-8")
    lines = [
        f"result_dir={result_dir}",
        f"best_checkpoint={checkpoint}",
    ]
    if include_evaluation:
        metrics = result_dir / "all_results.csv"
        summary = result_dir / "run_summary.json"
        metrics.write_text("acc,loss\n0.9,0.1\n", encoding="utf-8")
        summary.write_text(
            json.dumps({"iterations": 1, "metrics": {"acc": {"count": 1, "mean": 0.9}}}),
            encoding="utf-8",
        )
        lines.extend((f"test_metrics={metrics}", f"run_summary={summary}"))
        primary = {"acc": {"count": 1, "mean": 0.9, "sample_std": None}}
    else:
        primary = {}
    lines.extend((f"primary_metrics={json.dumps(primary)}", "run=completed"))
    (rec.run_dir / "run.log").write_text(
        "training output\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result_dir


def test_direct_cli_result_is_the_only_scientific_root(tmp_path: Path):
    repo, rec = record(tmp_path)
    result_dir = write_direct_result(repo, rec)
    (result_dir / "plot.png").write_bytes(b"png")

    foreign = repo / "results" / "demo" / "newer-foreign-run"
    foreign.mkdir(parents=True)
    (foreign / "all_results.csv").write_text("acc\n0.01\n", encoding="utf-8")
    (foreign / "foreign.png").write_bytes(b"foreign")
    newer = (result_dir.stat().st_mtime + 60, result_dir.stat().st_mtime + 60)
    os.utime(foreign / "all_results.csv", newer)
    os.utime(foreign / "foreign.png", newer)

    bundle = discover_results(repo, rec)
    assert bundle.direct.completed
    assert bundle.direct.result_dir == result_dir.resolve()
    assert bundle.roots == (rec.run_dir.resolve(), result_dir.resolve())
    assert all("newer-foreign-run" not in str(item.path) for item in bundle.artifacts)
    groups = artifact_groups(bundle)
    assert groups["image"][0].path.name == "plot.png"
    assert bundle.metrics[0].rows[0]["acc"] == "0.9"
    assert primary_metric_headlines(bundle.direct.primary_metrics) == (("acc", 0.9),)


def test_cli_paths_outside_reported_result_dir_are_not_consumed(tmp_path: Path):
    repo, rec = record(tmp_path)
    result_dir = write_direct_result(repo, rec)
    outside = repo / "results" / "demo" / "other" / "all_results.csv"
    outside.parent.mkdir(parents=True)
    outside.write_text("acc\n1.0\n", encoding="utf-8")
    log = (rec.run_dir / "run.log").read_text(encoding="utf-8")
    log = log.replace(
        f"test_metrics={result_dir / 'all_results.csv'}", f"test_metrics={outside}"
    )
    (rec.run_dir / "run.log").write_text(log, encoding="utf-8")

    bundle = discover_results(repo, rec)
    assert bundle.direct.test_metrics is None
    assert any("outside reported result_dir" in item for item in bundle.warnings)
    assert all(table.source != outside.resolve() for table in bundle.metrics)


def test_failed_run_does_not_accept_stale_success_lines(tmp_path: Path):
    repo, rec = record(tmp_path, status="failed", exit_code=7)
    result_dir = write_direct_result(repo, rec)
    bundle = discover_results(repo, rec)
    assert not bundle.direct.completed
    assert bundle.direct.result_dir is None
    assert bundle.roots == (rec.run_dir.resolve(),)
    assert result_dir.resolve() not in bundle.roots
    assert any("non-successful" in item for item in bundle.warnings)


def test_success_without_final_trailer_does_not_scan_output_root(tmp_path: Path):
    repo, rec = record(tmp_path)
    foreign = repo / "results" / "demo" / "foreign"
    foreign.mkdir(parents=True)
    (foreign / "all_results.csv").write_text("acc\n1.0\n", encoding="utf-8")
    (rec.run_dir / "run.log").write_text(
        "training ended without public trailer\n", encoding="utf-8"
    )
    bundle = discover_results(repo, rec)
    assert bundle.direct.result_dir is None
    assert bundle.roots == (rec.run_dir.resolve(),)
    assert not bundle.metrics
    assert any("run=completed" in item for item in bundle.warnings)


def test_training_only_run_does_not_require_test_files(tmp_path: Path):
    repo, rec = record(tmp_path, test_after_fit=False)
    result_dir = write_direct_result(repo, rec, include_evaluation=False)
    direct = parse_direct_results(repo, rec)
    assert direct.completed
    assert direct.result_dir == result_dir.resolve()
    assert direct.best_checkpoint is not None
    assert direct.evaluation_requested is False
    assert direct.test_metrics is None
    assert direct.run_summary is None
    assert not any("Evaluation was requested" in item for item in direct.warnings)


def test_primary_metric_headlines_use_cli_primary_summary_not_arbitrary_columns():
    primary = {
        "test_f1_demo": {"count": 3, "mean": 0.81, "sample_std": 0.02},
        "test_acc_demo": {"count": 3, "mean": 0.9, "sample_std": 0.01},
        "bad": {"mean": float("nan")},
    }
    assert primary_metric_headlines(primary) == (
        ("test_f1_demo", 0.81),
        ("test_acc_demo", 0.9),
    )


def test_malformed_json_becomes_warning(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text("{bad", encoding="utf-8")
    table = load_metric_table(path)
    assert not table.rows
    assert "Could not parse metrics" in table.warning


def test_large_metric_file_is_not_parsed(tmp_path: Path):
    path = tmp_path / "all_results.csv"
    path.write_text("a\n" + ("1\n" * 100), encoding="utf-8")
    table = load_metric_table(path, DiscoveryLimits(max_metric_bytes=10))
    assert "parsing is limited" in table.warning


def test_symlink_escape_inside_direct_result_is_skipped(tmp_path: Path):
    repo, rec = record(tmp_path)
    result_dir = write_direct_result(repo, rec)
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": 1}')
    try:
        (result_dir / "extra.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        return
    bundle = discover_results(repo, rec)
    assert all(item.path.name != "extra.json" for item in bundle.artifacts)


def test_scan_limits_report_truncation_only_inside_exact_roots(tmp_path: Path):
    repo, rec = record(tmp_path)
    result_dir = write_direct_result(repo, rec)
    for index in range(8):
        (result_dir / f"{index}.txt").write_text("x")
    bundle = discover_results(repo, rec, limits=DiscoveryLimits(max_files=3))
    assert bundle.truncated
    assert set(bundle.roots) == {rec.run_dir.resolve(), result_dir.resolve()}


def test_large_log_reads_bounded_tail_containing_cli_trailer(tmp_path: Path):
    repo, rec = record(tmp_path)
    result_dir = write_direct_result(repo, rec)
    trailer = (rec.run_dir / "run.log").read_text(encoding="utf-8")
    (rec.run_dir / "run.log").write_text(
        ("verbose training line\n" * 5000) + trailer,
        encoding="utf-8",
    )
    direct = parse_direct_results(
        repo, rec, limits=DiscoveryLimits(max_log_bytes=2048)
    )
    assert direct.completed
    assert direct.result_dir == result_dir.resolve()
    assert direct.test_metrics is not None
    assert direct.run_summary is not None

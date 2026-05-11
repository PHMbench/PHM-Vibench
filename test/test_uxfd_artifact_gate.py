import json
from pathlib import Path

import yaml

from scripts.uxfd_artifact_gate import (
    CONDITIONAL_RUN_META_FIELDS,
    QUEUE_METADATA_TO_RUN_META,
    REQUIRED_RUN_META_FIELDS,
    evaluate_artifact_gate,
    main,
)
from scripts.uxfd_gpu_queue import DEFAULT_QUEUE


def _write_valid_artifact(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"accuracy": 1.0}\n', encoding="utf-8")
    (run_dir / "run.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    (run_dir / "run_meta.yaml").write_text(
        "\n".join(
            [
                "cuda_visible_devices: '0'",
                "source_queue_id: 'Q1'",
                "paper_id: 'TII_operator_attention'",
                "phase: 'proposed'",
                "entry_id: 'P00'",
                "gpu_model: 'NVIDIA GeForce RTX 4090'",
                "gpu_count: 1",
                "seed: 0",
                "dataset_split: 'cwru_seed0'",
                "preprocessing_signature: 'sha256:demo'",
                "batch_size: 16",
                "precision: 'fp32'",
                "runtime: '00:01:00'",
                "command: 'CUDA_VISIBLE_DEVICES=0 python main.py --config demo.yaml'",
                "git_sha_or_submodule_sha: 'abc123'",
                "config_path: 'config.yaml'",
                "log_path: 'run.log'",
                "metrics_path: 'metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_artifact_gate_accepts_complete_run_meta(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper01" / "run0")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is True
    assert len(report.records) == 1
    assert report.records[0].accepted is True
    assert report.blockers == ()
    assert report.expected_queue_runs == 0
    assert report.covered_queue_runs == 0
    assert report.queue_coverage_by_paper == {}


def test_artifact_gate_blocks_incomplete_queue_coverage(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")

    report = evaluate_artifact_gate(
        tmp_path,
        queue_path=DEFAULT_QUEUE,
        require_queue_coverage=True,
    )

    assert report.accepted is False
    assert report.records[0].accepted is True
    assert report.expected_queue_runs == 97
    assert report.covered_queue_runs == 1
    assert len(report.missing_queue_runs) == 96
    assert any("queue coverage incomplete" in item for item in report.blockers)
    assert report.queue_coverage_by_paper["TII_operator_attention"]["covered"] == 1
    assert report.queue_coverage_by_paper["TII_operator_attention"]["missing"] == 13
    assert report.queue_coverage_by_paper["TII_operator_attention"]["expected"] == 14


def test_artifact_gate_markdown_reports_queue_coverage_summary(tmp_path: Path) -> None:
    output = tmp_path / "gate" / "artifact_gate.md"

    assert (
        main(
            [
                str(tmp_path / "empty"),
                "--format",
                "markdown",
                "--output",
                str(output),
                "--require-queue-coverage",
                "--allow-not-ready",
            ]
        )
        == 0
    )

    text = output.read_text(encoding="utf-8")
    assert "Queue coverage: `0/97`" in text
    assert "## Queue Coverage By Paper" in text
    assert "`TII_operator_attention`" in text


def test_artifact_gate_blocks_missing_metadata_and_missing_root(tmp_path: Path) -> None:
    missing = evaluate_artifact_gate(tmp_path / "missing")
    assert missing.accepted is False
    assert any("artifact root does not exist" in item for item in missing.blockers)

    run_dir = tmp_path / "bad" / "run0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.yaml").write_text("cuda_visible_devices: '2'\n", encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path / "bad")
    assert report.accepted is False
    assert report.records[0].accepted is False
    assert any("cuda_visible_devices" in item for item in report.records[0].issues)


def test_artifact_gate_rejects_template_placeholders(tmp_path: Path) -> None:
    run_dir = tmp_path / "template" / "run0"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "run.log").write_text("template\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    (run_dir / "run_meta.yaml").write_text(
        "\n".join(
            [
                "accepted_evidence: false",
                "cuda_visible_devices: '0'",
                "gpu_model: 'TODO: NVIDIA GeForce RTX 4090'",
                "gpu_count: 1",
                "seed: 0",
                "dataset_split: 'TODO'",
                "preprocessing_signature: 'TODO'",
                "batch_size: 16",
                "precision: 'fp32'",
                "runtime: 'TODO'",
                "command: 'CUDA_VISIBLE_DEVICES=0 python main.py --config demo.yaml'",
                "git_sha_or_submodule_sha: 'TODO'",
                "config_path: 'config.yaml'",
                "log_path: 'run.log'",
                "metrics_path: 'metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert any("accepted_evidence is false" in item for item in report.records[0].issues)
    assert any("TODO" in item for item in report.records[0].issues)


def test_artifact_gate_cli_writes_json_and_preserves_blocked_exit(tmp_path: Path) -> None:
    output = tmp_path / "gate" / "artifact_gate.json"

    assert main([str(tmp_path / "empty"), "--format", "json", "--output", str(output)]) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accepted"] is False
    assert payload["records"] == []

    assert (
        main(
            [
                str(tmp_path / "empty"),
                "--format",
                "json",
                "--output",
                str(output),
                "--allow-not-ready",
            ]
        )
        == 0
    )


def test_artifact_gate_metadata_contract_matches_gpu_queue() -> None:
    queue = yaml.safe_load(
        Path("paper/UXFD_paper/goal/09_gpu_execution_queue.yaml").read_text(
            encoding="utf-8"
        )
    )

    required_from_queue = set(queue["accepted_run_metadata_required"])
    assert required_from_queue == set(QUEUE_METADATA_TO_RUN_META)

    artifact_fields = set(REQUIRED_RUN_META_FIELDS) | set(CONDITIONAL_RUN_META_FIELDS)
    assert set(QUEUE_METADATA_TO_RUN_META.values()) <= artifact_fields
    assert "oom_or_failure_reason" in CONDITIONAL_RUN_META_FIELDS

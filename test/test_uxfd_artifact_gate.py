import json
from pathlib import Path

import yaml

from scripts.uxfd_artifact_gate import (
    CONDITIONAL_RUN_META_FIELDS,
    QUEUE_METADATA_TO_RUN_META,
    REQUIRED_RUN_META_FIELDS,
    _command_cuda_visible_devices,
    evaluate_artifact_gate,
    main,
    render_markdown,
)
from scripts.uxfd_gpu_queue import DEFAULT_QUEUE


PERSISTED_ARTIFACT_GATE_QUEUE_COVERAGE = Path(
    "paper/UXFD_paper/results/artifact_gate_queue_coverage.md"
)
DEFAULT_ACCEPTED_RUNS_ROOT = Path("paper/UXFD_paper/results/accepted_runs")
VALID_PREPROCESSING_SIGNATURE = "sha256:" + "0123456789abcdef" * 4
VALID_QUEUE_COMMAND = (
    "CUDA_VISIBLE_DEVICES=0 python main.py --config "
    "paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml "
    "--override trainer.num_epochs=1 --override data.num_workers=0"
)


def _write_valid_artifact(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"accuracy": 1.0}\n', encoding="utf-8")
    (run_dir / "run.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    (run_dir / "run_meta.yaml").write_text(
        "\n".join(
            [
                "accepted_evidence: true",
                "cuda_visible_devices: '0'",
                "source_queue_id: 'Q1'",
                "paper_id: 'TII_operator_attention'",
                "phase: 'proposed'",
                "entry_id: 'P00'",
                "gpu_model: 'NVIDIA GeForce RTX 4090'",
                "gpu_count: 1",
                "seed: 0",
                "dataset_split: 'cwru_seed0'",
                f"preprocessing_signature: '{VALID_PREPROCESSING_SIGNATURE}'",
                "batch_size: 16",
                "precision: 'fp32'",
                "runtime: '00:01:00'",
                "evidence_level: 'accepted_same_protocol'",
                f"command: '{VALID_QUEUE_COMMAND}'",
                "git_sha_or_submodule_sha: 'abc123'",
                "source_tree_status: 'clean'",
                "config_path: 'config.yaml'",
                "log_path: 'run.log'",
                "metrics_path: 'metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_single_entry_queue(tmp_path: Path, minimum_seeds: int = 2) -> Path:
    matrix_path = tmp_path / "paper" / "submission_prep" / "baseline_ablation_matrix.yaml"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "paper_id": "TII_operator_attention",
                "proposed": {
                    "id": "P00",
                    "label": "proposed",
                    "command": VALID_QUEUE_COMMAND,
                    "accepted_evidence_status": "pending accepted run",
                },
                "baselines": [],
                "ablations": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(
            {
                "paper_queue": [
                    {
                        "queue_id": "Q1",
                        "paper_id": "TII_operator_attention",
                        "matrix_path": str(matrix_path),
                        "minimum_seeds": minimum_seeds,
                    }
                ],
                "top_representative_bindings": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return queue_path


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
    assert report.expected_queue_runs == 104
    assert report.covered_queue_runs == 1
    assert len(report.missing_queue_runs) == 103
    assert any("queue coverage incomplete" in item for item in report.blockers)
    assert any("queue seed coverage incomplete" in item for item in report.blockers)
    assert report.queue_coverage_by_paper["TII_operator_attention"]["covered"] == 1
    assert report.queue_coverage_by_paper["TII_operator_attention"]["missing"] == 14
    assert report.queue_coverage_by_paper["TII_operator_attention"]["expected"] == 15


def test_artifact_gate_rejects_unknown_queue_coverage_key(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["source_queue_id"] = "UNKNOWN-Q"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(
        tmp_path,
        queue_path=DEFAULT_QUEUE,
        require_queue_coverage=True,
    )

    assert report.records[0].accepted is True
    assert report.covered_queue_runs == 0
    assert any(
        "unknown accepted run_meta.yaml keys" in item for item in report.blockers
    )


def test_artifact_gate_rejects_duplicate_queue_coverage_key(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    _write_valid_artifact(tmp_path / "paper07" / "run1")

    report = evaluate_artifact_gate(
        tmp_path,
        queue_path=DEFAULT_QUEUE,
        require_queue_coverage=True,
    )

    assert all(record.accepted for record in report.records)
    assert report.covered_queue_runs == 1
    assert any(
        "duplicate accepted run_meta.yaml queue+seed keys" in item
        for item in report.blockers
    )


def test_artifact_gate_allows_same_queue_entry_with_distinct_seeds(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "seed0")
    _write_valid_artifact(tmp_path / "paper07" / "seed1")
    run_meta = tmp_path / "paper07" / "seed1" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["seed"] = 1
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is True
    assert len(report.records) == 2
    assert len({record.queue_key for record in report.records}) == 1
    assert len({record.queue_seed_key for record in report.records}) == 2


def test_artifact_gate_requires_minimum_distinct_seeds_for_covered_queue(
    tmp_path: Path,
) -> None:
    queue_path = _write_single_entry_queue(tmp_path, minimum_seeds=2)
    artifact_root = tmp_path / "accepted"
    _write_valid_artifact(artifact_root / "paper07" / "seed0")

    shortfall = evaluate_artifact_gate(
        artifact_root,
        queue_path=queue_path,
        require_queue_coverage=True,
    )

    assert shortfall.covered_queue_runs == 1
    assert shortfall.missing_queue_runs == ()
    assert any("queue seed coverage incomplete" in item for item in shortfall.blockers)

    _write_valid_artifact(artifact_root / "paper07" / "seed1")
    run_meta = artifact_root / "paper07" / "seed1" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["seed"] = 1
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    accepted = evaluate_artifact_gate(
        artifact_root,
        queue_path=queue_path,
        require_queue_coverage=True,
    )

    assert accepted.accepted is True
    assert accepted.covered_queue_runs == 1
    assert accepted.blockers == ()


def test_artifact_gate_rejects_command_mismatch_for_queue_key(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["command"] = "CUDA_VISIBLE_DEVICES=0 python main.py --config wrong.yaml"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(
        tmp_path,
        queue_path=DEFAULT_QUEUE,
        require_queue_coverage=True,
    )

    assert report.records[0].accepted is True
    assert report.covered_queue_runs == 1
    assert any(
        "command does not match queue command" in item for item in report.blockers
    )


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
    assert "Queue coverage: `0/104`" in text
    assert "## Queue Coverage By Paper" in text
    assert "`TII_operator_attention`" in text


def test_persisted_artifact_gate_queue_coverage_matches_current_gate() -> None:
    report = evaluate_artifact_gate(
        DEFAULT_ACCEPTED_RUNS_ROOT,
        queue_path=DEFAULT_QUEUE,
        require_queue_coverage=True,
    )

    assert PERSISTED_ARTIFACT_GATE_QUEUE_COVERAGE.read_text(
        encoding="utf-8"
    ) == render_markdown(report)


def test_accepted_runs_readme_requires_gpu_and_queue_preflight() -> None:
    text = (DEFAULT_ACCEPTED_RUNS_ROOT / "README.md").read_text(encoding="utf-8")

    assert "uxfd_gpu_queue --live-preflight --require-preflight" in text
    assert "Blocked: static queue validation can_execute=False" in text
    assert (
        "uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage"
        in text
    )
    assert "Do not place smoke outputs, templates" in text


def test_artifact_gate_blocks_missing_metadata_and_missing_root(tmp_path: Path) -> None:
    missing = evaluate_artifact_gate(tmp_path / "missing")
    assert missing.accepted is False
    assert any("artifact root does not exist" in item for item in missing.blockers)

    run_dir = tmp_path / "bad" / "run0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.yaml").write_text(
        "cuda_visible_devices: '2'\n", encoding="utf-8"
    )

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
                "evidence_level: 'TODO'",
                "command: 'CUDA_VISIBLE_DEVICES=0 python main.py --config demo.yaml'",
                "git_sha_or_submodule_sha: 'TODO'",
                "source_tree_status: 'TODO'",
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
    assert any(
        "accepted_evidence must be true" in item for item in report.records[0].issues
    )
    assert any("TODO" in item for item in report.records[0].issues)


def test_artifact_gate_requires_explicit_accepted_evidence_true(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data.pop("accepted_evidence")
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "accepted_evidence must be true" in report.records[0].issues


def test_artifact_gate_requires_queue_identity_fields(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data.pop("source_queue_id")
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "missing source_queue_id" in report.records[0].issues


def test_artifact_gate_checks_gpu_count_against_visible_devices(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "top" / "run0")
    run_meta = tmp_path / "top" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["source_queue_id"] = "TOP-Q1-GTM"
    data["phase"] = "top_representatives"
    data["entry_id"] = "B04,B05,A04"
    data["cuda_visible_devices"] = "0,1"
    data["gpu_count"] = 1
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "gpu_count must be 2 for cuda_visible_devices=0,1" in report.records[0].issues
    )


def test_artifact_gate_rejects_non_numeric_run_controls(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["seed"] = "seed0"
    data["batch_size"] = 0
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "seed must be a non-negative integer" in report.records[0].issues
    assert "batch_size must be a positive integer" in report.records[0].issues


def test_artifact_gate_requires_positive_runtime_format(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["runtime"] = "00:00:00"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "runtime must be positive HH:MM:SS" in report.records[0].issues


def test_artifact_gate_rejects_unknown_precision(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["precision"] = "mixed precision"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "precision must be one of fp32, tf32, fp16, bf16, amp"
        in report.records[0].issues
    )


def test_artifact_gate_rejects_nonaccepted_evidence_level(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["evidence_level"] = "smoke"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "evidence_level must be accepted_same_protocol" in report.records[0].issues
    assert (
        "evidence_level must not reference smoke evidence"
        in report.records[0].issues
    )


def test_artifact_gate_requires_explicit_rtx_4090_gpu_model(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["gpu_model"] = "NVIDIA 4090"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "gpu_model must record RTX 4090-class hardware" in report.records[0].issues


def test_artifact_gate_rejects_nonlocal_gpu_model_markers(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["gpu_model"] = "NVIDIA GeForce RTX 4090 + A100 cloud fallback"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "gpu_model must not reference nonlocal accelerator A100"
        in report.records[0].issues
    )
    assert (
        "gpu_model must not reference nonlocal accelerator CLOUD"
        in report.records[0].issues
    )


def test_artifact_gate_rejects_dirty_source_tree_status(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["source_tree_status"] = "dirty"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "source_tree_status must be clean" in report.records[0].issues


def test_artifact_gate_rejects_dirty_sha_provenance_markers(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["git_sha_or_submodule_sha"] = "abc123-dirty"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "git_sha_or_submodule_sha must not contain dirty" in report.records[0].issues


def test_artifact_gate_rejects_smoke_demo_or_pending_protocol_markers(
    tmp_path: Path,
) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["dataset_split"] = "cwru_demo_seed0"
    data["preprocessing_signature"] = "pending-preprocess"
    data["command"] = "CUDA_VISIBLE_DEVICES=0 python scripts/run_smoke.py"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert "dataset_split must not reference demo evidence" in report.records[0].issues
    assert (
        "preprocessing_signature must not reference pending evidence"
        in report.records[0].issues
    )
    assert "command must not reference smoke evidence" in report.records[0].issues


def test_artifact_gate_requires_sha256_preprocessing_signature(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["preprocessing_signature"] = "sha256:accepted-protocol"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "preprocessing_signature must match sha256:<64 lowercase hex>"
        in report.records[0].issues
    )


def test_command_cuda_visible_devices_parser() -> None:
    assert _command_cuda_visible_devices("CUDA_VISIBLE_DEVICES=0 python run.py") == "0"
    assert (
        _command_cuda_visible_devices(
            "cd run && CUDA_VISIBLE_DEVICES=0,1 python run.py"
        )
        == "0,1"
    )
    assert _command_cuda_visible_devices("python run.py") == ""


def test_artifact_gate_rejects_command_device_mismatch(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["command"] = data["command"].replace(
        "CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=1"
    )
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "command CUDA_VISIBLE_DEVICES=1 does not match cuda_visible_devices=0"
        in report.records[0].issues
    )


def test_artifact_gate_allows_top_representative_command_source_without_cuda_prefix(
    tmp_path: Path,
) -> None:
    _write_valid_artifact(tmp_path / "top" / "run0")
    run_meta = tmp_path / "top" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["source_queue_id"] = "TOP-Q1-GTM"
    data["phase"] = "top_representatives"
    data["entry_id"] = "B04,B05,A04"
    data["cuda_visible_devices"] = "0,1"
    data["gpu_count"] = 2
    data["command"] = "paper-local baseline_ablation_matrix.yaml entries B04/B05/A04"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is True
    assert report.records[0].issues == ()


def test_artifact_gate_rejects_empty_metrics_json(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    (tmp_path / "paper07" / "run0" / "metrics.json").write_text(
        "{}\n", encoding="utf-8"
    )

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "metrics_path JSON must contain at least one metric" in report.records[0].issues
    )


def test_artifact_gate_rejects_json_without_numeric_metrics(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    (tmp_path / "paper07" / "run0" / "metrics.json").write_text(
        '{"status": "ok", "notes": ["accepted protocol"]}\n',
        encoding="utf-8",
    )

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "metrics_path JSON must contain at least one numeric metric"
        in report.records[0].issues
    )


def test_artifact_gate_accepts_metrics_csv_with_data_row(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_dir = tmp_path / "paper07" / "run0"
    (run_dir / "metrics.csv").write_text(
        "metric,value\naccuracy,1.0\n", encoding="utf-8"
    )
    run_meta = run_dir / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["metrics_path"] = "metrics.csv"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is True
    assert report.records[0].issues == ()


def test_artifact_gate_rejects_csv_without_numeric_metrics(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_dir = tmp_path / "paper07" / "run0"
    (run_dir / "metrics.csv").write_text("metric,value\nstatus,ok\n", encoding="utf-8")
    run_meta = run_dir / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["metrics_path"] = "metrics.csv"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "metrics_path CSV must contain at least one numeric metric"
        in report.records[0].issues
    )


def test_artifact_gate_rejects_absolute_referenced_paths(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    run_dir = tmp_path / "paper07" / "run0"
    run_meta = run_dir / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["metrics_path"] = str((run_dir / "metrics.json").resolve())
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "metrics_path must be relative to the run_meta.yaml directory"
        in report.records[0].issues
    )


def test_artifact_gate_rejects_referenced_paths_outside_run_dir(tmp_path: Path) -> None:
    _write_valid_artifact(tmp_path / "paper07" / "run0")
    shared_metrics = tmp_path / "paper07" / "metrics.json"
    shared_metrics.write_text('{"accuracy": 1.0}\n', encoding="utf-8")
    run_meta = tmp_path / "paper07" / "run0" / "run_meta.yaml"
    data = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    data["metrics_path"] = "../metrics.json"
    run_meta.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_artifact_gate(tmp_path)

    assert report.accepted is False
    assert report.records[0].accepted is False
    assert (
        "metrics_path must stay inside the run_meta.yaml directory"
        in report.records[0].issues
    )


def test_artifact_gate_cli_writes_json_and_preserves_blocked_exit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gate" / "artifact_gate.json"

    assert (
        main([str(tmp_path / "empty"), "--format", "json", "--output", str(output)])
        == 2
    )

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
    assert {"source_queue_id", "paper_id", "phase", "entry_id"} <= artifact_fields

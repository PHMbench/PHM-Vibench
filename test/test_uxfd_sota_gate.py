import json
from pathlib import Path

import yaml

from scripts.uxfd_sota_gate import (
    DEFAULT_SOTA_ROOT,
    build_payload,
    evaluate_sota_gate,
    main,
    render_markdown,
)


PERSISTED_SOTA_GATE_JSON = Path("paper/UXFD_paper/results/sota_gate_current.json")
PERSISTED_SOTA_GATE_MD = Path("paper/UXFD_paper/results/sota_gate_current.md")


def _statistics() -> dict:
    return {
        "mean": 0.95,
        "std": 0.01,
        "ci95_low": 0.94,
        "ci95_high": 0.96,
    }


def _write_minimal_queue(tmp_path: Path) -> tuple[Path, Path]:
    matrix_path = tmp_path / "paper" / "submission_prep" / "baseline_ablation_matrix.yaml"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "paper_id": "ExamplePaper",
                "baselines": [{"id": f"B{index:02d}"} for index in range(1, 7)],
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
                        "queue_id": "QX",
                        "paper_id": "ExamplePaper",
                        "matrix_path": str(matrix_path),
                        "minimum_seeds": 3,
                    }
                ],
                "top_representative_bindings": [
                    {
                        "binding_id": "TOP-QX",
                        "paper_id": "ExamplePaper",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return queue_path, matrix_path


def _write_valid_aggregate(root: Path) -> Path:
    aggregate_path = root / "ExamplePaper" / "sota_aggregate.yaml"
    aggregate_path.parent.mkdir(parents=True)
    seeds = [0, 1, 2]
    aggregate_path.write_text(
        yaml.safe_dump(
            {
                "paper_id": "ExamplePaper",
                "accepted_sota_evidence": True,
                "claim_scope": "representative_only",
                "primary_metric": "accuracy",
                "proposed": {
                    "entry_id": "P00",
                    "seed_values": seeds,
                    "statistics": _statistics(),
                },
                "comparators": [
                    {
                        "entry_id": f"B{index:02d}",
                        "role": "baseline",
                        "seed_values": seeds,
                        "statistics": _statistics(),
                        "effect_size_vs_proposed": 0.5,
                    }
                    for index in range(1, 7)
                ],
                "top_representatives": [
                    {
                        "binding_id": "TOP-QX",
                        "scope": "representative",
                        "seed_values": seeds,
                        "statistics": _statistics(),
                        "paired_test": {"name": "wilcoxon", "p_value": 0.01},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return aggregate_path


def test_sota_gate_blocks_missing_current_aggregates() -> None:
    report = evaluate_sota_gate()

    assert report.ready is False
    assert report.aggregate_root == str(DEFAULT_SOTA_ROOT)
    assert report.expected_papers == 7
    assert report.accepted_papers == 0
    assert len(report.records) == 7
    assert any("sota aggregate root does not exist" in item for item in report.blockers)
    assert all(record.accepted is False for record in report.records)
    assert all("missing sota_aggregate.yaml" in record.issues for record in report.records)


def test_sota_gate_accepts_complete_matched_seed_aggregate(tmp_path: Path) -> None:
    queue_path, _ = _write_minimal_queue(tmp_path)
    aggregate_root = tmp_path / "sota_aggregates"
    _write_valid_aggregate(aggregate_root)

    report = evaluate_sota_gate(aggregate_root, queue_path=queue_path)

    assert report.ready is True
    assert report.expected_papers == 1
    assert report.accepted_papers == 1
    assert report.blockers == ()
    assert report.records[0].accepted is True


def test_sota_gate_rejects_single_seed_or_missing_statistics(tmp_path: Path) -> None:
    queue_path, _ = _write_minimal_queue(tmp_path)
    aggregate_root = tmp_path / "sota_aggregates"
    aggregate_path = _write_valid_aggregate(aggregate_root)
    data = yaml.safe_load(aggregate_path.read_text(encoding="utf-8"))
    data["proposed"]["seed_values"] = [0]
    data["comparators"][0]["statistics"].pop("ci95_high")
    data["comparators"][1].pop("effect_size_vs_proposed")
    aggregate_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = evaluate_sota_gate(aggregate_root, queue_path=queue_path)

    assert report.ready is False
    issues = "\n".join(report.records[0].issues)
    assert "proposed.seed_values must include at least 3 seeds" in issues
    assert "comparators[B01].statistics.ci95_high must be numeric" in issues
    assert "comparators[B02] must include numeric effect_size_vs_proposed" in issues


def test_persisted_sota_gate_reports_match_current_gate() -> None:
    report = evaluate_sota_gate()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_SOTA_GATE_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_SOTA_GATE_MD.read_text(encoding="utf-8") == render_markdown(report)


def test_sota_gate_cli_writes_blocking_json_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "sota" / "gate.json"

    assert main(["--format", "json", "--output", str(output)]) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["accepted_papers"] == 0

    markdown = tmp_path / "sota" / "gate.md"
    assert main(["--format", "markdown", "--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "UXFD SOTA Gate" in text
    assert "Accepted papers: `0/7`" in text

    assert (
        main(
            [
                "--format",
                "json",
                "--output",
                str(output),
                "--allow-not-ready",
            ]
        )
        == 0
    )

import json
from pathlib import Path

from scripts.uxfd_submission_gate import evaluate_submission_gate, main


def test_submission_gate_reports_all_papers_not_ready() -> None:
    report = evaluate_submission_gate()

    assert report.ready is False
    assert report.queue_can_execute is False
    assert report.artifact_gate_accepted is False
    assert report.artifact_gate_records == 0
    assert len(report.papers) == 7
    assert report.queue_summary["total"] == 104
    assert all(paper.baselines >= 6 for paper in report.papers)
    assert all(paper.ablations >= 6 for paper in report.papers)
    assert all(paper.submission_ready is False for paper in report.papers)
    assert any("submission_ready is false" in item for item in report.blockers)
    assert any("gpu queue blocked" in item for item in report.blockers)
    assert any("artifact gate blocked" in item for item in report.blockers)
    assert len(report.next_actions) == 7
    assert {action["paper_id"] for action in report.next_actions} == {
        paper.paper_id for paper in report.papers
    }
    assert all(action["unblock_condition"] for action in report.next_actions)
    assert report.next_actions[0]["queue_id"] == "Q1"
    checklist = {item["requirement"]: item for item in report.objective_checklist}
    assert checklist["goal file 00_overall_goal.md"]["status"] == "met"
    assert checklist["goal file 08_recent_work_citation_readme.md"]["status"] == "met"
    assert checklist["goal file 09_gpu_execution_queue.yaml"]["status"] == "met"
    assert checklist["Claude Code Team task spec"]["status"] == "met"
    assert checklist["Claude Code Team launch/block log"]["status"] == "met"
    assert checklist["Codex xhigh subagent launch log"]["status"] == "met"
    assert checklist["seven paper-local matrices"]["status"] == "met"
    assert checklist["6+ baselines and 6+ ablations per paper"]["status"] == "met"
    assert checklist["accepted run artifact metadata"]["status"] == "not_met"
    assert checklist["submission readiness achieved"]["status"] == "not_met"


def test_submission_gate_cli_writes_blocking_json_report(tmp_path: Path) -> None:
    output = tmp_path / "gate" / "submission_gate.json"

    assert main(["--format", "json", "--output", str(output)]) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["queue_can_execute"] is False
    assert payload["artifact_gate_accepted"] is False
    assert payload["artifact_gate_records"] == 0
    assert payload["artifact_gate_blockers"]
    assert len(payload["papers"]) == 7
    assert len(payload["next_actions"]) == 7
    assert len(payload["objective_checklist"]) >= 15
    assert payload["queue_summary"]["top_representatives"] == 7

    markdown = tmp_path / "gate" / "submission_gate.md"
    assert main(["--format", "markdown", "--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "Ready: `False`" in text
    assert "Artifact gate accepted: `False`" in text
    assert "## Blockers" in text
    assert "## Next Actions" in text
    assert "## Objective Checklist" in text


def test_submission_gate_cli_can_be_used_as_non_failing_audit(tmp_path: Path) -> None:
    output = tmp_path / "gate" / "audit.json"

    assert (
        main(["--format", "json", "--output", str(output), "--allow-not-ready"])
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False


def test_submission_gate_blocks_artifact_root_without_full_queue_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "accepted" / "paper01" / "run0"
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

    report = evaluate_submission_gate(artifact_root=tmp_path / "accepted")

    assert report.ready is False
    assert report.artifact_gate_accepted is False
    assert report.artifact_gate_records == 1
    assert any("artifact gate blocked" in item for item in report.blockers)
    assert any("queue coverage incomplete" in item for item in report.artifact_gate_blockers)

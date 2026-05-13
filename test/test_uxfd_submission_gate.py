import json
from pathlib import Path

import yaml

from scripts.uxfd_submission_gate import (
    build_payload,
    evaluate_submission_gate,
    main,
    render_markdown,
)


PERSISTED_SUBMISSION_GATE_JSON = Path(
    "paper/UXFD_paper/results/submission_gate_current.json"
)
PERSISTED_SUBMISSION_GATE_MD = Path(
    "paper/UXFD_paper/results/submission_gate_current.md"
)


def test_submission_gate_reports_all_papers_not_ready() -> None:
    report = evaluate_submission_gate()

    assert report.ready is False
    assert report.queue_can_execute is False
    assert report.artifact_gate_accepted is False
    assert report.artifact_gate_records == 0
    assert report.recent_work_policy_ready is True
    assert report.recent_work_evidence_ready is False
    assert report.recent_work_matrix_rows == 7
    assert len(report.recent_work_blockers) == 7
    assert report.low_tier_source_ready is True
    assert report.low_tier_source_blocker_count == 0
    assert report.low_tier_source_triage_count > 0
    assert report.submodule_dirty_clean is False
    assert report.submodule_dirty_entries > 0
    assert report.submodule_dirty_submodules == 3
    assert len(report.papers) == 7
    assert report.queue_summary["total"] == 104
    assert all(paper.baselines >= 6 for paper in report.papers)
    assert all(paper.ablations >= 6 for paper in report.papers)
    assert all(paper.submission_ready is False for paper in report.papers)
    assert any("submission_ready is false" in item for item in report.blockers)
    assert any("gpu queue blocked" in item for item in report.blockers)
    assert any("artifact gate blocked" in item for item in report.blockers)
    assert any("recent-work evidence blocked" in item for item in report.blockers)
    assert any("submodule dirty triage blocked" in item for item in report.blockers)
    assert not any("low-tier source hygiene blocked" in item for item in report.blockers)
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
    assert checklist["GPU launch scripts enforce static queue gate"]["status"] == "met"
    assert checklist["goal clarity audit report"]["status"] == "met"
    assert checklist["commit recovery plan"]["status"] == "met"
    assert checklist["Paper07 rejection-recovery innovation contract"]["status"] == "met"
    assert (
        checklist["TOP recent-work policy and paper-local matrix coverage"]["status"]
        == "met"
    )
    assert checklist["low-tier source hygiene"]["status"] == "met"
    assert (
        checklist["paper submodule working trees clean before handoff"]["status"]
        == "not_met"
    )
    assert checklist["TOP representative accepted artifacts"]["status"] == "not_met"
    assert checklist["accepted run artifact metadata"]["status"] == "not_met"
    assert checklist["submission readiness achieved"]["status"] == "not_met"


def test_persisted_submission_gate_reports_match_current_gate() -> None:
    report = evaluate_submission_gate()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_SUBMISSION_GATE_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_SUBMISSION_GATE_MD.read_text(encoding="utf-8") == render_markdown(
        report
    )


def test_submission_gate_cli_writes_blocking_json_report(tmp_path: Path) -> None:
    output = tmp_path / "gate" / "submission_gate.json"

    assert main(["--format", "json", "--output", str(output)]) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["queue_can_execute"] is False
    assert payload["artifact_gate_accepted"] is False
    assert payload["artifact_gate_records"] == 0
    assert payload["artifact_gate_blockers"]
    assert payload["recent_work_policy_ready"] is True
    assert payload["recent_work_evidence_ready"] is False
    assert payload["recent_work_matrix_rows"] == 7
    assert len(payload["recent_work_blockers"]) == 7
    assert payload["low_tier_source_ready"] is True
    assert payload["low_tier_source_blocker_count"] == 0
    assert payload["low_tier_source_triage_count"] > 0
    assert payload["submodule_dirty_clean"] is False
    assert payload["submodule_dirty_entries"] > 0
    assert payload["submodule_dirty_submodules"] == 3
    assert len(payload["papers"]) == 7
    assert len(payload["next_actions"]) == 7
    assert len(payload["objective_checklist"]) >= 15
    assert payload["queue_summary"]["top_representatives"] == 7

    markdown = tmp_path / "gate" / "submission_gate.md"
    assert main(["--format", "markdown", "--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "Ready: `False`" in text
    assert "Artifact gate accepted: `False`" in text
    assert "Recent-work policy ready: `True`" in text
    assert "Recent-work evidence ready: `False`" in text
    assert "Low-tier source hygiene ready: `True`" in text
    assert "Submodule dirty clean: `False`" in text
    assert "## Blockers" in text
    assert "## Next Actions" in text
    assert "## Objective Checklist" in text
    assert "GPU launch scripts enforce static queue gate" in text


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


def test_submission_gate_blocks_ready_matrix_with_pending_evidence_statuses(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "paper" / "submission_prep" / "baseline_ablation_matrix.yaml"
    matrix_path.parent.mkdir(parents=True)
    matrix = {
        "paper_id": "ExamplePaper",
        "submission_ready": True,
        "strict_blockers": [],
        "proposed": {
            "id": "P00",
            "label": "proposed",
            "command": "CUDA_VISIBLE_DEVICES=0 python main.py --config demo.yaml",
            "accepted_evidence_status": "pending same-protocol GPU run",
        },
        "baselines": [
            {
                "id": f"B{index:02d}",
                "label": f"baseline {index}",
                "command": "CUDA_VISIBLE_DEVICES=0 python main.py --config demo.yaml",
                "accepted_evidence_status": "accepted_gpu_and_artifacts",
            }
            for index in range(1, 7)
        ],
        "ablations": [
            {
                "id": f"A{index:02d}",
                "label": f"ablation {index}",
                "command": "CUDA_VISIBLE_DEVICES=0 python main.py --config demo.yaml",
                "accepted_evidence_status": "accepted_gpu_and_artifacts",
            }
            for index in range(1, 7)
        ],
    }
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        yaml.safe_dump(
            {
                "status": "blocked_resource_preflight",
                "resource_preflight": {
                    "required_devices": ["0", "1"],
                    "required_gpu_class": "RTX 4090",
                    "current_session_result": {
                        "torch_cuda_available": False,
                        "torch_cuda_device_count": 0,
                        "gpu_names": [],
                        "verdict": "blocked in test",
                    },
                },
                "scheduler": {"default_devices": ["0", "1"]},
                "paper_queue": [
                    {
                        "queue_id": "QX",
                        "paper_id": "ExamplePaper",
                        "goal_file": "goal.md",
                        "matrix_path": str(matrix_path),
                        "base_config": "demo.yaml",
                        "priority_reason": "test",
                        "unblock_condition": "test",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_submission_gate(
        queue_path=queue_path,
        artifact_root=tmp_path / "accepted_runs",
    )

    assert report.ready is False
    assert any(
        "submission_ready true but 1 proposed/baseline/ablation evidence entries"
        in blocker
        for blocker in report.blockers
    )

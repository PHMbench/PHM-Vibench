import json
from pathlib import Path

import scripts.uxfd_objective_audit as audit
from scripts.uxfd_objective_audit import (
    ACCEPTED_RUN_ROOT_README,
    PARENT_GOAL_CHECKPOINT_PATHS,
    SOTA_AGGREGATE_TEMPLATE_README,
    build_payload,
    evaluate_objective_audit,
    LATEST_CONTINUATION_HANDOFF_PATH,
    main,
    render_markdown,
)


PERSISTED_OBJECTIVE_AUDIT_JSON = Path(
    "paper/UXFD_paper/results/objective_audit_current.json"
)
PERSISTED_OBJECTIVE_AUDIT_MD = Path(
    "paper/UXFD_paper/results/objective_audit_current.md"
)
SPEC_TASKS = Path("specs/006-uxfd-ieee-trans-submission-readiness/tasks.md")
CLAUDE_TEAM_DIR = Path(".codex/claude-team-runs/20260511-uxfd-ieee-trans-review")
CONTINUATION_HANDOFF = Path(".claude/handoffs/2026-05-12-uxfd-goal-continuation.md")
EXECUTION_GATE_HANDOFF = Path(
    ".claude/handoffs/2026-05-13-uxfd-execution-gate-check.md"
)


def _items_by_requirement(report):
    return {item.requirement: item for item in report.items}


def test_objective_audit_maps_prompt_requirements_to_artifacts() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)

    assert report.achieved is False
    assert report.met > 20
    assert report.not_met > 0
    assert report.blocked >= 0
    assert items["named goal file 00_overall_goal.md"].status == "met"
    assert items["named goal file 08_recent_work_citation_readme.md"].status == "met"
    assert items["named goal file 99_submission_readiness_matrix.md"].status == "met"
    assert items["named goal file README.md"].status == "met"
    assert items["Spec Kit artifact spec.md"].status == "met"
    assert items["Spec Kit artifact plan.md"].status == "met"
    assert items["Spec Kit artifact tasks.md"].status == "met"
    assert items["handoff document"].status == "met"
    assert items["continuation handoff document"].status == "met"
    assert items["execution gate handoff document"].status == "met"
    assert items["latest continuation handoff document"].status == "met"
    assert items["Claude Team task spec"].status == "met"
    assert items["Claude Team launch log"].status == "met"
    assert items["Codex xhigh subagent launch log"].status == "met"
    assert (
        items["six xhigh/subagent or Claude Team execution evidence"].status
        == "met"
    )
    assert (
        items["six xhigh/subagent or Claude Team execution evidence"].details
        == "subagents=6, xhigh=True, deliverables=3"
    )
    assert items["Claude Team deliverable report.md"].status == "met"
    assert items["Claude Team deliverable risks.md"].status == "met"
    assert items["Claude Team deliverable test-log.md"].status == "met"
    assert items["SOTA aggregate gate JSON report"].status == "met"
    assert items["SOTA aggregate gate markdown report"].status == "met"
    assert items["SOTA aggregate template manifest"].status == "met"
    assert items["SOTA aggregate scaffold report"].status == "met"
    assert items["seven paper-local baseline/ablation matrices"].status == "met"
    assert items["TOP recent-work policy"].status == "met"
    assert "source_verification_ready=True" in items["TOP recent-work policy"].details
    assert items["low-tier source audit report"].status == "met"
    assert items["low-tier source hygiene"].status == "met"
    assert items["TOP representative accepted artifacts"].status == "not_met"
    assert items["2x4090 GPU queue executable"].status == "blocked"
    assert items["accepted run artifact metadata"].status == "not_met"
    assert items["cross-paper submission gate"].status == "not_met"
    assert items["submodule dirty triage report"].status == "met"
    assert items["submodule dirty triage JSON report"].status == "met"
    assert items["parent result artifact triage report"].status == "met"
    assert items["GPU launch scripts enforce static queue gate"].status == "met"
    assert "exit 2" in items["GPU launch scripts enforce static queue gate"].details
    assert items["accepted metrics contain numeric values"].status == "met"
    assert "numeric metric" in items["accepted metrics contain numeric values"].details
    assert items["accepted artifacts require clean source trees"].status == "met"
    assert "source_tree_status clean" in items[
        "accepted artifacts require clean source trees"
    ].details
    assert items["accepted artifacts require numeric run controls"].status == "met"
    assert "integer seed and batch_size" in items[
        "accepted artifacts require numeric run controls"
    ].details
    assert "unique queue+seed keys" in items[
        "accepted artifacts require numeric run controls"
    ].details
    assert "minimum_seeds coverage" in items[
        "accepted artifacts require numeric run controls"
    ].details
    assert items["accepted artifacts require positive runtime metadata"].status == "met"
    assert "positive HH:MM:SS runtime" in items[
        "accepted artifacts require positive runtime metadata"
    ].details
    assert (
        items["accepted artifacts require enumerated precision metadata"].status
        == "met"
    )
    assert "precision enum" in items[
        "accepted artifacts require enumerated precision metadata"
    ].details
    assert (
        items["accepted artifacts require accepted_same_protocol evidence level"].status
        == "met"
    )
    assert "smoke/demo/dummy/template/pending" in items[
        "accepted artifacts require accepted_same_protocol evidence level"
    ].details
    assert (
        items["accepted artifacts require hashed preprocessing signatures"].status
        == "met"
    )
    assert "sha256 preprocessing_signature" in items[
        "accepted artifacts require hashed preprocessing signatures"
    ].details
    assert items["accepted artifacts require clean SHA provenance"].status == "met"
    assert "dirty SHA provenance" in items[
        "accepted artifacts require clean SHA provenance"
    ].details
    assert (
        items["accepted-run evidence root requires GPU and queue preflight"].status
        == "met"
    )
    assert "live GPU preflight" in items[
        "accepted-run evidence root requires GPU and queue preflight"
    ].details
    assert (
        items["SOTA aggregate activation requires accepted run coverage"].status
        == "met"
    )
    assert "accepted run_meta refs" in items[
        "SOTA aggregate activation requires accepted run coverage"
    ].details
    assert (
        items[
            "SOTA comparison requires multi-seed same-protocol aggregate evidence"
        ].status
        == "met"
    )
    assert "block single-run SOTA" in items[
        "SOTA comparison requires multi-seed same-protocol aggregate evidence"
    ].details
    assert "accepted run refs" in items[
        "SOTA comparison requires multi-seed same-protocol aggregate evidence"
    ].details
    assert items["readiness execution backlog"].status == "met"
    assert items["goal clarity audit report"].status == "met"
    assert items["commit recovery plan"].status == "met"
    assert items["Paper07 rejection-recovery innovation contract"].status == "met"
    assert (
        items["paper submodule working trees clean before parent handoff"].status
        in {"met", "not_met", "unverified"}
    )
    assert (
        items["parent UXFD goal-control checkpoint committed"].status
        in {"met", "not_met", "unverified"}
    )
    assert any("submission-ready" in item.requirement for item in report.items)
    assert any("IEEE Transactions submission-ready" in item.requirement for item in report.items)


def test_objective_audit_records_each_paper_matrix_as_covered_but_not_ready() -> None:
    report = evaluate_objective_audit()
    paper_matrix_items = [
        item
        for item in report.items
        if item.requirement.endswith("6+ baselines and 6+ ablations")
    ]
    paper_ready_items = [
        item
        for item in report.items
        if item.requirement.endswith("IEEE Transactions submission-ready")
    ]

    assert len(paper_matrix_items) == 7
    assert len(paper_ready_items) == 7
    assert all(item.status == "met" for item in paper_matrix_items)
    assert all(item.status == "not_met" for item in paper_ready_items)
    assert all("submission_ready=False" in item.details for item in paper_matrix_items)
    assert all("strict blockers remaining=" in item.details for item in paper_ready_items)


def test_spec_tasks_match_local_xhigh_subagent_evidence() -> None:
    text = SPEC_TASKS.read_text(encoding="utf-8")
    launch_text = (CLAUDE_TEAM_DIR / "CODEX_SUBAGENT_LAUNCH.md").read_text(
        encoding="utf-8"
    )

    assert "- [x] T026 [US3]" in text
    assert "- [x] T027 [US3]" in text
    assert "six local Codex xhigh read-only subagents" in text
    assert "keep `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` non-ready" in text

    for filename in (
        "CODEX_SUBAGENT_LAUNCH.md",
        "report.md",
        "risks.md",
        "test-log.md",
    ):
        assert (CLAUDE_TEAM_DIR / filename).exists()

    assert "reasoning_effort=xhigh" in launch_text
    assert launch_text.count("read-only audit") == 6
    assert launch_text.count("019e1769-") == 6


def test_objective_audit_covers_latest_continuation_handoff() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)

    assert CONTINUATION_HANDOFF.exists()
    assert items["continuation handoff document"].evidence == str(CONTINUATION_HANDOFF)
    assert items["continuation handoff document"].status == "met"


def test_objective_audit_covers_latest_execution_gate_handoff() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)

    assert EXECUTION_GATE_HANDOFF.exists()
    assert items["execution gate handoff document"].evidence == str(EXECUTION_GATE_HANDOFF)
    assert items["execution gate handoff document"].status == "met"


def test_objective_audit_covers_latest_continuation_handoff() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)

    assert LATEST_CONTINUATION_HANDOFF_PATH.exists()
    assert items["latest continuation handoff document"].evidence == str(
        LATEST_CONTINUATION_HANDOFF_PATH
    )
    assert items["latest continuation handoff document"].status == "met"


def test_persisted_objective_audit_reports_match_current_audit() -> None:
    report = evaluate_objective_audit()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_OBJECTIVE_AUDIT_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_OBJECTIVE_AUDIT_MD.read_text(encoding="utf-8") == render_markdown(
        report
    )


def test_objective_audit_covers_accepted_run_root_activation_gate() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)
    item = items["accepted-run evidence root requires GPU and queue preflight"]

    assert ACCEPTED_RUN_ROOT_README.exists()
    assert str(ACCEPTED_RUN_ROOT_README) in item.evidence
    assert item.status == "met"
    assert "artifact gate queue coverage" in item.details


def test_objective_audit_covers_sota_aggregate_activation_gate() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)
    item = items["SOTA aggregate activation requires accepted run coverage"]

    assert SOTA_AGGREGATE_TEMPLATE_README.exists()
    assert str(SOTA_AGGREGATE_TEMPLATE_README) in item.evidence
    assert item.status == "met"
    assert "artifact gate queue coverage" in item.details


def test_objective_audit_cli_writes_blocking_json_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "objective" / "audit.json"

    assert main(["--format", "json", "--output", str(output)]) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["achieved"] is False
    assert payload["met"] > 20
    assert payload["not_met"] > 0
    assert payload["blocked"] >= 1
    assert any(
        item["requirement"] == "cross-paper submission gate"
        and item["status"] == "not_met"
        for item in payload["items"]
    )

    markdown = tmp_path / "objective" / "audit.md"
    assert main(["--format", "markdown", "--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "Achieved: `False`" in text
    assert "## Prompt-to-Artifact Checklist" in text
    assert "## Blockers" in text


def test_submodule_cleanliness_item_detects_dirty_submodule(monkeypatch) -> None:
    def fake_git_status_lines(path: Path) -> tuple[str, ...]:
        if path.name == "Explainable_FD_Toolkit":
            return (" M VIBENCH.md", "?? scripts/run_toolkit_ablations.py")
        return ()

    monkeypatch.setattr(audit, "_git_status_lines", fake_git_status_lines)

    item = audit._paper_submodule_cleanliness_item()

    assert item.requirement == "paper submodule working trees clean before parent handoff"
    assert item.status == "not_met"
    assert "Explainable_FD_Toolkit:2" in item.details


def test_submodule_cleanliness_item_accepts_clean_submodules(monkeypatch) -> None:
    monkeypatch.setattr(audit, "_git_status_lines", lambda path: ())

    item = audit._paper_submodule_cleanliness_item()

    assert item.status == "met"
    assert "7 paper submodules clean" in item.details


def test_parent_goal_checkpoint_item_detects_dirty_parent_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        audit,
        "_git_status_lines_for_paths",
        lambda paths: (" M paper/UXFD_paper/goal/README.md", "?? test/test_uxfd_goal_clarity.py"),
    )

    item = audit._parent_goal_checkpoint_item()

    assert item.requirement == "parent UXFD goal-control checkpoint committed"
    assert item.status == "not_met"
    assert "dirty_parent_goal_control_paths=2" in item.details


def test_parent_goal_checkpoint_item_accepts_clean_parent_paths(monkeypatch) -> None:
    monkeypatch.setattr(audit, "_git_status_lines_for_paths", lambda paths: ())

    item = audit._parent_goal_checkpoint_item()

    assert item.status == "met"
    assert "parent goal-control paths clean" in item.details


def test_parent_goal_checkpoint_paths_exclude_self_updating_outputs() -> None:
    paths = {str(path) for path in PARENT_GOAL_CHECKPOINT_PATHS}

    assert str(PERSISTED_OBJECTIVE_AUDIT_JSON) not in paths
    assert str(PERSISTED_OBJECTIVE_AUDIT_MD) not in paths
    assert "paper/UXFD_paper/results/readiness_backlog.md" not in paths
    assert str(EXECUTION_GATE_HANDOFF) in paths
    assert str(LATEST_CONTINUATION_HANDOFF_PATH) in paths
    assert "paper/UXFD_paper/goal/09_gpu_execution_queue.yaml" in paths
    assert "paper/UXFD_paper/goal/status" in paths
    assert "paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md" in paths
    assert "paper/UXFD_paper/results/submodule_dirty_triage.md" in paths
    assert "paper/UXFD_paper/results/submodule_dirty_triage.json" in paths
    assert "paper/UXFD_paper/results/recent_work_gate_current.json" in paths
    assert "paper/UXFD_paper/results/recent_work_gate_current.md" in paths
    assert "paper/UXFD_paper/results/sota_gate_current.json" in paths
    assert "paper/UXFD_paper/results/sota_gate_current.md" in paths
    assert "paper/UXFD_paper/results/queue_launch_shards/gpu0.sh" in paths
    assert "paper/UXFD_paper/results/queue_launch_shards/gpu1.sh" in paths
    assert "paper/UXFD_paper/results/parent_result_artifact_triage.md" in paths
    assert "paper/UXFD_paper/results/accepted_runs" in paths
    assert "paper/UXFD_paper/results/accepted_run_templates" in paths
    assert "paper/UXFD_paper/results/sota_aggregate_templates" in paths
    assert "paper/UXFD_paper/results/.gitignore" in paths
    assert "paper/UXFD_paper/results/low_tier_source_audit.json" in paths
    assert "scripts/uxfd_artifact_gate.py" in paths
    assert "scripts/uxfd_artifact_scaffold.py" in paths
    assert "scripts/uxfd_sota_scaffold.py" in paths
    assert "scripts/uxfd_goal_status.py" in paths
    assert "scripts/uxfd_gpu_queue.py" in paths
    assert "scripts/uxfd_low_tier_source_audit.py" in paths
    assert "scripts/uxfd_parent_result_artifact_triage.py" in paths
    assert "scripts/uxfd_readiness_backlog.py" in paths
    assert "scripts/uxfd_recent_work_gate.py" in paths
    assert "scripts/uxfd_sota_gate.py" in paths
    assert "scripts/uxfd_submodule_dirty_triage.py" in paths
    assert "test/test_uxfd_low_tier_source_audit.py" in paths
    assert "test/test_uxfd_parent_result_artifact_triage.py" in paths
    assert "test/test_uxfd_paper01_control_docs.py" in paths
    assert "test/test_uxfd_artifact_gate.py" in paths
    assert "test/test_uxfd_artifact_scaffold.py" in paths
    assert "test/test_uxfd_sota_scaffold.py" in paths
    assert "test/test_uxfd_gpu_queue.py" in paths
    assert "test/test_uxfd_paper02_control_docs.py" in paths
    assert "test/test_uxfd_paper02_runner_policy.py" in paths
    assert "test/test_uxfd_paper04_control_docs.py" in paths
    assert "test/test_uxfd_paper04_runner_policy.py" in paths
    assert "test/test_uxfd_paper04_truth_manuscript.py" in paths
    assert "test/test_uxfd_readiness_backlog.py" in paths
    assert "test/test_uxfd_recent_work_gate.py" in paths
    assert "test/test_uxfd_sota_gate.py" in paths
    assert "test/test_uxfd_submodule_dirty_triage.py" in paths
    assert "test/test_uxfd_goal_status.py" in paths


def test_objective_audit_cli_can_be_used_as_non_failing_audit(tmp_path: Path) -> None:
    output = tmp_path / "objective" / "audit.json"

    assert (
        main(["--format", "json", "--output", str(output), "--allow-not-achieved"])
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["achieved"] is False

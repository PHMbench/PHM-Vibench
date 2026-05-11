import json
from pathlib import Path

import scripts.uxfd_objective_audit as audit
from scripts.uxfd_objective_audit import (
    PARENT_GOAL_CHECKPOINT_PATHS,
    build_payload,
    evaluate_objective_audit,
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
    assert items["Claude Team task spec"].status == "met"
    assert items["Claude Team launch log"].status == "met"
    assert items["Codex xhigh subagent launch log"].status == "met"
    assert (
        items["six xhigh/subagent or Claude Team execution evidence"].status
        == "met"
    )
    assert items["Claude Team deliverable report.md"].status == "met"
    assert items["Claude Team deliverable risks.md"].status == "met"
    assert items["Claude Team deliverable test-log.md"].status == "met"
    assert items["seven paper-local baseline/ablation matrices"].status == "met"
    assert items["TOP recent-work policy"].status == "met"
    assert items["low-tier source audit report"].status == "met"
    assert items["low-tier source hygiene"].status == "met"
    assert items["TOP representative accepted artifacts"].status == "not_met"
    assert items["2x4090 GPU queue executable"].status == "blocked"
    assert items["accepted run artifact metadata"].status == "not_met"
    assert items["cross-paper submission gate"].status == "not_met"
    assert items["submodule dirty triage report"].status == "met"
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


def test_objective_audit_covers_latest_continuation_handoff() -> None:
    report = evaluate_objective_audit()
    items = _items_by_requirement(report)

    assert CONTINUATION_HANDOFF.exists()
    assert items["continuation handoff document"].evidence == str(CONTINUATION_HANDOFF)
    assert items["continuation handoff document"].status == "met"


def test_persisted_objective_audit_reports_match_current_audit() -> None:
    report = evaluate_objective_audit()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_OBJECTIVE_AUDIT_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_OBJECTIVE_AUDIT_MD.read_text(encoding="utf-8") == render_markdown(
        report
    )


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
    assert "paper/UXFD_paper/results/submodule_dirty_triage.md" in paths
    assert "paper/UXFD_paper/results/low_tier_source_audit.json" in paths
    assert "scripts/uxfd_low_tier_source_audit.py" in paths
    assert "scripts/uxfd_readiness_backlog.py" in paths
    assert "scripts/uxfd_submodule_dirty_triage.py" in paths
    assert "test/test_uxfd_low_tier_source_audit.py" in paths
    assert "test/test_uxfd_paper01_control_docs.py" in paths
    assert "test/test_uxfd_artifact_gate.py" in paths
    assert "test/test_uxfd_gpu_queue.py" in paths
    assert "test/test_uxfd_paper02_runner_policy.py" in paths
    assert "test/test_uxfd_paper04_runner_policy.py" in paths
    assert "test/test_uxfd_readiness_backlog.py" in paths
    assert "test/test_uxfd_submodule_dirty_triage.py" in paths


def test_objective_audit_cli_can_be_used_as_non_failing_audit(tmp_path: Path) -> None:
    output = tmp_path / "objective" / "audit.json"

    assert (
        main(["--format", "json", "--output", str(output), "--allow-not-achieved"])
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["achieved"] is False

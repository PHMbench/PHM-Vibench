import json
from pathlib import Path

import scripts.uxfd_objective_audit as audit
from scripts.uxfd_objective_audit import evaluate_objective_audit, main


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
    assert items["TOP representative accepted artifacts"].status == "not_met"
    assert items["2x4090 GPU queue executable"].status == "blocked"
    assert items["accepted run artifact metadata"].status == "not_met"
    assert items["cross-paper submission gate"].status == "not_met"
    assert (
        items["paper submodule working trees clean before parent handoff"].status
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


def test_objective_audit_cli_can_be_used_as_non_failing_audit(tmp_path: Path) -> None:
    output = tmp_path / "objective" / "audit.json"

    assert (
        main(["--format", "json", "--output", str(output), "--allow-not-achieved"])
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["achieved"] is False

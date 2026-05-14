import json
from pathlib import Path

from scripts.uxfd_owner_review_gate import (
    DEFAULT_DECISION_FILE,
    PENDING_DECISION,
    build_payload,
    evaluate_owner_review_gate,
    main,
    render_markdown,
)
from scripts.uxfd_submodule_dirty_triage import (
    OWNER_REVIEW_ACTION_PACKET,
    OWNER_REVIEW_DECISION_TEMPLATE,
    OWNER_REVIEW_EVIDENCE_INDEX,
    OWNER_REVIEW_RECOMMENDATIONS,
)


PERSISTED_OWNER_REVIEW_GATE_JSON = Path(
    "paper/UXFD_paper/results/submodule_owner_review_gate_current.json"
)
PERSISTED_OWNER_REVIEW_GATE_MD = Path(
    "paper/UXFD_paper/results/submodule_owner_review_gate_current.md"
)


def _approved_decisions_file(tmp_path: Path) -> Path:
    payload = json.loads(OWNER_REVIEW_DECISION_TEMPLATE.read_text(encoding="utf-8"))
    payload["status"] = "owner_review_decisions"
    for record in payload["records"]:
        record["decision"] = record["recommended_decisions"][0]
        record["reviewer"] = "Liqi Thu"
        record["review_date"] = "2026-05-14"
    path = tmp_path / "submodule_owner_review_decisions.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_owner_review_gate_blocks_template_as_approval() -> None:
    report = evaluate_owner_review_gate()

    assert report.ready is False
    assert report.decision_file == str(DEFAULT_DECISION_FILE)
    assert report.template_file == str(OWNER_REVIEW_DECISION_TEMPLATE)
    assert report.source_is_template is True
    assert report.expected_records == 6
    assert len(report.records) == 6
    assert report.pending_records == 6
    assert report.approved_records == 0
    assert all(record.decision == PENDING_DECISION for record in report.records)
    assert any("owner decision file missing" in blocker for blocker in report.blockers)
    assert "template file is not owner approval" in report.blockers


def test_owner_review_template_instructs_status_change_for_real_decisions() -> None:
    payload = json.loads(OWNER_REVIEW_DECISION_TEMPLATE.read_text(encoding="utf-8"))
    instructions = " ".join(payload["instructions"])

    assert payload["status"] == "template_only_not_owner_approved"
    assert payload["supporting_files"] == {
        "action_packet": str(OWNER_REVIEW_ACTION_PACKET),
        "recommendations": str(OWNER_REVIEW_RECOMMENDATIONS),
        "evidence_index": str(OWNER_REVIEW_EVIDENCE_INDEX),
    }
    assert "supporting_files.evidence_index" in instructions
    assert "decision_id unchanged" in instructions
    assert "status to owner_review_decisions" in instructions
    assert "template_only_not_owner_approved status is rejected" in instructions


def test_owner_review_gate_accepts_complete_owner_decisions(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is True
    assert report.source_is_template is False
    assert report.pending_records == 0
    assert report.approved_records == 6
    assert report.blockers == ()
    assert all(not record.issues for record in report.records)


def test_owner_review_gate_rejects_decision_file_with_template_status(
    tmp_path: Path,
) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["status"] = "template_only_not_owner_approved"
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert "owner decision file must be marked owner_review_decisions" in report.blockers


def test_owner_review_gate_rejects_missing_supporting_files(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    del payload["supporting_files"]
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert "supporting_files does not match owner-review support policy" in report.blockers


def test_owner_review_gate_rejects_placeholder_reviewer(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"][0]["reviewer"] = "paper-owner"
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert "approved decision requires a non-TODO reviewer" in report.records[0].issues


def test_owner_review_gate_rejects_non_iso_review_date(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"][0]["review_date"] = "2026/05/14"
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert "approved decision requires an ISO YYYY-MM-DD review_date" in report.records[0].issues


def test_owner_review_gate_rejects_future_review_date(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"][0]["review_date"] = "2999-01-01"
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert "approved decision review_date cannot be in the future" in report.records[0].issues


def test_owner_review_gate_rejects_missing_current_owner_entry(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"] = payload["records"][:-1]
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert any("decision records missing current owner-review entries" in item for item in report.blockers)


def test_owner_review_gate_rejects_stale_dirty_triage_metadata(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"][0]["current_status"] = "M"
    payload["records"][0]["category"] = "experiment_output"
    payload["records"][0]["risk_markers"] = ["stale_exec_root"]
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert (
        "current_status does not match current dirty triage"
        in report.records[0].issues[0]
    )
    assert any(
        "category does not match current dirty triage" in issue
        for issue in report.records[0].issues
    )
    assert "risk_markers do not match current dirty triage" in report.records[0].issues


def test_owner_review_gate_rejects_stale_decision_id(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"][0]["decision_id"] = "OR-99"
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert (
        "decision_id does not match current owner-review queue"
        in report.records[0].issues[0]
    )


def test_owner_review_gate_rejects_duplicate_decision_id(tmp_path: Path) -> None:
    decision_file = _approved_decisions_file(tmp_path)
    payload = json.loads(decision_file.read_text(encoding="utf-8"))
    payload["records"][1]["decision_id"] = payload["records"][0]["decision_id"]
    decision_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = evaluate_owner_review_gate(decision_file=decision_file)

    assert report.ready is False
    assert any(
        "decision records contain duplicate decision_id values" in item
        for item in report.blockers
    )


def test_owner_review_gate_cli_writes_reports(tmp_path: Path) -> None:
    markdown = tmp_path / "owner_review_gate.md"
    json_path = tmp_path / "owner_review_gate.json"

    assert main(["--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "UXFD Owner Review Gate" in text
    assert "Ready: `False`" in text
    assert "template file is not owner approval" in text

    assert main(["--format", "json", "--output", str(json_path), "--allow-not-ready"]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(build_payload(evaluate_owner_review_gate())))
    assert payload == expected
    assert payload["ready"] is False
    assert payload["pending_records"] == 6


def test_owner_review_gate_markdown_contains_record_table() -> None:
    report = evaluate_owner_review_gate()
    text = render_markdown(report)

    assert "## Owner Decision Workflow" in text
    assert "This gate cannot approve the template by itself" in text
    assert str(OWNER_REVIEW_ACTION_PACKET) in text
    assert "submodule_owner_review_recommendations.md" in text
    assert str(OWNER_REVIEW_EVIDENCE_INDEX) in text
    assert str(DEFAULT_DECISION_FILE) in text
    assert "owner_review_decisions" in text
    assert "commit_after_review" in text
    assert "rewrite_then_commit" in text
    assert "discard_from_submodule" in text
    assert "YYYY-MM-DD" in text
    assert "## Records" in text
    assert "| ID | Submodule | Path | Decision | Reviewer | Review date | Issues |" in text
    assert "pending_owner_review" in text


def test_persisted_owner_review_gate_reports_match_current_gate() -> None:
    report = evaluate_owner_review_gate()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_OWNER_REVIEW_GATE_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_OWNER_REVIEW_GATE_MD.read_text(encoding="utf-8") == render_markdown(
        report
    )

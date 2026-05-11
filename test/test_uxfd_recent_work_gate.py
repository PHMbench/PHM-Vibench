import json
from pathlib import Path

from scripts.uxfd_recent_work_gate import (
    LOW_TIER_MARKERS,
    evaluate_recent_work_gate,
    main,
)


def test_recent_work_gate_policy_ready_but_artifact_evidence_pending() -> None:
    report = evaluate_recent_work_gate()

    assert report.ready is False
    assert report.policy_ready is True
    assert report.evidence_ready is False
    assert report.accepted_pool_rows >= 10
    assert len(report.top_2026_ids) >= 6
    assert not report.low_tier_violations
    assert len(report.per_paper_coverage) == 7
    assert all(item.top_count >= 3 for item in report.per_paper_coverage)
    assert all(item.has_2026 for item in report.per_paper_coverage)
    assert all(item.policy_ready for item in report.per_paper_coverage)
    assert len(report.matrix_coverage) == 7
    assert all(item.top_count >= 3 for item in report.matrix_coverage)
    assert all(item.has_2026 for item in report.matrix_coverage)
    assert all(not item.unknown_ids for item in report.matrix_coverage)
    assert all(item.policy_ready for item in report.matrix_coverage)
    assert len(report.bindings) == 7
    assert all(binding.external_work_id.startswith("RWTOP2026-") for binding in report.bindings)
    assert "RWTOP2026-CALTSFM" not in {
        binding.external_work_id for binding in report.bindings
    }
    assert all(binding.status == "pending_gpu_and_artifacts" for binding in report.bindings)
    assert all(binding.representative_only for binding in report.bindings)
    assert all(not binding.evidence_ready for binding in report.bindings)
    assert not report.policy_blockers
    assert len(report.evidence_blockers) == 7


def test_recent_work_gate_low_tier_markers_are_not_in_accepted_pool() -> None:
    report = evaluate_recent_work_gate()

    assert not report.low_tier_violations
    assert all(marker not in report.accepted_pool_ids for marker in LOW_TIER_MARKERS)
    assert "RWTOP2026-TIMESEG" in report.accepted_pool_ids
    assert "RWTOP2026-GTM" in report.accepted_pool_ids
    assert "RWTOP2026-TSPULSE" in report.accepted_pool_ids


def test_recent_work_gate_cli_writes_blocking_json_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "recent" / "recent_work_gate.json"

    assert main(["--format", "json", "--output", str(output)]) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["policy_ready"] is True
    assert payload["evidence_ready"] is False
    assert payload["accepted_pool_rows"] >= 10
    assert len(payload["per_paper_coverage"]) == 7
    assert len(payload["matrix_coverage"]) == 7
    assert len(payload["bindings"]) == 7
    assert len(payload["evidence_blockers"]) == 7

    markdown = tmp_path / "recent" / "recent_work_gate.md"
    assert main(["--format", "markdown", "--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "Ready: `False`" in text
    assert "Policy ready: `True`" in text
    assert "Evidence ready: `False`" in text
    assert "## Paper-Local Matrix Coverage" in text
    assert "## TOP Representative Bindings" in text
    assert "## Blockers" in text


def test_recent_work_gate_cli_can_be_used_as_non_failing_audit(tmp_path: Path) -> None:
    output = tmp_path / "recent" / "audit.json"

    assert (
        main(["--format", "json", "--output", str(output), "--allow-not-ready"])
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["policy_ready"] is True

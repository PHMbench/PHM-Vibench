import json
from pathlib import Path

from scripts.uxfd_recent_work_gate import (
    DEFAULT_RECENT_WORK_README,
    LOW_TIER_MARKERS,
    build_payload,
    evaluate_recent_work_gate,
    main,
    render_markdown,
)


PERSISTED_RECENT_WORK_GATE_JSON = Path(
    "paper/UXFD_paper/results/recent_work_gate_current.json"
)
PERSISTED_RECENT_WORK_GATE_MD = Path(
    "paper/UXFD_paper/results/recent_work_gate_current.md"
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
    assert all(binding.local_proxy_matrix_entries for binding in report.bindings)
    assert any(
        binding.binding_id == "TOP-Q7-TIMESEG"
        and binding.local_proxy_matrix_entries == ("B02", "A05", "A07")
        for binding in report.bindings
    )
    assert not report.policy_blockers
    assert len(report.evidence_blockers) == 7


def test_recent_work_gate_low_tier_markers_are_not_in_accepted_pool() -> None:
    report = evaluate_recent_work_gate()

    assert not report.low_tier_violations
    assert all(marker not in report.accepted_pool_ids for marker in LOW_TIER_MARKERS)
    assert "RWTOP2026-TIMESEG" in report.accepted_pool_ids
    assert "RWTOP2026-GTM" in report.accepted_pool_ids
    assert "RWTOP2026-TSPULSE" in report.accepted_pool_ids


def test_recent_work_gate_rejects_accepted_pool_rows_outside_2024_2026(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "recent_work.md"
    text = DEFAULT_RECENT_WORK_README.read_text(encoding="utf-8")
    readme.write_text(text.replace("| RWTOP2024-TIMEXPP | 2024 |", "| RWTOP2024-TIMEXPP | 2023 |"), encoding="utf-8")

    report = evaluate_recent_work_gate(recent_work_readme=readme)

    assert report.policy_ready is False
    assert any("outside 2024-2026" in blocker for blocker in report.policy_blockers)


def test_recent_work_gate_rejects_non_top_accepted_pool_venue_tier(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "recent_work.md"
    text = DEFAULT_RECENT_WORK_README.read_text(encoding="utf-8")
    readme.write_text(text.replace("| RWTOP2024-TIMEXPP | 2024 | `top-conference` |", "| RWTOP2024-TIMEXPP | 2024 | `application-only` |"), encoding="utf-8")

    report = evaluate_recent_work_gate(recent_work_readme=readme)

    assert report.policy_ready is False
    assert any("venue tier" in blocker for blocker in report.policy_blockers)


def test_persisted_recent_work_gate_reports_match_current_gate() -> None:
    report = evaluate_recent_work_gate()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_RECENT_WORK_GATE_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_RECENT_WORK_GATE_MD.read_text(encoding="utf-8") == render_markdown(report)


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
    assert "Local Proxy Entries" in text
    assert "`B02, A05, A07`" in text
    assert "representative only" in text
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

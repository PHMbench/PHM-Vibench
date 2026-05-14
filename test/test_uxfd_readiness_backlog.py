import json
from pathlib import Path

from scripts.uxfd_readiness_backlog import (
    ACCEPTED_RUN_ARTIFACT_ACTION_PACKET,
    GPU_EXECUTION_RUNBOOK,
    GPU_LIVE_PREFLIGHT,
    GPU_PREFLIGHT_ACTION_PACKET,
    OWNER_REVIEW_EVIDENCE_INDEX,
    PRELAUNCH_GATE_JSON,
    PRELAUNCH_GATE_MARKDOWN,
    evaluate_readiness_backlog,
    main,
    render_markdown,
)
from scripts.uxfd_owner_review_gate import APPROVED_DECISION_STATUS, DEFAULT_DECISION_FILE
from scripts.uxfd_submodule_dirty_triage import (
    OWNER_REVIEW_ACTION_PACKET,
    OWNER_REVIEW_DECISION_TEMPLATE,
    OWNER_REVIEW_RECOMMENDATIONS,
)


PERSISTED_READINESS_BACKLOG_MD = Path("paper/UXFD_paper/results/readiness_backlog.md")


def test_readiness_backlog_prioritizes_gpu_and_paper07() -> None:
    report = evaluate_readiness_backlog()

    assert report.ready is False
    assert report.open_items > 10
    assert report.items[0].item_id == "Q0-PRELAUNCH-GATE"
    assert report.items[0].category == "prelaunch-gate"
    assert "python -m scripts.uxfd_prelaunch_gate --format markdown" in report.items[0].next_action
    assert str(PRELAUNCH_GATE_MARKDOWN) in report.items[0].evidence
    assert str(PRELAUNCH_GATE_JSON) in report.items[0].evidence
    assert report.items[1].item_id == "Q0-GPU-PREFLIGHT"
    assert report.items[2].item_id == "Q0-ARTIFACT-COVERAGE"
    assert report.items[3].item_id == "Q0-SOTA-AGGREGATE"
    assert str(GPU_PREFLIGHT_ACTION_PACKET) in report.items[1].next_action
    assert str(GPU_PREFLIGHT_ACTION_PACKET) in report.items[1].evidence
    assert str(GPU_EXECUTION_RUNBOOK) in report.items[1].evidence
    assert str(GPU_LIVE_PREFLIGHT) in report.items[1].evidence
    assert str(ACCEPTED_RUN_ARTIFACT_ACTION_PACKET) in report.items[2].next_action
    assert str(ACCEPTED_RUN_ARTIFACT_ACTION_PACKET) in report.items[2].evidence
    assert "integer seed/batch_size" in report.items[2].next_action
    assert "positive runtime" in report.items[2].next_action
    assert "enumerated precision" in report.items[2].next_action
    assert "accepted_same_protocol evidence_level" in report.items[2].next_action
    assert "hashed preprocessing_signature" in report.items[2].next_action
    assert "numeric metrics" in report.items[2].next_action
    assert "`source_tree_status: clean`" in report.items[2].next_action
    assert "clean SHA provenance" in report.items[2].next_action
    assert "paper-specific `minimum_seeds` distinct accepted seeds" in report.items[2].next_action
    assert "matched-seed aggregate statistics" in report.items[2].next_action
    assert "sota_aggregate.yaml" in report.items[3].next_action
    assert "mean/std/95% CI" in report.items[3].next_action
    assert "`accepted_run_refs`" in report.items[3].next_action
    assert "`run_meta.yaml`" in report.items[3].next_action
    assert any(item.scope == "TII_operator_attention" for item in report.items[:10])
    assert not any(item.item_id == "Q0-PAPER02-PLANNING-COMMIT" for item in report.items)
    parent_checkpoint_items = [
        item for item in report.items if item.item_id == "Q0-PARENT-GOAL-CHECKPOINT-COMMIT"
    ]
    assert len(parent_checkpoint_items) <= 1
    assert all(item.category != "commit-recovery" for item in report.items if item not in parent_checkpoint_items)
    assert all(item.category != "low-tier-source-hygiene" for item in report.items)
    assert any(item.category == "submodule-dirty-review" for item in report.items)
    dirty_items = [item for item in report.items if item.category == "submodule-dirty-review"]
    assert dirty_items
    assert all("owner_review_pending=" in item.blocker for item in dirty_items)
    assert all("Owner-review recommendation summary:" in item.next_action for item in dirty_items)
    assert all("rewrite_then_commit" in item.next_action for item in dirty_items)
    assert all("discard_from_submodule" in item.next_action for item in dirty_items)
    assert all("pending_owner_review" in item.next_action for item in dirty_items)
    assert all(str(OWNER_REVIEW_ACTION_PACKET) in item.next_action for item in dirty_items)
    assert all(str(OWNER_REVIEW_RECOMMENDATIONS) in item.next_action for item in dirty_items)
    assert all(str(OWNER_REVIEW_EVIDENCE_INDEX) in item.next_action for item in dirty_items)
    assert all(str(OWNER_REVIEW_DECISION_TEMPLATE) in item.next_action for item in dirty_items)
    assert all(str(DEFAULT_DECISION_FILE) in item.next_action for item in dirty_items)
    assert all(f"`{APPROVED_DECISION_STATUS}`" in item.next_action for item in dirty_items)
    assert all("non-placeholder reviewer" in item.next_action for item in dirty_items)
    assert all("`YYYY-MM-DD` review date" in item.next_action for item in dirty_items)
    assert all("python -m scripts.uxfd_owner_review_gate" in item.next_action for item in dirty_items)
    assert all("submodule_dirty_triage.json" in item.evidence for item in dirty_items)
    assert all(str(OWNER_REVIEW_ACTION_PACKET) in item.evidence for item in dirty_items)
    assert all(str(OWNER_REVIEW_RECOMMENDATIONS) in item.evidence for item in dirty_items)
    assert all(str(OWNER_REVIEW_EVIDENCE_INDEX) in item.evidence for item in dirty_items)
    assert all(str(OWNER_REVIEW_DECISION_TEMPLATE) in item.evidence for item in dirty_items)
    assert all(str(DEFAULT_DECISION_FILE) in item.evidence for item in dirty_items)
    top_items = [item for item in report.items if item.category == "top-representative-evidence"]
    assert len(top_items) == 7
    assert any(item.item_id == "TOP-Q7-TIMESEG" for item in top_items)
    assert any("local proxy entries=B02, A05, A07" in item.blocker for item in top_items)
    assert all("representative-only" in item.next_action for item in top_items)


def test_persisted_readiness_backlog_matches_current_backlog() -> None:
    report = evaluate_readiness_backlog()

    assert PERSISTED_READINESS_BACKLOG_MD.read_text(encoding="utf-8") == render_markdown(report)


def test_readiness_backlog_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    markdown = tmp_path / "backlog.md"
    json_path = tmp_path / "backlog.json"

    assert main(["--output", str(markdown)]) == 2
    text = markdown.read_text(encoding="utf-8")
    assert "UXFD Readiness Backlog" in text
    assert "not accepted experiment evidence" in text
    assert "Q0-PRELAUNCH-GATE" in text
    assert "Q0-GPU-PREFLIGHT" in text

    assert (
        main(
            [
                "--format",
                "json",
                "--output",
                str(json_path),
                "--allow-not-ready",
            ]
        )
        == 0
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert payload["open_items"] == len(payload["items"])

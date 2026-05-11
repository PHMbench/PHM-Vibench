import json
from pathlib import Path

from scripts.uxfd_readiness_backlog import (
    evaluate_readiness_backlog,
    main,
    render_markdown,
)


PERSISTED_READINESS_BACKLOG_MD = Path("paper/UXFD_paper/results/readiness_backlog.md")


def test_readiness_backlog_prioritizes_gpu_and_paper07() -> None:
    report = evaluate_readiness_backlog()

    assert report.ready is False
    assert report.open_items > 10
    assert report.items[0].item_id == "Q0-GPU-PREFLIGHT"
    assert report.items[1].item_id == "Q0-ARTIFACT-COVERAGE"
    assert any(item.scope == "TII_operator_attention" for item in report.items[:10])
    assert not any(item.item_id == "Q0-PAPER02-PLANNING-COMMIT" for item in report.items)
    assert not any(item.item_id == "Q0-PARENT-GOAL-CHECKPOINT-COMMIT" for item in report.items)
    assert all(item.category != "commit-recovery" for item in report.items)
    assert all(item.category != "low-tier-source-hygiene" for item in report.items)
    assert any(item.category == "submodule-dirty-review" for item in report.items)


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

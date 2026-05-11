import json
from pathlib import Path

from scripts.uxfd_readiness_backlog import evaluate_readiness_backlog, main


def test_readiness_backlog_prioritizes_gpu_and_paper07() -> None:
    report = evaluate_readiness_backlog()

    assert report.ready is False
    assert report.open_items > 10
    assert report.items[0].item_id == "Q0-GPU-PREFLIGHT"
    assert report.items[1].item_id == "Q0-ARTIFACT-COVERAGE"
    assert any(item.scope == "TII_operator_attention" for item in report.items[:10])
    assert any(item.category == "submodule-dirty-review" for item in report.items)


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

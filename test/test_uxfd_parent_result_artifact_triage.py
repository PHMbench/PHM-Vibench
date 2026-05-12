from pathlib import Path

from scripts.uxfd_parent_result_artifact_triage import (
    DEFAULT_OUTPUT,
    PROMOTE_ONLY_THROUGH_GATE,
    evaluate_parent_artifact_triage,
    render_markdown,
)


def test_parent_result_artifact_triage_marks_generated_figures(tmp_path: Path) -> None:
    figure_root = tmp_path / "figures"
    figure_root.mkdir()
    figure = figure_root / "training_history.png"
    figure.write_bytes(b"demo")

    report = evaluate_parent_artifact_triage((figure_root,))

    assert report.clean is False
    assert len(report.entries) == 1
    assert report.entries[0].path == str(figure)
    assert report.entries[0].recommended_action == PROMOTE_ONLY_THROUGH_GATE
    assert "not accepted experiment evidence" in render_markdown(report)


def test_persisted_parent_result_artifact_triage_matches_current_report() -> None:
    report = evaluate_parent_artifact_triage()

    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == render_markdown(report)

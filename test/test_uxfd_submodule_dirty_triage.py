from pathlib import Path
from types import SimpleNamespace

from scripts.uxfd_submodule_dirty_triage import (
    DO_NOT_AUTO_COMMIT,
    PRESERVE_SESSION,
    PROMOTE_ONLY_THROUGH_GATE,
    DirtyEntry,
    _classify_path,
    _content_risk_markers,
    _summarize_entries,
    evaluate_dirty_triage,
    render_markdown,
)


PERSISTED_DIRTY_TRIAGE_MD = Path("paper/UXFD_paper/results/submodule_dirty_triage.md")


def test_classify_dirty_paths_by_review_policy() -> None:
    assert _classify_path(".codex/state.json") == ("agent_workspace", PRESERVE_SESSION)
    assert _classify_path("sessions/run.md") == ("session_workspace", PRESERVE_SESSION)
    assert _classify_path("results/metrics.json") == (
        "experiment_output",
        PROMOTE_ONLY_THROUGH_GATE,
    )
    assert _classify_path("manuscript/final_tex/main.tex") == (
        "manuscript_draft",
        DO_NOT_AUTO_COMMIT,
    )
    assert _classify_path("manuscript/AUTORESEARCH_EVIDENCE.md") == (
        "historical_autoresearch_evidence_draft",
        DO_NOT_AUTO_COMMIT,
    )
    assert _classify_path("doc/demo_explanation.txt") == (
        "generated_or_result_artifact",
        PROMOTE_ONLY_THROUGH_GATE,
    )
    assert _classify_path("scripts/run_probe.py") == (
        "source_or_experiment_script",
        DO_NOT_AUTO_COMMIT,
    )


def test_summarize_entries_counts_modified_and_untracked() -> None:
    entries = (
        DirtyEntry("paper/A", "M", "README.md", "project_document", DO_NOT_AUTO_COMMIT),
        DirtyEntry("paper/A", "??", "outputs/run.log", "experiment_output", PROMOTE_ONLY_THROUGH_GATE),
        DirtyEntry("paper/B", "??", "sessions/a.md", "session_workspace", PRESERVE_SESSION),
    )

    summaries = _summarize_entries(entries)

    assert len(summaries) == 2
    assert summaries[0].submodule == "paper/A"
    assert summaries[0].total == 2
    assert summaries[0].modified == 1
    assert summaries[0].untracked == 1
    assert summaries[1].categories == {"session_workspace": 1}


def test_content_risk_markers_flag_stale_claims_and_gpu_bindings(tmp_path: Path) -> None:
    submodule = tmp_path / "paper"
    evidence = submodule / "manuscript" / "AUTORESEARCH_EVIDENCE.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        "\n".join(
            [
                "- accepted: `True`",
                "- exec_root: `/tmp/PHM-Vibench copy 2`",
                "- command: `CUDA_VISIBLE_DEVICES=5 python run.py`",
            ]
        ),
        encoding="utf-8",
    )

    markers = _content_risk_markers(submodule, "manuscript/AUTORESEARCH_EVIDENCE.md")

    assert markers == (
        "stale_exec_root",
        "historical_accepted_claim",
        "nonlocal_gpu_binding",
    )


def test_render_markdown_marks_report_as_non_evidence() -> None:
    entries = (
        DirtyEntry("paper/A", "??", "outputs/run.log", "experiment_output", PROMOTE_ONLY_THROUGH_GATE),
    )

    report = SimpleNamespace(
        clean=False,
        summaries=_summarize_entries(entries),
        entries=entries,
    )

    text = render_markdown(report)

    assert "not accepted experiment evidence" in text
    assert PROMOTE_ONLY_THROUGH_GATE in text
    assert "Risk Markers" in text


def test_persisted_dirty_triage_report_matches_current_triage() -> None:
    report = evaluate_dirty_triage()

    assert PERSISTED_DIRTY_TRIAGE_MD.read_text(encoding="utf-8") == render_markdown(report)

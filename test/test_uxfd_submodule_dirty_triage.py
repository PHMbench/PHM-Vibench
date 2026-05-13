import json
from pathlib import Path
from types import SimpleNamespace

from scripts.uxfd_submodule_dirty_triage import (
    DO_NOT_AUTO_COMMIT,
    PRESERVE_SESSION,
    PROMOTE_ONLY_THROUGH_GATE,
    DirtyEntry,
    _action_counts,
    _action_counts_by_submodule,
    _classify_path,
    _content_risk_markers,
    _owner_decision_template,
    _path_risk_markers,
    _risk_marker_counts,
    _review_command,
    _summarize_entries,
    build_payload,
    evaluate_dirty_triage,
    render_markdown,
)


PERSISTED_DIRTY_TRIAGE_MD = Path("paper/UXFD_paper/results/submodule_dirty_triage.md")
PERSISTED_DIRTY_TRIAGE_JSON = Path("paper/UXFD_paper/results/submodule_dirty_triage.json")


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
                "- status: `ready`",
                "- command: `python main.py --config_dir configs/old.yaml`",
                "- command: `CUDA_VISIBLE_DEVICES=5 python run.py`",
            ]
        ),
        encoding="utf-8",
    )

    markers = _content_risk_markers(submodule, "manuscript/AUTORESEARCH_EVIDENCE.md")

    assert markers == (
        "stale_exec_root",
        "deprecated_config_dir_dispatch",
        "unaccepted_readiness_claim",
        "historical_accepted_claim",
        "nonlocal_gpu_binding",
    )


def test_path_risk_markers_flag_tracked_generated_artifacts() -> None:
    assert _path_risk_markers(
        " M",
        "benchmark_results/method_comparison_radar.png",
        "experiment_output",
    ) == (
        "tracked_generated_artifact_dirty",
        "binary_or_large_artifact",
    )
    assert _path_risk_markers("??", "results/demo.log", "experiment_output") == ()
    assert _path_risk_markers(" M", "best_model.pth", "generated_or_result_artifact") == (
        "tracked_generated_artifact_dirty",
        "binary_or_large_artifact",
    )


def test_action_and_risk_counts_summarize_commit_blockers() -> None:
    entries = (
        DirtyEntry(
            "paper/A",
            "M",
            "results/metrics.json",
            "experiment_output",
            PROMOTE_ONLY_THROUGH_GATE,
            ("tracked_generated_artifact_dirty",),
        ),
        DirtyEntry(
            "paper/A",
            "??",
            "manuscript/AUTORESEARCH_EVIDENCE.md",
            "historical_autoresearch_evidence_draft",
            DO_NOT_AUTO_COMMIT,
            (
                "stale_exec_root",
                "deprecated_config_dir_dispatch",
                "unaccepted_readiness_claim",
                "historical_accepted_claim",
            ),
        ),
        DirtyEntry(
            "paper/B",
            "??",
            "sessions/run.md",
            "session_workspace",
            PRESERVE_SESSION,
        ),
    )

    assert _action_counts(entries) == {
        DO_NOT_AUTO_COMMIT: 1,
        PRESERVE_SESSION: 1,
        PROMOTE_ONLY_THROUGH_GATE: 1,
    }
    assert _risk_marker_counts(entries) == {
        "deprecated_config_dir_dispatch": 1,
        "historical_accepted_claim": 1,
        "stale_exec_root": 1,
        "tracked_generated_artifact_dirty": 1,
        "unaccepted_readiness_claim": 1,
    }


def test_action_counts_by_submodule_builds_owner_review_queue() -> None:
    entries = (
        DirtyEntry("paper/A", "M", "results/a.json", "experiment_output", PROMOTE_ONLY_THROUGH_GATE),
        DirtyEntry("paper/A", "??", "EXPERIMENT_DESIGN.md", "planning_or_contract_draft", DO_NOT_AUTO_COMMIT),
        DirtyEntry("paper/B", "??", "sessions/run.md", "session_workspace", PRESERVE_SESSION),
    )

    assert _action_counts_by_submodule(entries) == {
        "paper/A": {
            DO_NOT_AUTO_COMMIT: 1,
            PROMOTE_ONLY_THROUGH_GATE: 1,
        },
        "paper/B": {PRESERVE_SESSION: 1},
    }


def test_owner_decision_template_keeps_owner_review_entries_pending() -> None:
    entries = (
        DirtyEntry("paper/A", "M", "results/a.json", "experiment_output", PROMOTE_ONLY_THROUGH_GATE),
        DirtyEntry("paper/A", "??", "EXPERIMENT_DESIGN.md", "planning_or_contract_draft", DO_NOT_AUTO_COMMIT),
    )

    assert _owner_decision_template(entries) == (
        {
            "submodule": "paper/A",
            "path": "EXPERIMENT_DESIGN.md",
            "decision": "pending_owner_review",
            "reviewer": "TODO",
            "notes": "TODO",
        },
    )


def test_review_command_is_non_destructive() -> None:
    modified = DirtyEntry(
        "paper/A",
        "M",
        "results/a.json",
        "experiment_output",
        PROMOTE_ONLY_THROUGH_GATE,
    )
    untracked = DirtyEntry(
        "paper/A",
        "??",
        "EXPERIMENT_DESIGN.md",
        "planning_or_contract_draft",
        DO_NOT_AUTO_COMMIT,
    )

    assert _review_command(modified) == "git -C paper/A diff -- results/a.json"
    assert (
        _review_command(untracked)
        == "git -C paper/A status --short -- EXPERIMENT_DESIGN.md"
    )


def test_content_risk_markers_flag_chinese_readiness_claim(tmp_path: Path) -> None:
    submodule = tmp_path / "paper"
    summary = submodule / "results" / "PAPER_READY_SUMMARY.md"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "# 论文就绪结果汇总\n\n## 投稿状态评估\n\n表格模板可直接用于论文。\n",
        encoding="utf-8",
    )

    assert _content_risk_markers(
        submodule,
        "results/PAPER_READY_SUMMARY.md",
    ) == ("unaccepted_readiness_claim",)


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
    assert "Commit-Blocking Verdict" in text
    assert "Owner Review Queue" in text
    assert "Owner-Review Entry Checklist" in text
    assert "Owner Decision Template" in text
    assert "pending_owner_review" in text
    assert "Artifact-Gate Promotion Checklist" in text
    assert "`git -C paper/A status --short`" in text
    assert "Auto-commit safe entries: `0`" in text
    assert PROMOTE_ONLY_THROUGH_GATE in text
    assert "Risk Markers" in text


def test_persisted_dirty_triage_report_matches_current_triage() -> None:
    report = evaluate_dirty_triage()

    assert PERSISTED_DIRTY_TRIAGE_MD.read_text(encoding="utf-8") == render_markdown(report)


def test_persisted_dirty_triage_json_matches_current_triage() -> None:
    report = evaluate_dirty_triage()

    assert json.loads(PERSISTED_DIRTY_TRIAGE_JSON.read_text(encoding="utf-8")) == build_payload(report)

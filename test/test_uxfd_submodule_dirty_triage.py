import json
from pathlib import Path
from types import SimpleNamespace

from scripts.uxfd_submodule_dirty_triage import (
    DO_NOT_AUTO_COMMIT,
    OWNER_REVIEW_ACTION_PACKET,
    OWNER_REVIEW_DECISION_TEMPLATE,
    OWNER_REVIEW_EVIDENCE_INDEX,
    OWNER_REVIEW_RECOMMENDATIONS,
    PRESERVE_SESSION,
    PROMOTE_ONLY_THROUGH_GATE,
    DirtyEntry,
    DirtyTriageReport,
    _action_counts,
    _action_counts_by_submodule,
    _classify_path,
    _content_review_command,
    _content_risk_markers,
    _owner_decision_template,
    _owner_review_packets,
    _owner_review_note,
    _owner_resolution_gates,
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
            "decision_id": "OR-01",
            "submodule": "paper/A",
            "path": "EXPERIMENT_DESIGN.md",
            "current_status": "??",
            "category": "planning_or_contract_draft",
            "risk_markers": [],
            "recommended_decisions": [
                "rewrite_then_commit",
                "discard_from_submodule",
            ],
            "decision": "pending_owner_review",
            "reviewer": "TODO",
            "review_date": "TODO",
            "notes": "Useful planning draft only after current-root, parent-gated rewrite.",
        },
    )


def test_owner_resolution_gates_define_commit_cleanup_decisions() -> None:
    gates = {row["decision"]: row["required_gate"] for row in _owner_resolution_gates()}

    assert set(gates) == {
        "commit_after_review",
        "rewrite_then_commit",
        "discard_from_submodule",
    }
    assert "readiness claims" in gates["commit_after_review"]
    assert "reruns dirty triage" in gates["rewrite_then_commit"]
    assert "do not delete it automatically" in gates["discard_from_submodule"]


def test_owner_review_packets_include_machine_readable_review_steps() -> None:
    entries = (
        DirtyEntry("paper/A", "M", "results/a.json", "experiment_output", PROMOTE_ONLY_THROUGH_GATE),
        DirtyEntry(
            "paper/A",
            "??",
            "EXPERIMENT_DESIGN.md",
            "planning_or_contract_draft",
            DO_NOT_AUTO_COMMIT,
            ("deprecated_config_dir_dispatch",),
        ),
    )

    packets = _owner_review_packets(entries)

    assert packets == (
        {
            "decision_id": "OR-01",
            "submodule": "paper/A",
            "path": "EXPERIMENT_DESIGN.md",
            "status": "??",
            "category": "planning_or_contract_draft",
            "risk_markers": ["deprecated_config_dir_dispatch"],
            "review_command": "git -C paper/A status --short -- EXPERIMENT_DESIGN.md",
            "content_review_command": "sed -n '1,220p' -- paper/A/EXPERIMENT_DESIGN.md",
            "decision_state": "pending_owner_review",
            "allowed_decisions": [
                "commit_after_review",
                "rewrite_then_commit",
                "discard_from_submodule",
            ],
            "recommended_decisions": [
                "rewrite_then_commit",
                "discard_from_submodule",
            ],
            "default_next_action": (
                "paper owner must choose an allowed decision before this entry is "
                "staged, rewritten, or cleaned up"
            ),
        },
    )


def test_owner_review_note_preserves_all_risk_categories() -> None:
    entry = DirtyEntry(
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
            "nonlocal_gpu_binding",
        ),
    )

    note = _owner_review_note(entry)

    assert "local GPU 0,1 policy" in note
    assert "python main.py --config" in note
    assert "accepted_runs=0" in note
    assert "submission_ready=false" in note
    assert "stale execution-root" in note


def test_build_payload_exposes_owner_review_recommendations() -> None:
    report = DirtyTriageReport(
        clean=False,
        summaries=(),
        entries=(),
    )

    payload = build_payload(report)

    assert payload["owner_review_recommendations"] == {
        "path": str(OWNER_REVIEW_RECOMMENDATIONS),
        "exists": OWNER_REVIEW_RECOMMENDATIONS.is_file(),
        "action_packet_path": str(OWNER_REVIEW_ACTION_PACKET),
        "action_packet_exists": OWNER_REVIEW_ACTION_PACKET.is_file(),
        "evidence_index_path": str(OWNER_REVIEW_EVIDENCE_INDEX),
        "evidence_index_exists": OWNER_REVIEW_EVIDENCE_INDEX.is_file(),
        "decision_template_path": str(OWNER_REVIEW_DECISION_TEMPLATE),
        "decision_template_exists": OWNER_REVIEW_DECISION_TEMPLATE.is_file(),
        "status": "decision_support_only",
        "required_use": (
            "paper owners should read the action packet, recommendation note, and "
            "line-level evidence index before "
            "choosing commit_after_review, rewrite_then_commit, or discard_from_submodule"
        ),
    }


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
    assert _content_review_command(modified) == "git -C paper/A diff -- results/a.json"
    assert (
        _content_review_command(untracked)
        == "sed -n '1,220p' -- paper/A/EXPERIMENT_DESIGN.md"
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
    assert "Owner Review Recommendations" in text
    assert str(OWNER_REVIEW_RECOMMENDATIONS) in text
    assert str(OWNER_REVIEW_ACTION_PACKET) in text
    assert str(OWNER_REVIEW_EVIDENCE_INDEX) in text
    assert str(OWNER_REVIEW_DECISION_TEMPLATE) in text
    assert "Owner-Review Entry Checklist" in text
    assert "Owner Decision Template" in text
    assert "Recommended Decisions" in text
    assert "Review Date" in text
    assert "Owner Review Packets" in text
    assert "Decision ID" in text
    assert "Content Review Command" in text
    assert "Owner Resolution Gates" in text
    assert "pending_owner_review" in text
    assert "commit_after_review" in text
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


def test_owner_review_decision_template_is_machine_readable() -> None:
    template = json.loads(OWNER_REVIEW_DECISION_TEMPLATE.read_text(encoding="utf-8"))

    assert template["status"] == "template_only_not_owner_approved"
    assert set(template["allowed_decisions"]) == {
        "commit_after_review",
        "rewrite_then_commit",
        "discard_from_submodule",
    }
    assert len(template["records"]) == 6
    assert [record["decision_id"] for record in template["records"]] == [
        "OR-01",
        "OR-02",
        "OR-03",
        "OR-04",
        "OR-05",
        "OR-06",
    ]
    assert all(record["decision"] == "pending_owner_review" for record in template["records"])
    assert {
        (record["submodule"], record["path"]) for record in template["records"]
    } == {
        ("paper/UXFD_paper/Explainable_FD_Toolkit", "EXPERIMENT_DESIGN.md"),
        (
            "paper/UXFD_paper/Explainable_FD_Toolkit",
            "manuscript/AUTORESEARCH_EVIDENCE.md",
        ),
        ("paper/UXFD_paper/1D-2D_fusion_explainable", "EXPERIMENT_DESIGN.md"),
        (
            "paper/UXFD_paper/1D-2D_fusion_explainable",
            "manuscript/AUTORESEARCH_EVIDENCE.md",
        ),
        ("paper/UXFD_paper/MOE_explainable", "EXPERIMENT_DESIGN.md"),
        ("paper/UXFD_paper/MOE_explainable", "manuscript/AUTORESEARCH_EVIDENCE.md"),
    }


def test_owner_review_action_packet_is_non_approval_response_form() -> None:
    template = json.loads(OWNER_REVIEW_DECISION_TEMPLATE.read_text(encoding="utf-8"))
    text = OWNER_REVIEW_ACTION_PACKET.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "not owner approval" in text
    assert "not accepted experiment evidence" in normalized
    assert "submodule_owner_review_decisions.json" in text
    assert "OR-01" in text
    assert "OR-06" in text
    assert "python -m scripts.uxfd_owner_review_gate --format markdown" in text
    assert "template_only_not_owner_approved" in text
    assert "pending_owner_review" in text
    for record in template["records"]:
        assert record["submodule"] in text
        assert record["path"] in text


def test_owner_review_decision_template_matches_current_dirty_triage() -> None:
    template = json.loads(OWNER_REVIEW_DECISION_TEMPLATE.read_text(encoding="utf-8"))
    report = evaluate_dirty_triage()

    assert template["records"] == list(_owner_decision_template(report.entries))

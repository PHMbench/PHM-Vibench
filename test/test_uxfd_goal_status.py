from pathlib import Path

from scripts.uxfd_goal_status import DEFAULT_STATUS_DIR, generate_status_reports


def test_goal_status_generator_writes_current_non_evidence_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "status"

    written = generate_status_reports(output_dir, generated_on="2026-05-14")

    assert len(written) == 10
    overall = (output_dir / "status_00_overall.md").read_text(encoding="utf-8")
    gpu = (output_dir / "status_09_gpu_execution.md").read_text(encoding="utf-8")
    citation = (output_dir / "status_08_citation_readiness.md").read_text(
        encoding="utf-8"
    )

    assert "not accepted experiment evidence" in overall
    assert "## 2026-05-16 Stage-2 Task Binding" in overall
    assert ".specify/goals/v2/status/uxfd_goal_stage_report_2026-05-16.md" in overall
    assert ".specify/goals/v2/tasks/uxfd_goal_followup_tasks_2026-05-16.md" in overall
    assert "control-plane readiness: strong progress" in overall
    assert "evidence-plane readiness: blocked" in overall
    assert "submission readiness: not achieved" in overall
    assert "`T00` -> `T01` -> `T02` -> `T03` -> `T04` -> `T05`" in overall
    assert "Do not mark the active goal complete" in overall
    assert "Objective audit: `met=87`, `not_met=13`, `blocked=1`" in overall
    assert "Experiment launch gate ready: `False`" in overall
    assert "Experiment launch blockers: `3`" in overall
    assert "Live launch preflight accepted: `False`" in overall
    assert "The experiment launch gate is the only authority" in overall
    assert (
        "only after the experiment launch gate passes without override flags"
        in overall
    )
    assert "recorded owner decisions" in overall
    assert "Artifact coverage: `0/104`" in overall
    assert "SOTA gate ready: `False`" in overall
    assert "SOTA aggregate records: `7`" in overall
    assert "Owner-review gate ready: `False`" in overall
    assert "Owner-review pending records: `6`" in overall
    assert "## Owner-Review Decision Gate" in overall
    assert "submodule_owner_review_decisions.template.json" in overall
    assert "owner decision file missing" in overall
    assert "## Dirty Submodule Owner Review Queue" in overall
    assert "Do not auto-commit these entries" in overall
    assert "| `paper/UXFD_paper/Explainable_FD_Toolkit` | 2 | 20 | 0 |" in overall
    assert "Queue dry-run entries: `104`" in gpu
    assert "Bound GPU tasks: `T04` restore local GPU visibility" in gpu
    assert "Experiment launch gate ready: `False`" in gpu
    assert "Experiment launch blockers: `3`" in gpu
    assert "Owner-review gate ready: `False`" in gpu
    assert "Owner-review pending records: `6`" in gpu
    assert "Live preflight accepted: `False`" in gpu
    assert "## Current Launch Gate Blockers" in gpu
    assert "owner-review gate not ready" in gpu
    assert "gpu queue static gate not executable" in gpu
    assert "live GPU preflight not accepted" in gpu
    assert "Static launch gate enabled: `True`" in gpu
    assert "## Experiment Launch Decision" in gpu
    assert "Do not launch `queue_launch_plan.sh`" in gpu
    assert "python -m scripts.uxfd_experiment_launch_gate --format markdown" in gpu
    assert "python -m scripts.uxfd_owner_review_gate --format markdown" in gpu
    assert (
        "python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight"
        in gpu
    )
    assert "must not copy the template into an approved decision file" in gpu
    assert "`log_path` must point to a non-empty log file" in gpu
    assert "`config_path` must point to parseable, non-empty YAML" in gpu
    assert "`batch_size` must be a positive integer" in gpu
    assert "`runtime` must be a positive `HH:MM:SS` duration" in gpu
    assert "`precision` must be one of" in gpu
    assert "`evidence_level` must be `accepted_same_protocol`" in gpu
    assert "`sha256:<64 lowercase hex>`" in gpu
    assert "at least one finite numeric metric" in gpu
    assert "TODO, NaN, and infinite payloads are rejected" in gpu
    assert "dirty, modified, unknown, or uncommitted" in gpu
    assert "matched-seed aggregate evidence" in gpu
    assert "a single accepted run is not SOTA evidence" in gpu
    assert "## TOP Representative Execution Bindings" in gpu
    assert "`TOP-Q7-TIMESEG`" in gpu
    assert "`B02, A05, A07`" in gpu
    assert "representative-only" in gpu
    assert "`pending_gpu_and_artifacts`" in gpu
    assert "Evidence ready: `False`" in citation
    assert "Bound manuscript task: `M-04` refresh recent-work README/citations" in citation
    assert "Bound SOTA task: `SOTA-03` create accepted-run refs" in citation
    assert "Source verification ready: `True`" in citation
    assert "## Paper-Local Exact-Status Scope" in citation
    assert "Unscoped Exact Claims" in citation
    assert "| `LLM_Explainable_FD_Toolkit` | 7 | 0 | 0 | `True` |" in citation
    assert "## Evidence Activation Workflow" in citation
    assert "Policy and source verification are literature hygiene only" in citation
    assert "`paper/UXFD_paper/results/accepted_runs`" in citation
    assert "external exact code/config is integrated" in citation
    assert "`python -m scripts.uxfd_artifact_gate`" in citation
    assert "`python -m scripts.uxfd_sota_gate`" in citation
    assert "`python -m scripts.uxfd_recent_work_gate`" in citation


def test_persisted_goal_status_reports_match_generator(tmp_path: Path) -> None:
    output_dir = tmp_path / "status"

    generated = generate_status_reports(output_dir, generated_on="2026-05-14")

    for generated_path in generated:
        persisted_path = DEFAULT_STATUS_DIR / generated_path.name
        assert persisted_path.exists()
        assert persisted_path.read_text(encoding="utf-8") == generated_path.read_text(
            encoding="utf-8"
        )

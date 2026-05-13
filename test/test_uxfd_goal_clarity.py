from pathlib import Path


GOAL_DIR = Path("paper/UXFD_paper/goal")
GOAL_README = GOAL_DIR / "README.md"
CLARITY_AUDIT = Path("paper/UXFD_paper/results/goal_clarity_audit_current.md")
COMMIT_RECOVERY_PLAN = Path("paper/UXFD_paper/results/commit_recovery_plan.md")

STALE_PERSISTED_GATE_OUTPUTS = (
    "paper/UXFD_paper/results/objective_audit.json",
    "paper/UXFD_paper/results/objective_audit.md",
    "paper/UXFD_paper/results/submission_gate.json",
    "paper/UXFD_paper/results/submission_gate.md",
    "paper/UXFD_paper/results/recent_work_gate.json",
    "paper/UXFD_paper/results/recent_work_gate.md",
    "paper/UXFD_paper/results/artifact_gate.json",
)

STALE_EXECUTION_MARKERS = (
    "PHM-Vibench copy 2",
    "Paper/1D",
    "--config_dir",
    "config_dir",
)

LOW_TIER_MARKERS = (
    "Scientific Reports",
    "MDPI",
    "IEEE TIM",
    "IEEE Transactions on Instrumentation and Measurement",
    "IEEE Access",
    "Applied Sciences",
    "Sensors",
    "Mathematics",
)


def _goal_files() -> tuple[Path, ...]:
    return tuple(sorted(GOAL_DIR.glob("*.md"))) + (GOAL_DIR / "09_gpu_execution_queue.yaml",)


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    assert start >= 0, f"missing section {heading!r}"
    rest = text[start + len(marker) :]
    next_heading = rest.find("\n## ")
    if next_heading >= 0:
        return rest[:next_heading]
    return rest


def test_goal_readme_exposes_clarity_audit_without_treating_it_as_evidence() -> None:
    text = GOAL_README.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert str(CLARITY_AUDIT) in text
    assert "not a submission-readiness gate" in normalized
    assert "not accepted experiment evidence" in normalized


def test_goal_readme_uses_persisted_current_gate_outputs() -> None:
    text = GOAL_README.read_text(encoding="utf-8")

    assert "paper/UXFD_paper/results/objective_audit_current.json" in text
    assert "paper/UXFD_paper/results/objective_audit_current.md" in text
    assert "paper/UXFD_paper/results/submodule_owner_review_gate_current.json" in text
    assert "paper/UXFD_paper/results/submodule_owner_review_gate_current.md" in text
    assert "paper/UXFD_paper/results/submission_gate_current.json" in text
    assert "paper/UXFD_paper/results/submission_gate_current.md" in text
    assert "paper/UXFD_paper/results/recent_work_gate_current.json" in text
    assert "paper/UXFD_paper/results/recent_work_gate_current.md" in text
    assert "paper/UXFD_paper/results/artifact_gate_queue_coverage.md" in text
    assert "paper/UXFD_paper/results/accepted_runs" in text

    stale = [path for path in STALE_PERSISTED_GATE_OUTPUTS if path in text]
    assert not stale


def test_goal_readme_exposes_owner_review_gate_as_pre_execution_gate() -> None:
    text = GOAL_README.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "## Owner Review Gate" in text
    assert "python -m scripts.uxfd_owner_review_gate --format markdown" in text
    assert "submodule_owner_review_decisions.json" in text
    assert "submodule_owner_review_decisions.template.json" in text
    assert "template is decision support only" in normalized
    assert "it is not paper-owner approval" in normalized
    assert "pending_owner_review" in text


def test_goal_clarity_audit_exists_and_records_non_ready_verdict() -> None:
    text = CLARITY_AUDIT.read_text(encoding="utf-8")

    assert "The goal package is clear enough" in text
    assert "not clear to treat any paper as submission-ready" in text
    assert "GPU queue cannot execute" in text
    assert "zero accepted records" in text


def test_goal_files_do_not_use_stale_execution_paths() -> None:
    violations: list[str] = []
    for path in _goal_files():
        text = path.read_text(encoding="utf-8")
        for marker in STALE_EXECUTION_MARKERS:
            if marker in text:
                violations.append(f"{path}:{marker}")

    assert not violations


def test_low_tier_sources_are_only_documented_as_excluded_context() -> None:
    allowed_context_files = {
        GOAL_DIR / "00_overall_goal.md",
        GOAL_DIR / "08_recent_work_citation_readme.md",
        GOAL_DIR / "99_submission_readiness_matrix.md",
    }
    violations: list[str] = []
    for path in _goal_files():
        text = path.read_text(encoding="utf-8")
        for marker in LOW_TIER_MARKERS:
            if marker in text and path not in allowed_context_files:
                violations.append(f"{path}:{marker}")

    assert not violations

    overall = (GOAL_DIR / "00_overall_goal.md").read_text(encoding="utf-8")
    recent = (GOAL_DIR / "08_recent_work_citation_readme.md").read_text(encoding="utf-8")
    assert "Excluded sources for core claims" in overall
    assert "Rejected for UXFD core related work" in recent


def test_commit_recovery_plan_keeps_objective_audit_refresh_separate() -> None:
    text = COMMIT_RECOVERY_PLAN.read_text(encoding="utf-8")
    phase3 = _section(text, "Phase 3: Parent Checkpoint Commit")
    phase4 = _section(text, "Phase 4: Objective Audit Refresh Commit")
    phase3_staging_command = phase3.split("Do not stage:")[0]

    assert str(COMMIT_RECOVERY_PLAN) in GOAL_README.read_text(encoding="utf-8")
    assert "objective_audit_current.json" not in phase3_staging_command
    assert "objective_audit_current.md" not in phase3_staging_command
    assert "objective_audit_current.json" in phase4
    assert "objective_audit_current.md" in phase4
    assert "parent UXFD goal-control checkpoint committed" in phase4

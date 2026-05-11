from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from scripts.uxfd_recent_work_gate import evaluate_recent_work_gate
from scripts.uxfd_submission_gate import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_QUEUE,
    GOAL_DIR,
    REQUIRED_GOAL_FILES,
    evaluate_submission_gate,
)


SPEC_DIR = Path("specs/006-uxfd-ieee-trans-submission-readiness")
CLAUDE_TEAM_DIR = Path(".codex/claude-team-runs/20260511-uxfd-ieee-trans-review")
HANDOFF_PATH = Path(
    ".claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md"
)

REQUIRED_SPEC_FILES = (
    "spec.md",
    "plan.md",
    "tasks.md",
    "research.md",
    "data-model.md",
    "quickstart.md",
    "contracts/uxfd-ieee-trans-submission-readiness-contract.md",
    "checklists/requirements.md",
    "checklists/submission-readiness.md",
)

CLAUDE_TEAM_OUTPUTS = (
    "report.md",
    "risks.md",
    "test-log.md",
)

CODEX_SUBAGENT_LAUNCH = "CODEX_SUBAGENT_LAUNCH.md"

EXECUTION_ARTIFACTS = (
    ("GPU execution runbook", Path("paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md")),
    ("live GPU preflight snapshot", Path("paper/UXFD_paper/results/gpu_queue_live_preflight.json")),
    ("combined GPU launch plan", Path("paper/UXFD_paper/results/queue_launch_plan.sh")),
    ("GPU0 launch shard", Path("paper/UXFD_paper/results/queue_launch_shards/gpu0.sh")),
    ("GPU1 launch shard", Path("paper/UXFD_paper/results/queue_launch_shards/gpu1.sh")),
    (
        "accepted-run template manifest",
        Path("paper/UXFD_paper/results/accepted_run_templates/manifest.json"),
    ),
    (
        "artifact queue coverage report",
        Path("paper/UXFD_paper/results/artifact_gate_queue_coverage.md"),
    ),
)

PAPER_SUBMODULES = (
    Path("paper/UXFD_paper/Explainable_FD_Toolkit"),
    Path("paper/UXFD_paper/1D-2D_fusion_explainable"),
    Path("paper/UXFD_paper/LLM_Explainable_FD_Toolkit"),
    Path("paper/UXFD_paper/MOE_explainable"),
    Path("paper/UXFD_paper/Paper_fuzzy_XFD"),
    Path("paper/UXFD_paper/Neuralsymbolic_theory"),
    Path("paper/UXFD_paper/TII_operator_attention"),
)


@dataclass(frozen=True)
class ObjectiveAuditItem:
    requirement: str
    evidence: str
    status: str
    details: str


@dataclass(frozen=True)
class ObjectiveAuditReport:
    achieved: bool
    objective: str
    items: Tuple[ObjectiveAuditItem, ...]
    blockers: Tuple[str, ...]
    met: int
    not_met: int
    blocked: int
    unverified: int


def _item(requirement: str, evidence: Path | str, status: str, details: str) -> ObjectiveAuditItem:
    return ObjectiveAuditItem(
        requirement=requirement,
        evidence=str(evidence),
        status=status,
        details=details,
    )


def _exists_item(requirement: str, path: Path) -> ObjectiveAuditItem:
    return _item(
        requirement=requirement,
        evidence=path,
        status="met" if path.exists() else "not_met",
        details="exists" if path.exists() else "missing",
    )


def _text_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def _git_status_lines(path: Path) -> Tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def _paper_submodule_cleanliness_item(
    submodule_paths: Sequence[Path] = PAPER_SUBMODULES,
) -> ObjectiveAuditItem:
    dirty: List[str] = []
    unreadable: List[str] = []
    for path in submodule_paths:
        try:
            status_lines = _git_status_lines(path)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            unreadable.append(f"{path} ({exc.__class__.__name__})")
            continue
        if status_lines:
            dirty.append(f"{path.name}:{len(status_lines)}")

    if unreadable:
        return _item(
            requirement="paper submodule working trees clean before parent handoff",
            evidence="git -C <paper_submodule> status --porcelain",
            status="unverified",
            details="unreadable_submodules=" + ", ".join(unreadable),
        )
    if dirty:
        return _item(
            requirement="paper submodule working trees clean before parent handoff",
            evidence="git -C <paper_submodule> status --porcelain",
            status="not_met",
            details="dirty_submodules=" + ", ".join(dirty),
        )
    return _item(
        requirement="paper submodule working trees clean before parent handoff",
        evidence="git -C <paper_submodule> status --porcelain",
        status="met",
        details=f"{len(submodule_paths)} paper submodules clean",
    )


def evaluate_objective_audit(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ObjectiveAuditReport:
    objective = (
        "Execute the UXFD seven-paper goal package, use Spec Kit/Claude Team/"
        "handoff workflow, maintain TOP recent-work and 2x4090 constraints, and "
        "drive all seven papers toward IEEE Transactions submission readiness."
    )
    items: List[ObjectiveAuditItem] = []

    for filename in REQUIRED_GOAL_FILES:
        items.append(_exists_item(f"named goal file {filename}", GOAL_DIR / filename))

    for filename in REQUIRED_SPEC_FILES:
        items.append(_exists_item(f"Spec Kit artifact {filename}", SPEC_DIR / filename))

    items.append(_exists_item("handoff document", HANDOFF_PATH))
    items.append(_exists_item("Claude Team task spec", CLAUDE_TEAM_DIR / "TASK_SPEC.md"))
    items.append(_exists_item("Claude Team launch log", CLAUDE_TEAM_DIR / "LAUNCH_LOG.md"))
    items.append(
        _exists_item(
            "Codex xhigh subagent launch log",
            CLAUDE_TEAM_DIR / CODEX_SUBAGENT_LAUNCH,
        )
    )

    launch_log = CLAUDE_TEAM_DIR / "LAUNCH_LOG.md"
    launch_blocked = _text_contains(launch_log, "Prepared but not launched") or _text_contains(
        launch_log, "rejected by policy"
    )
    subagent_launch = CLAUDE_TEAM_DIR / CODEX_SUBAGENT_LAUNCH
    subagent_outputs_ready = subagent_launch.exists() and all(
        (CLAUDE_TEAM_DIR / filename).exists() for filename in CLAUDE_TEAM_OUTPUTS
    )
    items.append(
        _item(
            requirement="six xhigh/subagent or Claude Team execution evidence",
            evidence=CLAUDE_TEAM_DIR,
            status=(
                "met"
                if subagent_outputs_ready
                else "blocked" if launch_blocked else "unverified"
            ),
            details=(
                "six local Codex xhigh subagents launched and deliverables exist"
                if subagent_outputs_ready
                else "local Claude Team reports are absent; launch log records policy block"
                if launch_blocked
                else "no local report proving team execution"
            ),
        )
    )
    for filename in CLAUDE_TEAM_OUTPUTS:
        path = CLAUDE_TEAM_DIR / filename
        items.append(
            _item(
                requirement=f"Claude Team deliverable {filename}",
                evidence=path,
                status="not_met" if not path.exists() else "met",
                details=(
                    "missing because team launch is blocked and local subagent synthesis is absent"
                    if not path.exists()
                    else "exists"
                ),
            )
        )

    for requirement, path in EXECUTION_ARTIFACTS:
        items.append(_exists_item(requirement, path))

    items.append(_paper_submodule_cleanliness_item())

    submission = evaluate_submission_gate(queue_path=queue_path, artifact_root=artifact_root)
    recent = evaluate_recent_work_gate(queue_path=queue_path)

    items.append(
        _item(
            requirement="seven paper-local baseline/ablation matrices",
            evidence="submission_prep/baseline_ablation_matrix.yaml",
            status="met" if len(submission.papers) == 7 else "not_met",
            details=f"{len(submission.papers)} matrices discovered by submission gate",
        )
    )
    for paper in submission.papers:
        matrix_ready = paper.baselines >= 6 and paper.ablations >= 6
        items.append(
            _item(
                requirement=f"{paper.paper_id}: 6+ baselines and 6+ ablations",
                evidence=paper.matrix_path,
                status="met" if matrix_ready else "not_met",
                details=(
                    f"baselines={paper.baselines}, ablations={paper.ablations}, "
                    f"submission_ready={paper.submission_ready}"
                ),
            )
        )
        items.append(
            _item(
                requirement=f"{paper.paper_id}: IEEE Transactions submission-ready",
                evidence=paper.matrix_path,
                status="not_met" if not paper.submission_ready else "met",
                details=f"strict blockers remaining={len(paper.strict_blockers)}",
            )
        )

    items.append(
        _item(
            requirement="TOP recent-work policy",
            evidence=GOAL_DIR / "08_recent_work_citation_readme.md",
            status="met" if recent.policy_ready else "not_met",
            details=(
                f"accepted_pool_rows={recent.accepted_pool_rows}, "
                f"2026_ids={len(recent.top_2026_ids)}, "
                f"low_tier_violations={len(recent.low_tier_violations)}"
            ),
        )
    )
    items.append(
        _item(
            requirement="TOP representative accepted artifacts",
            evidence=GOAL_DIR / "09_gpu_execution_queue.yaml",
            status="met" if recent.evidence_ready else "not_met",
            details=f"pending_or_blocked_bindings={len(recent.evidence_blockers)}",
        )
    )
    queue_resource_reason = submission.queue_resource_reason.lower()
    queue_status = (
        "met"
        if submission.queue_can_execute
        else "blocked"
        if "blocked" in queue_resource_reason
        else "not_met"
    )
    items.append(
        _item(
            requirement="2x4090 GPU queue executable",
            evidence=queue_path,
            status=queue_status,
            details=submission.queue_resource_reason,
        )
    )
    items.append(
        _item(
            requirement="accepted run artifact metadata",
            evidence=artifact_root,
            status="met" if submission.artifact_gate_accepted else "not_met",
            details=(
                f"records={submission.artifact_gate_records}, "
                f"blockers={len(submission.artifact_gate_blockers)}"
            ),
        )
    )
    items.append(
        _item(
            requirement="cross-paper submission gate",
            evidence="scripts.uxfd_submission_gate",
            status="met" if submission.ready else "not_met",
            details=f"ready={submission.ready}, blockers={len(submission.blockers)}",
        )
    )

    blockers = tuple(
        f"{item.requirement}: {item.details}"
        for item in items
        if item.status != "met"
    )
    status_counts: Mapping[str, int] = {
        status: sum(1 for item in items if item.status == status)
        for status in ("met", "not_met", "blocked", "unverified")
    }
    achieved = not blockers
    return ObjectiveAuditReport(
        achieved=achieved,
        objective=objective,
        items=tuple(items),
        blockers=blockers,
        met=status_counts["met"],
        not_met=status_counts["not_met"],
        blocked=status_counts["blocked"],
        unverified=status_counts["unverified"],
    )


def build_payload(report: ObjectiveAuditReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: ObjectiveAuditReport) -> str:
    lines = [
        "# UXFD Objective Audit",
        "",
        f"- Achieved: `{report.achieved}`",
        f"- Met: `{report.met}`",
        f"- Not met: `{report.not_met}`",
        f"- Blocked: `{report.blocked}`",
        f"- Unverified: `{report.unverified}`",
        "",
        "## Objective",
        "",
        report.objective,
        "",
        "## Prompt-to-Artifact Checklist",
        "",
        "| Status | Requirement | Evidence | Details |",
        "|---|---|---|---|",
    ]
    for item in report.items:
        details = item.details.replace("|", "\\|")
        lines.append(
            f"| `{item.status}` | {item.requirement} | `{item.evidence}` | {details} |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the UXFD active-thread objective")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-achieved", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_objective_audit(queue_path=args.queue, artifact_root=args.artifact_root)
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")

    if report.achieved or args.allow_not_achieved:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

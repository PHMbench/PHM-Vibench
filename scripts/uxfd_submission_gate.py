from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_artifact_gate import ArtifactGateReport, evaluate_artifact_gate
from scripts.uxfd_gpu_queue import DEFAULT_QUEUE, summarize_rows, validate_queue, expand_queue
from scripts.uxfd_low_tier_source_audit import (
    LowTierSourceAuditReport,
    evaluate_low_tier_source_audit,
)
from scripts.uxfd_recent_work_gate import RecentWorkGateReport, evaluate_recent_work_gate


GOAL_DIR = Path("paper/UXFD_paper/goal")
CLAUDE_TEAM_DIR = Path(".codex/claude-team-runs/20260511-uxfd-ieee-trans-review")
DEFAULT_ARTIFACT_ROOT = Path("paper/UXFD_paper/results/accepted_runs")
GOAL_CLARITY_AUDIT = Path("paper/UXFD_paper/results/goal_clarity_audit_current.md")
COMMIT_RECOVERY_PLAN = Path("paper/UXFD_paper/results/commit_recovery_plan.md")
PAPER07_GOAL = GOAL_DIR / "07_tii_operator_attention.md"
PAPER07_REJECTION_CONTRACT = Path(
    "paper/UXFD_paper/TII_operator_attention/submission_prep/"
    "rejection_recovery_contract.md"
)
PAPER07_REJECTION_NEEDLES = (
    "Rejection-Recovery Focus",
    "Dynamic Sparse Operator Attention v2",
    "reviewer-response style trace",
    "must not use SOTA",
    "paper remains not submission-ready",
    "Q0 preflight",
)
REQUIRED_GOAL_FILES = (
    "README.md",
    "00_overall_goal.md",
    "01_explainable_fd_toolkit.md",
    "02_1d2d_fusion.md",
    "03_llm_explainable_fd_toolkit.md",
    "04_moe_explainable.md",
    "05_fuzzy_xfd.md",
    "06_neuralsymbolic_theory.md",
    "07_tii_operator_attention.md",
    "08_recent_work_citation_readme.md",
    "09_gpu_execution_queue.yaml",
    "99_submission_readiness_matrix.md",
)
MATRIX_ACCEPTED_EVIDENCE_STATUSES = frozenset(
    {
        "accepted_gpu_and_artifacts",
        "accepted_exact_artifacts",
        "accepted_representative_artifacts",
    }
)


@dataclass(frozen=True)
class PaperSubmissionGate:
    paper_id: str
    matrix_path: str
    submission_ready: bool
    baselines: int
    ablations: int
    strict_blockers: Tuple[str, ...]


@dataclass(frozen=True)
class SubmissionGateReport:
    ready: bool
    papers: Tuple[PaperSubmissionGate, ...]
    blockers: Tuple[str, ...]
    next_actions: Tuple[Mapping[str, str], ...]
    objective_checklist: Tuple[Mapping[str, str], ...]
    artifact_gate_accepted: bool
    artifact_gate_root: str
    artifact_gate_records: int
    artifact_gate_blockers: Tuple[str, ...]
    recent_work_policy_ready: bool
    recent_work_evidence_ready: bool
    recent_work_matrix_rows: int
    recent_work_blockers: Tuple[str, ...]
    low_tier_source_ready: bool
    low_tier_source_findings: int
    low_tier_source_blocker_count: int
    low_tier_source_triage_count: int
    low_tier_source_blockers: Tuple[str, ...]
    queue_can_execute: bool
    queue_resource_reason: str
    queue_summary: Mapping[str, Any]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _file_contains_all(path: Path, needles: Sequence[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(needle in text for needle in needles)


def _paper07_rejection_recovery_ready() -> bool:
    return _file_contains_all(
        PAPER07_GOAL,
        PAPER07_REJECTION_NEEDLES[:3],
    ) and _file_contains_all(
        PAPER07_REJECTION_CONTRACT,
        PAPER07_REJECTION_NEEDLES[3:],
    )


def _matrix_paths(queue_path: Path) -> Tuple[Path, ...]:
    queue = _load_yaml(queue_path)
    return tuple(Path(item["matrix_path"]) for item in queue["paper_queue"])


def _next_actions(queue_path: Path) -> Tuple[Mapping[str, str], ...]:
    queue = _load_yaml(queue_path)
    actions: List[Mapping[str, str]] = []
    for item in queue.get("paper_queue", []):
        paper_id = str(item.get("paper_id", ""))
        actions.append(
            {
                "queue_id": str(item.get("queue_id", "")),
                "paper_id": paper_id,
                "goal_file": str(item.get("goal_file", "")),
                "matrix_path": str(item.get("matrix_path", "")),
                "base_config": str(item.get("base_config", "")),
                "priority_reason": str(item.get("priority_reason", "")),
                "unblock_condition": str(item.get("unblock_condition", "")),
            }
        )
    return tuple(actions)


def _matrix_unaccepted_evidence_entries(matrix: Mapping[str, Any]) -> int:
    entries: List[Mapping[str, Any]] = []
    proposed = matrix.get("proposed")
    if isinstance(proposed, Mapping):
        entries.append(proposed)
    for phase in ("baselines", "ablations"):
        entries.extend(
            item for item in matrix.get(phase, ()) if isinstance(item, Mapping)
        )
    return sum(
        str(entry.get("accepted_evidence_status", ""))
        not in MATRIX_ACCEPTED_EVIDENCE_STATUSES
        for entry in entries
    )


def _objective_checklist(
    papers: Sequence[PaperSubmissionGate],
    queue_path: Path,
    artifact_report: ArtifactGateReport,
    recent_report: RecentWorkGateReport,
    low_tier_report: LowTierSourceAuditReport,
) -> Tuple[Mapping[str, str], ...]:
    items: List[Mapping[str, str]] = []
    for filename in REQUIRED_GOAL_FILES:
        path = GOAL_DIR / filename
        items.append(
            {
                "requirement": f"goal file {filename}",
                "evidence": str(path),
                "status": "met" if path.exists() else "missing",
            }
        )

    task_spec = CLAUDE_TEAM_DIR / "TASK_SPEC.md"
    launch_log = CLAUDE_TEAM_DIR / "LAUNCH_LOG.md"
    codex_subagent_launch = CLAUDE_TEAM_DIR / "CODEX_SUBAGENT_LAUNCH.md"
    items.extend(
        [
            {
                "requirement": "Claude Code Team task spec",
                "evidence": str(task_spec),
                "status": "met" if task_spec.exists() else "missing",
            },
            {
                "requirement": "Claude Code Team launch/block log",
                "evidence": str(launch_log),
                "status": "met" if launch_log.exists() else "missing",
            },
            {
                "requirement": "Codex xhigh subagent launch log",
                "evidence": str(codex_subagent_launch),
                "status": "met" if codex_subagent_launch.exists() else "missing",
            },
            {
                "requirement": "seven paper-local matrices",
                "evidence": ",".join(paper.matrix_path for paper in papers),
                "status": "met" if len(papers) == 7 else "missing",
            },
            {
                "requirement": "6+ baselines and 6+ ablations per paper",
                "evidence": "submission_prep/baseline_ablation_matrix.yaml",
                "status": (
                    "met"
                    if len(papers) == 7
                    and all(paper.baselines >= 6 and paper.ablations >= 6 for paper in papers)
                    else "missing"
                ),
            },
            {
                "requirement": "machine-readable GPU queue",
                "evidence": str(queue_path),
                "status": "met" if queue_path.exists() else "missing",
            },
            {
                "requirement": "goal clarity audit report",
                "evidence": str(GOAL_CLARITY_AUDIT),
                "status": "met" if GOAL_CLARITY_AUDIT.exists() else "missing",
            },
            {
                "requirement": "commit recovery plan",
                "evidence": str(COMMIT_RECOVERY_PLAN),
                "status": "met" if COMMIT_RECOVERY_PLAN.exists() else "missing",
            },
            {
                "requirement": "Paper07 rejection-recovery innovation contract",
                "evidence": f"{PAPER07_GOAL},{PAPER07_REJECTION_CONTRACT}",
                "status": "met" if _paper07_rejection_recovery_ready() else "not_met",
            },
            {
                "requirement": "TOP recent-work policy and paper-local matrix coverage",
                "evidence": "scripts.uxfd_recent_work_gate",
                "status": "met" if recent_report.policy_ready else "not_met",
            },
            {
                "requirement": "low-tier source hygiene",
                "evidence": "paper/UXFD_paper/results/low_tier_source_audit.md",
                "status": "met" if low_tier_report.ready else "not_met",
            },
            {
                "requirement": "TOP representative accepted artifacts",
                "evidence": str(GOAL_DIR / "09_gpu_execution_queue.yaml"),
                "status": "met" if recent_report.evidence_ready else "not_met",
            },
            {
                "requirement": "accepted run artifact metadata",
                "evidence": artifact_report.artifact_root,
                "status": "met" if artifact_report.accepted else "not_met",
            },
            {
                "requirement": "submission readiness achieved",
                "evidence": "all paper matrices submission_ready",
                "status": (
                    "not_met"
                    if papers and not all(paper.submission_ready for paper in papers)
                    else "met"
                ),
            },
        ]
    )
    return tuple(items)


def evaluate_submission_gate(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> SubmissionGateReport:
    papers: List[PaperSubmissionGate] = []
    blockers: List[str] = []

    for matrix_path in _matrix_paths(queue_path):
        if not matrix_path.exists():
            blockers.append(f"missing matrix: {matrix_path}")
            continue
        matrix = _load_yaml(matrix_path)
        paper_id = str(matrix.get("paper_id", matrix_path.parent.parent.name))
        strict_blockers = tuple(str(item) for item in matrix.get("strict_blockers", ()))
        baselines = len(matrix.get("baselines", ()))
        ablations = len(matrix.get("ablations", ()))
        submission_ready = bool(matrix.get("submission_ready"))
        papers.append(
            PaperSubmissionGate(
                paper_id=paper_id,
                matrix_path=str(matrix_path),
                submission_ready=submission_ready,
                baselines=baselines,
                ablations=ablations,
                strict_blockers=strict_blockers,
            )
        )
        if baselines < 6:
            blockers.append(f"{paper_id}: fewer than six baselines")
        if ablations < 6:
            blockers.append(f"{paper_id}: fewer than six ablations")
        if not submission_ready:
            blockers.append(f"{paper_id}: submission_ready is false")
        else:
            unaccepted_entries = _matrix_unaccepted_evidence_entries(matrix)
            if unaccepted_entries:
                blockers.append(
                    f"{paper_id}: submission_ready true but {unaccepted_entries} "
                    "proposed/baseline/ablation evidence entries are not accepted"
                )
        if strict_blockers:
            blockers.append(f"{paper_id}: {len(strict_blockers)} strict blockers remain")

    queue_validation = validate_queue(queue_path)
    queue_rows = expand_queue(queue_path)
    if not queue_validation.can_execute:
        blockers.append(f"gpu queue blocked: {queue_validation.resource_reason}")
    blockers.extend(queue_validation.structural_issues)

    artifact_report = evaluate_artifact_gate(
        artifact_root,
        queue_path=queue_path,
        require_queue_coverage=True,
    )
    if not artifact_report.accepted:
        blockers.append(
            f"artifact gate blocked: {len(artifact_report.blockers)} blockers under "
            f"{artifact_report.artifact_root}"
        )
    recent_report = evaluate_recent_work_gate(queue_path=queue_path)
    if not recent_report.policy_ready:
        blockers.append(
            f"recent-work policy blocked: {len(recent_report.policy_blockers)} blockers"
        )
    if not recent_report.evidence_ready:
        blockers.append(
            f"recent-work evidence blocked: {len(recent_report.evidence_blockers)} "
            "TOP representative blockers"
        )
    low_tier_report = evaluate_low_tier_source_audit()
    if not low_tier_report.ready:
        blockers.append(
            "low-tier source hygiene blocked: "
            f"{low_tier_report.blocker_count} blocker references and "
            f"{low_tier_report.triage_count} triage markers"
        )
    if not _paper07_rejection_recovery_ready():
        blockers.append("Paper07 rejection-recovery innovation contract blocked")

    ready = not blockers and len(papers) == 7 and all(item.submission_ready for item in papers)
    return SubmissionGateReport(
        ready=ready,
        papers=tuple(papers),
        blockers=tuple(blockers),
        next_actions=_next_actions(queue_path),
        objective_checklist=_objective_checklist(
            papers,
            queue_path,
            artifact_report,
            recent_report,
            low_tier_report,
        ),
        artifact_gate_accepted=artifact_report.accepted,
        artifact_gate_root=artifact_report.artifact_root,
        artifact_gate_records=len(artifact_report.records),
        artifact_gate_blockers=artifact_report.blockers,
        recent_work_policy_ready=recent_report.policy_ready,
        recent_work_evidence_ready=recent_report.evidence_ready,
        recent_work_matrix_rows=len(recent_report.matrix_coverage),
        recent_work_blockers=recent_report.blockers,
        low_tier_source_ready=low_tier_report.ready,
        low_tier_source_findings=len(low_tier_report.findings),
        low_tier_source_blocker_count=low_tier_report.blocker_count,
        low_tier_source_triage_count=low_tier_report.triage_count,
        low_tier_source_blockers=low_tier_report.blockers,
        queue_can_execute=queue_validation.can_execute,
        queue_resource_reason=queue_validation.resource_reason,
        queue_summary=summarize_rows(queue_rows),
    )


def build_payload(report: SubmissionGateReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: SubmissionGateReport) -> str:
    lines = [
        "# UXFD Submission Gate",
        "",
        f"- Ready: `{report.ready}`",
        f"- Queue can execute: `{report.queue_can_execute}`",
        f"- Queue resource reason: {report.queue_resource_reason}",
        f"- Artifact gate accepted: `{report.artifact_gate_accepted}`",
        f"- Artifact gate records: `{report.artifact_gate_records}`",
        f"- Recent-work policy ready: `{report.recent_work_policy_ready}`",
        f"- Recent-work evidence ready: `{report.recent_work_evidence_ready}`",
        f"- Recent-work matrix rows: `{report.recent_work_matrix_rows}`",
        f"- Low-tier source hygiene ready: `{report.low_tier_source_ready}`",
        f"- Low-tier source blockers: `{report.low_tier_source_blocker_count}`",
        f"- Low-tier source triage markers: `{report.low_tier_source_triage_count}`",
        f"- Blocking findings: `{len(report.blockers)}`",
        f"- Queue dry-run entries: `{report.queue_summary['total']}`",
        "",
        "| Paper | Ready | Baselines | Ablations | Strict blockers |",
        "|---|---:|---:|---:|---:|",
    ]
    for paper in report.papers:
        lines.append(
            f"| `{paper.paper_id}` | `{paper.submission_ready}` | "
            f"{paper.baselines} | {paper.ablations} | {len(paper.strict_blockers)} |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Next Actions", ""])
    for action in report.next_actions:
        lines.append(
            f"- `{action['queue_id']}` `{action['paper_id']}`: "
            f"{action['unblock_condition']}"
        )
    lines.extend(["", "## Objective Checklist", ""])
    for item in report.objective_checklist:
        lines.append(f"- `{item['status']}` {item['requirement']}: {item['evidence']}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate UXFD submission readiness gate")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_submission_gate(args.queue, args.artifact_root)
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")

    if report.ready or args.allow_not_ready:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

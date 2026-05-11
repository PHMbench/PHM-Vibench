from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_artifact_gate import ArtifactGateReport, evaluate_artifact_gate
from scripts.uxfd_gpu_queue import DEFAULT_QUEUE, summarize_rows, validate_queue, expand_queue


GOAL_DIR = Path("paper/UXFD_paper/goal")
CLAUDE_TEAM_DIR = Path(".codex/claude-team-runs/20260511-uxfd-ieee-trans-review")
DEFAULT_ARTIFACT_ROOT = Path("paper/UXFD_paper/results/accepted_runs")
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
    queue_can_execute: bool
    queue_resource_reason: str
    queue_summary: Mapping[str, Any]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _objective_checklist(
    papers: Sequence[PaperSubmissionGate],
    queue_path: Path,
    artifact_report: ArtifactGateReport,
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

    ready = not blockers and len(papers) == 7 and all(item.submission_ready for item in papers)
    return SubmissionGateReport(
        ready=ready,
        papers=tuple(papers),
        blockers=tuple(blockers),
        next_actions=_next_actions(queue_path),
        objective_checklist=_objective_checklist(papers, queue_path, artifact_report),
        artifact_gate_accepted=artifact_report.accepted,
        artifact_gate_root=artifact_report.artifact_root,
        artifact_gate_records=len(artifact_report.records),
        artifact_gate_blockers=artifact_report.blockers,
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

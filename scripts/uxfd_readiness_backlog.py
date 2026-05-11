from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from scripts.uxfd_submission_gate import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_QUEUE,
    evaluate_submission_gate,
)
from scripts.uxfd_submodule_dirty_triage import evaluate_dirty_triage


DEFAULT_OUTPUT = Path("paper/UXFD_paper/results/readiness_backlog.md")


@dataclass(frozen=True)
class BacklogItem:
    item_id: str
    priority: int
    scope: str
    category: str
    blocker: str
    next_action: str
    evidence: str


@dataclass(frozen=True)
class ReadinessBacklogReport:
    ready: bool
    open_items: int
    items: Tuple[BacklogItem, ...]


def _paper_action_map(next_actions: Sequence[Mapping[str, str]]) -> Mapping[str, Mapping[str, str]]:
    return {str(item.get("paper_id", "")): item for item in next_actions}


def evaluate_readiness_backlog(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ReadinessBacklogReport:
    submission = evaluate_submission_gate(queue_path=queue_path, artifact_root=artifact_root)
    dirty = evaluate_dirty_triage()
    paper_actions = _paper_action_map(submission.next_actions)
    items: List[BacklogItem] = []

    if not submission.queue_can_execute:
        items.append(
            BacklogItem(
                item_id="Q0-GPU-PREFLIGHT",
                priority=0,
                scope="cross-paper",
                category="gpu-preflight",
                blocker=submission.queue_resource_reason,
                next_action=(
                    "Restore local GPU visibility, then require `nvidia-smi -L` and "
                    "PyTorch CUDA to show RTX 4090 devices 0 and 1 before launching shards."
                ),
                evidence=str(queue_path),
            )
        )

    if not submission.artifact_gate_accepted:
        items.append(
            BacklogItem(
                item_id="Q0-ARTIFACT-COVERAGE",
                priority=1,
                scope="cross-paper",
                category="accepted-artifacts",
                blocker=(
                    f"{len(submission.artifact_gate_blockers)} artifact blockers; "
                    f"records={submission.artifact_gate_records}"
                ),
                next_action=(
                    "After real runs finish, promote filled `run_meta.yaml`, logs, metrics, "
                    "and configs under accepted_runs and rerun artifact gate with queue coverage."
                ),
                evidence=submission.artifact_gate_root,
            )
        )

    for index, paper in enumerate(submission.papers, start=1):
        action = paper_actions.get(paper.paper_id, {})
        base_priority = 10 + index
        if paper.paper_id == "TII_operator_attention":
            base_priority = 2
        next_action = action.get("unblock_condition") or "Resolve strict blockers in the paper matrix."
        for blocker_index, blocker in enumerate(paper.strict_blockers, start=1):
            items.append(
                BacklogItem(
                    item_id=f"{paper.paper_id}-B{blocker_index:02d}",
                    priority=base_priority,
                    scope=paper.paper_id,
                    category="paper-strict-blocker",
                    blocker=blocker,
                    next_action=next_action,
                    evidence=paper.matrix_path,
                )
            )

    for summary in dirty.summaries:
        categories = ", ".join(
            f"{name}={count}" for name, count in sorted(summary.categories.items())
        )
        items.append(
            BacklogItem(
                item_id=f"DIRTY-{Path(summary.submodule).name}",
                priority=90,
                scope=summary.submodule,
                category="submodule-dirty-review",
                blocker=(
                    f"{summary.total} dirty entries "
                    f"({summary.modified} modified, {summary.untracked} untracked): {categories}"
                ),
                next_action=(
                    "Review with the owning paper owner. Commit only intentional source/docs; "
                    "promote result artifacts only through the accepted artifact gate."
                ),
                evidence="paper/UXFD_paper/results/submodule_dirty_triage.md",
            )
        )

    sorted_items = tuple(sorted(items, key=lambda item: (item.priority, item.item_id)))
    return ReadinessBacklogReport(
        ready=not sorted_items and submission.ready,
        open_items=len(sorted_items),
        items=sorted_items,
    )


def build_payload(report: ReadinessBacklogReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: ReadinessBacklogReport) -> str:
    lines = [
        "# UXFD Readiness Backlog",
        "",
        "Status: execution backlog only. This file is not accepted experiment evidence.",
        "",
        f"- Ready: `{report.ready}`",
        f"- Open items: `{report.open_items}`",
        "",
        "| Priority | Item | Scope | Category | Blocker | Next action | Evidence |",
        "|---:|---|---|---|---|---|---|",
    ]
    for item in report.items:
        blocker = item.blocker.replace("|", "\\|")
        next_action = item.next_action.replace("|", "\\|")
        lines.append(
            f"| {item.priority} | `{item.item_id}` | `{item.scope}` | `{item.category}` | "
            f"{blocker} | {next_action} | `{item.evidence}` |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the UXFD submission readiness backlog")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_readiness_backlog(queue_path=args.queue, artifact_root=args.artifact_root)
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

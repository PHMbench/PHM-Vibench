from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from scripts.uxfd_objective_audit import PARENT_GOAL_CHECKPOINT_PATHS
from scripts.uxfd_owner_review_gate import APPROVED_DECISION_STATUS, DEFAULT_DECISION_FILE
from scripts.uxfd_recent_work_gate import evaluate_recent_work_gate
from scripts.uxfd_submission_gate import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_QUEUE,
    evaluate_submission_gate,
)
from scripts.uxfd_submodule_dirty_triage import (
    DO_NOT_AUTO_COMMIT,
    OWNER_REVIEW_DECISION_TEMPLATE,
    OWNER_REVIEW_RECOMMENDATIONS,
    evaluate_dirty_triage,
)


DEFAULT_OUTPUT = Path("paper/UXFD_paper/results/readiness_backlog.md")
COMMIT_RECOVERY_PLAN = Path("paper/UXFD_paper/results/commit_recovery_plan.md")
PAPER02_SUBMODULE = Path("paper/UXFD_paper/1D-2D_fusion_explainable")
PAPER02_PLANNING_FILES = (
    Path("plan/EXPERIMENT_PLAN_补充.md"),
    Path("program.md"),
)


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


def _git_status_lines_for_paths(paths: Sequence[Path]) -> Tuple[str, ...]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(path) for path in paths)],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def _git_status_lines_for_submodule_paths(
    submodule_path: Path,
    paths: Sequence[Path],
) -> Tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(submodule_path),
            "status",
            "--porcelain",
            "--",
            *(str(path) for path in paths),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line.strip())


def evaluate_readiness_backlog(
    queue_path: Path = DEFAULT_QUEUE,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ReadinessBacklogReport:
    submission = evaluate_submission_gate(queue_path=queue_path, artifact_root=artifact_root)
    recent_work = evaluate_recent_work_gate(queue_path=queue_path)
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
                    "and configs under accepted_runs. Require integer seed/batch_size, "
                    "positive runtime, enumerated precision, accepted_same_protocol "
                    "evidence_level, hashed "
                    "preprocessing_signature, numeric metrics, `source_tree_status: clean`, "
                    "clean SHA provenance, at least the paper-specific `minimum_seeds` "
                    "distinct accepted seeds for each covered queue item, and matched-seed "
                    "aggregate statistics before rerunning artifact and SOTA gates with "
                    "queue coverage."
                ),
                evidence=submission.artifact_gate_root,
            )
        )

    if not submission.sota_gate_ready:
        items.append(
            BacklogItem(
                item_id="Q0-SOTA-AGGREGATE",
                priority=2,
                scope="cross-paper",
                category="sota-aggregate-evidence",
                blocker=(
                    f"{len(submission.sota_gate_blockers)} SOTA blockers; "
                    f"records={submission.sota_gate_records}"
                ),
                next_action=(
                    "After accepted run coverage exists, build one "
                    "`sota_aggregate.yaml` per paper with matched seed sets, "
                    "six baseline comparators, runnable TOP representative scope, "
                    "mean/std/95% CI, effect size or paired-test evidence, and "
                    "`accepted_run_refs` pointing to existing accepted_runs "
                    "`run_meta.yaml` files."
                ),
                evidence=submission.sota_gate_root,
            )
        )

    try:
        paper02_planning_status = _git_status_lines_for_submodule_paths(
            PAPER02_SUBMODULE,
            PAPER02_PLANNING_FILES,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        paper02_planning_status = (f"unverified:{exc.__class__.__name__}",)
    if paper02_planning_status:
        blocker = (
            "targeted Paper02 planning status unverified: "
            f"{paper02_planning_status[0].split(':', 1)[1]}"
            if paper02_planning_status[0].startswith("unverified:")
            else (
                f"{len(paper02_planning_status)} targeted Paper02 planning files "
                "are edited but uncommitted"
            )
        )
        items.append(
            BacklogItem(
                item_id="Q0-PAPER02-PLANNING-COMMIT",
                priority=3,
                scope="1D-2D_fusion_explainable",
                category="commit-recovery",
                blocker=blocker,
                next_action=(
                    "Run commit recovery Phase 1: review and commit only "
                    "`plan/EXPERIMENT_PLAN_补充.md` and `program.md` inside the "
                    "Paper02 submodule."
                ),
                evidence=str(COMMIT_RECOVERY_PLAN),
            )
        )

    try:
        parent_checkpoint_status = _git_status_lines_for_paths(PARENT_GOAL_CHECKPOINT_PATHS)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        parent_checkpoint_status = (f"unverified:{exc.__class__.__name__}",)
    if parent_checkpoint_status:
        blocker = (
            "parent goal/control checkpoint status unverified: "
            f"{parent_checkpoint_status[0].split(':', 1)[1]}"
            if parent_checkpoint_status[0].startswith("unverified:")
            else f"{len(parent_checkpoint_status)} parent goal/control paths are dirty"
        )
        items.append(
            BacklogItem(
                item_id="Q0-PARENT-GOAL-CHECKPOINT-COMMIT",
                priority=4,
                scope="cross-paper",
                category="commit-recovery",
                blocker=blocker,
                next_action=(
                    "Run commit recovery Phases 2-4: sync parent reports, commit the "
                    "parent checkpoint, then regenerate objective audit outputs."
                ),
                evidence=str(COMMIT_RECOVERY_PLAN),
            )
        )

    if not submission.low_tier_source_ready:
        items.append(
            BacklogItem(
                item_id="Q0-SOURCE-HYGIENE",
                priority=1,
                scope="cross-paper",
                category="low-tier-source-hygiene",
                blocker=(
                    f"{submission.low_tier_source_blocker_count} blocker references; "
                    f"{submission.low_tier_source_triage_count} triage markers"
                ),
                next_action=(
                    "Remove or replace blocker references in active manuscripts and bibliography "
                    "entrypoints with TOP-journal or top-conference sources before any "
                    "submission-ready claim."
                ),
                evidence="paper/UXFD_paper/results/low_tier_source_audit.md",
            )
        )

    for binding in recent_work.bindings:
        if binding.evidence_ready:
            continue
        proxy_entries = ", ".join(binding.local_proxy_matrix_entries)
        items.append(
            BacklogItem(
                item_id=binding.binding_id,
                priority=5,
                scope=binding.paper_id,
                category="top-representative-evidence",
                blocker=(
                    f"{binding.external_work_id} binding is {binding.status}; "
                    f"local proxy entries={proxy_entries}; "
                    f"exact_status={binding.exact_reproduction_status}"
                ),
                next_action=(
                    "After Q0 GPU preflight passes, run and promote accepted artifacts for "
                    "the listed local proxy matrix entries. Keep the claim representative-only "
                    "unless exact external code/config evidence is integrated."
                ),
                evidence="paper/UXFD_paper/results/recent_work_gate_current.md",
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
        owner_review_pending = sum(
            1
            for entry in dirty.entries
            if entry.submodule == summary.submodule
            and entry.recommended_action == DO_NOT_AUTO_COMMIT
        )
        items.append(
            BacklogItem(
                item_id=f"DIRTY-{Path(summary.submodule).name}",
                priority=90,
                scope=summary.submodule,
                category="submodule-dirty-review",
                blocker=(
                    f"{summary.total} dirty entries "
                    f"({summary.modified} modified, {summary.untracked} untracked): "
                    f"{categories}; owner_review_pending={owner_review_pending}"
                ),
                next_action=(
                    "Resolve the `pending_owner_review` rows in "
                    "`paper/UXFD_paper/results/submodule_dirty_triage.json` with the owning "
                    "paper owner after reading "
                    f"`{OWNER_REVIEW_RECOMMENDATIONS}`. Copy "
                    f"`{OWNER_REVIEW_DECISION_TEMPLATE}` to "
                    f"`{DEFAULT_DECISION_FILE}`, change top-level `status` to "
                    f"`{APPROVED_DECISION_STATUS}`, replace every pending decision with an "
                    "allowed owner decision, and use a non-placeholder reviewer plus ISO "
                    "`YYYY-MM-DD` review date. Validate decisions with "
                    "`python -m scripts.uxfd_owner_review_gate`. Commit only intentional "
                    "source/docs; promote result artifacts only through the accepted "
                    "artifact gate."
                ),
                evidence=(
                    "paper/UXFD_paper/results/submodule_dirty_triage.md,"
                    "paper/UXFD_paper/results/submodule_dirty_triage.json,"
                    f"{OWNER_REVIEW_RECOMMENDATIONS},"
                    f"{OWNER_REVIEW_DECISION_TEMPLATE},"
                    f"{DEFAULT_DECISION_FILE}"
                ),
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

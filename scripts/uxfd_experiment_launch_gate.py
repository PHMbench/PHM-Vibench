from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from scripts.uxfd_gpu_queue import (
    DEFAULT_QUEUE,
    expand_queue,
    run_live_preflight,
    summarize_rows,
    validate_queue,
)
from scripts.uxfd_owner_review_gate import evaluate_owner_review_gate


@dataclass(frozen=True)
class ExperimentLaunchGateReport:
    ready: bool
    owner_review_ready: bool
    owner_review_source_path: str
    owner_review_pending_records: int
    queue_static_can_execute: bool
    queue_resource_reason: str
    queue_structural_issues: Tuple[str, ...]
    queue_summary: Mapping[str, Any]
    live_preflight_required: bool
    live_preflight_accepted: Optional[bool]
    live_preflight_reason: str
    blockers: Tuple[str, ...]


def evaluate_experiment_launch_gate(
    queue_path: Path = DEFAULT_QUEUE,
    require_live_preflight: bool = True,
) -> ExperimentLaunchGateReport:
    owner_report = evaluate_owner_review_gate()
    rows = expand_queue(queue_path)
    queue_validation = validate_queue(queue_path)
    live_preflight = run_live_preflight() if require_live_preflight else None

    blockers = []
    if not owner_report.ready:
        blockers.append(
            "owner-review gate not ready: "
            f"pending_records={owner_report.pending_records}, "
            f"blockers={len(owner_report.blockers)}"
        )
    if not queue_validation.can_execute:
        blockers.append(
            "gpu queue static gate not executable: "
            f"{queue_validation.resource_reason}"
        )
    if queue_validation.structural_issues:
        blockers.append(
            f"gpu queue structural issues: {len(queue_validation.structural_issues)}"
        )
    if require_live_preflight and live_preflight is not None and not live_preflight.accepted:
        blockers.append(f"live GPU preflight not accepted: {live_preflight.reason}")

    return ExperimentLaunchGateReport(
        ready=not blockers,
        owner_review_ready=owner_report.ready,
        owner_review_source_path=owner_report.source_path,
        owner_review_pending_records=owner_report.pending_records,
        queue_static_can_execute=queue_validation.can_execute,
        queue_resource_reason=queue_validation.resource_reason,
        queue_structural_issues=queue_validation.structural_issues,
        queue_summary=summarize_rows(rows),
        live_preflight_required=require_live_preflight,
        live_preflight_accepted=(
            live_preflight.accepted if live_preflight is not None else None
        ),
        live_preflight_reason=live_preflight.reason if live_preflight is not None else "",
        blockers=tuple(blockers),
    )


def build_payload(report: ExperimentLaunchGateReport) -> Mapping[str, Any]:
    return asdict(report)


def render_markdown(report: ExperimentLaunchGateReport) -> str:
    lines = [
        "# UXFD Experiment Launch Gate",
        "",
        "Status: experiment-launch authorization only. This report is not accepted "
        "experiment evidence and not submission readiness.",
        "",
        f"- Ready: `{report.ready}`",
        f"- Owner-review gate ready: `{report.owner_review_ready}`",
        f"- Owner-review source: `{report.owner_review_source_path}`",
        f"- Owner-review pending records: `{report.owner_review_pending_records}`",
        f"- GPU queue static gate executable: `{report.queue_static_can_execute}`",
        f"- GPU queue resource reason: {report.queue_resource_reason}",
        f"- GPU queue structural issues: `{len(report.queue_structural_issues)}`",
        f"- Queue dry-run entries: `{report.queue_summary['total']}`",
        f"- Live preflight required: `{report.live_preflight_required}`",
        f"- Live preflight accepted: `{report.live_preflight_accepted}`",
        f"- Live preflight reason: {report.live_preflight_reason or '-'}",
        "",
        "## Blockers",
        "",
    ]
    if report.blockers:
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Required Gates",
            "",
            "This gate intentionally does not require accepted run artifacts, SOTA "
            "aggregates, or the final submission gate; those are produced after "
            "successful experiment execution.",
            "",
            "```bash",
            "python -m scripts.uxfd_owner_review_gate --format markdown",
            "python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate UXFD experiment launch gate")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-live-preflight", action="store_true")
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_experiment_launch_gate(
        queue_path=args.queue,
        require_live_preflight=not args.skip_live_preflight,
    )
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

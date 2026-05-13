from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.uxfd_artifact_gate import evaluate_artifact_gate
from scripts.uxfd_gpu_queue import DEFAULT_QUEUE, expand_queue, summarize_rows, validate_queue
from scripts.uxfd_objective_audit import evaluate_objective_audit
from scripts.uxfd_recent_work_gate import evaluate_recent_work_gate
from scripts.uxfd_submission_gate import DEFAULT_ARTIFACT_ROOT, evaluate_submission_gate
from scripts.uxfd_submodule_dirty_triage import (
    DO_NOT_AUTO_COMMIT,
    PRESERVE_SESSION,
    PROMOTE_ONLY_THROUGH_GATE,
    evaluate_dirty_triage,
)


DEFAULT_STATUS_DIR = Path("paper/UXFD_paper/goal/status")
LAUNCH_SCRIPT_STATIC_GATE_PATHS = (
    Path("paper/UXFD_paper/results/queue_launch_plan.sh"),
    Path("paper/UXFD_paper/results/queue_launch_shards/gpu0.sh"),
    Path("paper/UXFD_paper/results/queue_launch_shards/gpu1.sh"),
)
LAUNCH_SCRIPT_STATIC_GATE_NEEDLES = (
    "Blocked: static queue validation can_execute=False",
    "exit 2",
)

PAPER_STATUS_FILES: Mapping[str, Tuple[str, str, str]] = {
    "Explainable_FD_Toolkit": (
        "status_01_explainable_fd_toolkit.md",
        "Paper 01 - Explainable FD Toolkit",
        "paper/UXFD_paper/goal/01_explainable_fd_toolkit.md",
    ),
    "1D-2D_fusion_explainable": (
        "status_02_1d2d_fusion.md",
        "Paper 02 - 1D-2D Fusion Explainable FD",
        "paper/UXFD_paper/goal/02_1d2d_fusion.md",
    ),
    "LLM_Explainable_FD_Toolkit": (
        "status_03_llm_explainable_fd_toolkit.md",
        "Paper 03 - LLM Explainable FD Toolkit",
        "paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md",
    ),
    "MOE_explainable": (
        "status_04_moe_explainable.md",
        "Paper 04 - MOE Explainable FD",
        "paper/UXFD_paper/goal/04_moe_explainable.md",
    ),
    "Paper_fuzzy_XFD": (
        "status_05_fuzzy_xfd.md",
        "Paper 05 - Fuzzy-XFD",
        "paper/UXFD_paper/goal/05_fuzzy_xfd.md",
    ),
    "Neuralsymbolic_theory": (
        "status_06_neuralsymbolic_theory.md",
        "Paper 06 - Neural-Symbolic Theory",
        "paper/UXFD_paper/goal/06_neuralsymbolic_theory.md",
    ),
    "TII_operator_attention": (
        "status_07_tii_operator_attention.md",
        "Paper 07 - TII Operator Attention",
        "paper/UXFD_paper/goal/07_tii_operator_attention.md",
    ),
}

PAPER_ORDER = (
    "Explainable_FD_Toolkit",
    "1D-2D_fusion_explainable",
    "LLM_Explainable_FD_Toolkit",
    "MOE_explainable",
    "Paper_fuzzy_XFD",
    "Neuralsymbolic_theory",
    "TII_operator_attention",
)


def _header(title: str, goal_file: str, generated_on: str) -> List[str]:
    return [
        f"# Status Report: {title}",
        "",
        "Status reports are generated control-plane summaries, not accepted experiment evidence.",
        "",
        f"- Generated: `{generated_on}`",
        f"- Goal file: `{goal_file}`",
        "",
    ]


def _dirty_by_paper(dirty_report: object) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for summary in dirty_report.summaries:
        counts[Path(summary.submodule).name] = summary.total
    return counts


def _dirty_actions_by_submodule(dirty_report: object) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for entry in dirty_report.entries:
        submodule_counts = counts.setdefault(
            entry.submodule,
            {
                DO_NOT_AUTO_COMMIT: 0,
                PROMOTE_ONLY_THROUGH_GATE: 0,
                PRESERVE_SESSION: 0,
            },
        )
        if entry.recommended_action not in submodule_counts:
            submodule_counts[entry.recommended_action] = 0
        submodule_counts[entry.recommended_action] += 1
    return counts


def _binding_by_paper(recent_report: object) -> Dict[str, object]:
    return {binding.paper_id: binding for binding in recent_report.bindings}


def _matrix_coverage_by_paper(recent_report: object) -> Dict[str, object]:
    return {coverage.paper_id: coverage for coverage in recent_report.matrix_coverage}


def _first_blockers(blockers: Sequence[str], limit: int = 8) -> Iterable[str]:
    for blocker in blockers[:limit]:
        yield blocker
    if len(blockers) > limit:
        yield f"... {len(blockers) - limit} additional blockers omitted; see gate reports."


def _launch_static_gate_ready() -> bool:
    for path in LAUNCH_SCRIPT_STATIC_GATE_PATHS:
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8")
        if any(needle not in text for needle in LAUNCH_SCRIPT_STATIC_GATE_NEEDLES):
            return False
    return True


def _render_overall(
    generated_on: str,
    objective_report: object,
    submission_report: object,
    artifact_report: object,
    dirty_report: object,
) -> str:
    lines = _header(
        "UXFD Overall Cross-Paper Progress",
        "paper/UXFD_paper/goal/00_overall_goal.md",
        generated_on,
    )
    lines.extend(
        [
            "## Current Verdict",
            "",
            "- Achieved: `False`",
            f"- Objective audit: `met={objective_report.met}`, "
            f"`not_met={objective_report.not_met}`, `blocked={objective_report.blocked}`",
            f"- Submission gate ready: `{submission_report.ready}`",
            f"- Queue can execute: `{submission_report.queue_can_execute}`",
            f"- Artifact coverage: `{artifact_report.covered_queue_runs}/"
            f"{artifact_report.expected_queue_runs}`",
            f"- Artifact records: `{len(artifact_report.records)}`",
            f"- Dirty submodule entries: `{len(dirty_report.entries)}`",
            "",
            "The project is ready for controlled execution only after local GPUs 0 and 1 "
            "are visible and the accepted artifact gate can be populated with real runs.",
            "",
            "## Paper Matrix",
            "",
            "| Paper | Ready | Baselines | Ablations | Strict Blockers |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for paper in submission_report.papers:
        lines.append(
            f"| `{paper.paper_id}` | `{paper.submission_ready}` | {paper.baselines} | "
            f"{paper.ablations} | {len(paper.strict_blockers)} |"
        )
    lines.extend(["", "## Blocking Findings", ""])
    for blocker in _first_blockers(submission_report.blockers):
        lines.append(f"- {blocker}")
    dirty_actions = _dirty_actions_by_submodule(dirty_report)
    lines.extend(
        [
            "",
            "## Dirty Submodule Owner Review Queue",
            "",
            "Do not auto-commit these entries. Commit only owner-reviewed source/docs; "
            "promote generated or result artifacts only through the accepted artifact gate.",
            "",
            "| Submodule | Owner Review | Artifact Gate Only | Preserve/Ignore |",
            "|---|---:|---:|---:|",
        ]
    )
    for summary in dirty_report.summaries:
        counts = dirty_actions.get(summary.submodule, {})
        lines.append(
            f"| `{summary.submodule}` | {counts.get(DO_NOT_AUTO_COMMIT, 0)} | "
            f"{counts.get(PROMOTE_ONLY_THROUGH_GATE, 0)} | "
            f"{counts.get(PRESERVE_SESSION, 0)} |"
        )
    return "\n".join(lines) + "\n"


def _render_paper(
    paper_id: str,
    generated_on: str,
    paper_report: object,
    artifact_report: object,
    recent_report: object,
    dirty_counts: Mapping[str, int],
) -> str:
    filename, title, goal_file = PAPER_STATUS_FILES[paper_id]
    del filename
    binding = _binding_by_paper(recent_report).get(paper_id)
    coverage = _matrix_coverage_by_paper(recent_report).get(paper_id)
    artifact_counts = artifact_report.queue_coverage_by_paper.get(
        paper_id,
        {"covered": 0, "expected": 0, "missing": 0},
    )
    lines = _header(title, goal_file, generated_on)
    lines.extend(
        [
            "## Current Verdict",
            "",
            f"- Submission ready: `{paper_report.submission_ready}`",
            f"- Baselines declared: `{paper_report.baselines}`",
            f"- Ablations declared: `{paper_report.ablations}`",
            f"- Strict blockers: `{len(paper_report.strict_blockers)}`",
            f"- Accepted artifact coverage: `{artifact_counts.get('covered', 0)}/"
            f"{artifact_counts.get('expected', 0)}`",
            f"- Dirty submodule entries: `{dirty_counts.get(paper_id, 0)}`",
        ]
    )
    if coverage is not None:
        lines.extend(
            [
                f"- TOP recent-work methods in matrix: `{coverage.top_count}`",
                f"- Has 2026 TOP method: `{coverage.has_2026}`",
            ]
        )
    if binding is not None:
        lines.extend(
            [
                f"- TOP binding: `{binding.binding_id}` -> `{binding.external_work_id}`",
                f"- TOP evidence ready: `{binding.evidence_ready}`",
                f"- TOP binding status: `{binding.status}`",
            ]
        )
    lines.extend(["", "## Strict Blockers", ""])
    if paper_report.strict_blockers:
        for blocker in paper_report.strict_blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Do not mark this paper submission-ready until same-protocol accepted "
            "baseline, ablation, TOP representative, GPU metadata, and SOTA evidence "
            "are present under the artifact gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_recent_work(generated_on: str, recent_report: object) -> str:
    lines = _header(
        "UXFD TOP Citation Readiness",
        "paper/UXFD_paper/goal/08_recent_work_citation_readme.md",
        generated_on,
    )
    lines.extend(
        [
            "## Current Verdict",
            "",
            f"- Ready: `{recent_report.ready}`",
            f"- Policy ready: `{recent_report.policy_ready}`",
            f"- Evidence ready: `{recent_report.evidence_ready}`",
            f"- Accepted TOP method rows: `{recent_report.accepted_pool_rows}`",
            f"- 2026 TOP IDs: `{len(recent_report.top_2026_ids)}`",
            f"- Low-tier violations in TOP pool: `{len(recent_report.low_tier_violations)}`",
            f"- Evidence blockers: `{len(recent_report.evidence_blockers)}`",
            "",
            "## Paper-Local Exact-Status Scope",
            "",
            "| Paper | TOP Methods | Missing Exact Status | Unscoped Exact Claims | Policy Ready |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for coverage in recent_report.matrix_coverage:
        lines.append(
            f"| `{coverage.paper_id}` | {coverage.top_count} | "
            f"{len(coverage.missing_exact_status_ids)} | "
            f"{len(coverage.unscoped_exact_claim_ids)} | "
            f"`{coverage.policy_ready}` |"
        )
    lines.extend(
        [
            "",
            "## TOP Representative Bindings",
            "",
            "| Binding | Paper | External Work | Status | Evidence Ready |",
            "|---|---|---|---|---:|",
        ]
    )
    for binding in recent_report.bindings:
        lines.append(
            f"| `{binding.binding_id}` | `{binding.paper_id}` | "
            f"`{binding.external_work_id}` | `{binding.status}` | "
            f"`{binding.evidence_ready}` |"
        )
    return "\n".join(lines) + "\n"


def _render_gpu_execution(
    generated_on: str,
    submission_report: object,
    artifact_report: object,
    recent_report: object,
) -> str:
    rows = expand_queue(DEFAULT_QUEUE)
    queue_summary = summarize_rows(rows)
    validation = validate_queue(DEFAULT_QUEUE)
    lines = _header(
        "UXFD GPU Execution Queue",
        "paper/UXFD_paper/goal/09_gpu_execution_queue.yaml",
        generated_on,
    )
    lines.extend(
        [
            "## Current Verdict",
            "",
            f"- Can execute now: `{validation.can_execute}`",
            f"- Resource reason: {validation.resource_reason}",
            f"- Structural issues: `{len(validation.structural_issues)}`",
            f"- Queue dry-run entries: `{queue_summary['total']}`",
            f"- Launchable entries: `{queue_summary['total'] - queue_summary['top_representatives']}`",
            f"- TOP representative entries: `{queue_summary['top_representatives']}`",
            f"- Artifact coverage: `{artifact_report.covered_queue_runs}/"
            f"{artifact_report.expected_queue_runs}`",
            f"- Submission gate ready: `{submission_report.ready}`",
            f"- Static launch gate enabled: `{_launch_static_gate_ready()}`",
            "",
            "## Required Before Q1",
            "",
            "- `nvidia-smi -L` must show local RTX 4090 GPUs 0 and 1.",
            "- PyTorch must report CUDA available with at least two devices.",
            "- Accepted artifacts must fill `run_meta.yaml`, logs, metrics, and configs "
            "with no TODO placeholders.",
            "- `seed` must be a non-negative integer and `batch_size` must be a "
            "positive integer.",
            "- `runtime` must be a positive `HH:MM:SS` duration.",
            "- `precision` must be one of `fp32`, `tf32`, `fp16`, `bf16`, `amp`.",
            "- `preprocessing_signature` must match `sha256:<64 lowercase hex>`.",
            "- `metrics.json` or `metrics.csv` must include at least one numeric metric; "
            "status-only payloads are rejected.",
            "- `git_sha_or_submodule_sha` must be a concrete clean revision without "
            "dirty, modified, unknown, or uncommitted markers.",
            "",
            "## TOP Representative Execution Bindings",
            "",
            "These rows are queue bindings, not accepted evidence. Keep claims "
            "representative-only until exact external code/config evidence is integrated.",
            "",
            "| Binding | Paper | Work | Local Proxy Entries | Exact Status | Status | Evidence Ready |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for binding in recent_report.bindings:
        proxy_entries = ", ".join(binding.local_proxy_matrix_entries)
        exact_status = binding.exact_reproduction_status.replace("|", "\\|")
        lines.append(
            f"| `{binding.binding_id}` | `{binding.paper_id}` | "
            f"`{binding.external_work_id}` | `{proxy_entries}` | "
            f"{exact_status} | `{binding.status}` | `{binding.evidence_ready}` |"
        )
    return "\n".join(lines) + "\n"


def generate_status_reports(
    output_dir: Path = DEFAULT_STATUS_DIR,
    generated_on: Optional[str] = None,
) -> Tuple[Path, ...]:
    generated = generated_on or date.today().isoformat()
    objective_report = evaluate_objective_audit()
    submission_report = evaluate_submission_gate()
    artifact_report = evaluate_artifact_gate(
        DEFAULT_ARTIFACT_ROOT,
        queue_path=DEFAULT_QUEUE,
        require_queue_coverage=True,
    )
    recent_report = evaluate_recent_work_gate()
    dirty_report = evaluate_dirty_triage()
    dirty_counts = _dirty_by_paper(dirty_report)
    paper_reports = {paper.paper_id: paper for paper in submission_report.papers}

    outputs: Dict[str, str] = {
        "status_00_overall.md": _render_overall(
            generated,
            objective_report,
            submission_report,
            artifact_report,
            dirty_report,
        ),
        "status_08_citation_readiness.md": _render_recent_work(generated, recent_report),
        "status_09_gpu_execution.md": _render_gpu_execution(
            generated,
            submission_report,
            artifact_report,
            recent_report,
        ),
    }
    for paper_id in PAPER_ORDER:
        filename = PAPER_STATUS_FILES[paper_id][0]
        outputs[filename] = _render_paper(
            paper_id,
            generated,
            paper_reports[paper_id],
            artifact_report,
            recent_report,
            dirty_counts,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for filename, text in sorted(outputs.items()):
        path = output_dir / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return tuple(written)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate UXFD goal status reports")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STATUS_DIR)
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)

    written = generate_status_reports(args.output_dir, generated_on=args.date)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

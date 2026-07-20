#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


START_MARKER = "<!-- AUTORESEARCH_SUBMISSION_BINDING:START -->"
END_MARKER = "<!-- AUTORESEARCH_SUBMISSION_BINDING:END -->"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_file(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return candidates[-1]


def copy_if_exists(source: Optional[Path], destination: Path) -> Optional[str]:
    if source is None or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def replace_marked_block(path: Path, lines: Iterable[str]) -> None:
    new_block = "\n".join([START_MARKER, *lines, END_MARKER]) + "\n"
    if path.exists():
        original = path.read_text(encoding="utf-8")
        if START_MARKER in original and END_MARKER in original:
            before, rest = original.split(START_MARKER, 1)
            _, after = rest.split(END_MARKER, 1)
            updated = before.rstrip() + "\n\n" + new_block + after.lstrip("\n")
        else:
            updated = original.rstrip() + "\n\n" + new_block
    else:
        updated = new_block
    path.write_text(updated, encoding="utf-8")


def read_ticket_queue(paper_root: Path) -> List[Dict[str, Any]]:
    queue_path = paper_root / "autoresearch" / "ticket_queue.jsonl"
    if not queue_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw_line in queue_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def accepted_ticket_ids(paper_root: Path) -> List[str]:
    return [
        row["ticket_id"]
        for row in read_ticket_queue(paper_root)
        if row.get("status") in {"accepted", "completed"}
    ]


def multi_dataset_sources(paper_root: Path) -> Dict[str, Path]:
    results_root = paper_root / "results" / "autoresearch"
    summary_json = latest_file(results_root, "**/multi_dataset_validation/multi_dataset_validation_metrics_summary.json")
    results_json = Path(load_json(summary_json)["results_file"])
    if not results_json.is_absolute():
        results_json = paper_root / results_json
    summary_md = summary_json.with_name("multi_dataset_validation_summary.md")
    return {
        "summary_json": summary_json,
        "results_json": results_json,
        "summary_md": summary_md,
    }


def stability_sources(paper_root: Path) -> Dict[str, Path]:
    results_root = paper_root / "results" / "autoresearch"
    summary_json = latest_file(results_root, "**/stability_three_seed/stability_metrics_summary.json")
    results_json = Path(load_json(summary_json)["results_file"])
    if not results_json.is_absolute():
        results_json = paper_root / results_json
    summary_md = summary_json.with_name("stability_test_summary.md")
    return {
        "summary_json": summary_json,
        "results_json": results_json,
        "summary_md": summary_md,
    }


def explainability_sources(paper_root: Path) -> List[str]:
    figures_root = paper_root / "results" / "autoresearch"
    figures = sorted(
        figures_root.glob("**/explainability_pack/figures/*"),
        key=lambda item: item.stat().st_mtime,
    )
    return [str(item) for item in figures if item.is_file()]


def explainability_quant_sources(paper_root: Path) -> Dict[str, Any]:
    results_root = paper_root / "results" / "autoresearch"
    summary_json = latest_file(results_root, "**/explainability_quant/explainability_metrics_summary.json")
    summary = load_json(summary_json)
    results_json = Path(summary["results_file"])
    if not results_json.is_absolute():
        results_json = paper_root / results_json
    summary_md = summary_json.with_name("explainability_metrics_summary.md")
    return {
        "summary_json": summary_json,
        "summary": summary,
        "results_json": results_json,
        "summary_md": summary_md,
        "figure_paths": summary.get("figure_paths", []),
    }


def truth_sync_sources(paper_root: Path) -> Dict[str, Any]:
    results_root = paper_root / "results" / "autoresearch"
    summary_json = latest_file(results_root, "**/manuscript_truth_sync/manuscript_truth_sync_summary.json")
    summary = load_json(summary_json)
    summary_md = summary_json.with_name("manuscript_truth_sync_summary.md")
    return {
        "summary_json": summary_json,
        "summary": summary,
        "summary_md": summary_md,
    }


def comparison_sources(paper_root: Path) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for name in ["comparison_moe.log", "comparison_tspn.log", "comparison_operator_attention.log"]:
        path = paper_root / name
        if path.exists():
            sources[name] = str(path)
    return sources


def resolve_artifact_inputs(paper_root: Path, entries: List[str]) -> List[str]:
    resolved: List[str] = []
    seen: set[str] = set()
    for entry in entries:
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = (paper_root / candidate).resolve()
        if candidate.exists():
            key = str(candidate)
            if key not in seen:
                resolved.append(key)
                seen.add(key)
            continue
        for match in sorted(paper_root.rglob(entry)):
            key = str(match)
            if key not in seen:
                resolved.append(key)
                seen.add(key)
    return resolved


def build_cross_dataset(output_dir: Path, paper_root: Path, artifact_inputs: Optional[List[str]] = None) -> Path:
    sources = multi_dataset_sources(paper_root)
    metrics = load_json(sources["summary_json"])
    results = load_json(sources["results_json"])
    resolved_inputs = resolve_artifact_inputs(paper_root, artifact_inputs or [])

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        "metrics_summary": copy_if_exists(sources["summary_json"], output_dir / "multi_dataset_validation_metrics_summary.json"),
        "results_json": copy_if_exists(sources["results_json"], output_dir / sources["results_json"].name),
        "summary_md": copy_if_exists(sources["summary_md"], output_dir / "multi_dataset_validation_summary.md"),
    }

    datasets = metrics.get("successful_datasets") or metrics.get("requested_datasets") or []
    summary = {
        "bound": True,
        "datasets": datasets,
        "success_count": metrics.get("success_count"),
        "failed_count": metrics.get("failed_count"),
        "accuracy": metrics.get("mean_test_acc"),
        "mean_test_acc": metrics.get("mean_test_acc"),
        "generalization_gap": metrics.get("generalization_gap"),
        "source_inputs": resolved_inputs,
        "source_files": copied,
        "successful_runs": [
            {
                "dataset": row.get("dataset"),
                "status": row.get("status"),
                "duration": row.get("duration"),
                "log_file": row.get("log_file"),
            }
            for row in results
        ],
    }
    summary_path = output_dir / "cross_dataset_binding_summary.json"
    dump_json(summary_path, summary)

    lines = [
        "# Cross-Dataset Generalization Binding Snapshot",
        "",
        f"- datasets: {', '.join(str(item) for item in datasets)}",
        f"- success_count: {metrics.get('success_count')}",
        f"- failed_count: {metrics.get('failed_count')}",
        f"- mean_test_acc: {metrics.get('mean_test_acc')}",
        f"- generalization_gap: {metrics.get('generalization_gap')}",
        f"- source_inputs: `{', '.join(resolved_inputs) if resolved_inputs else 'auto-discovered from local artifact candidates'}`",
        "",
        "## Source Artifacts",
        "",
    ]
    for label, source in copied.items():
        if source:
            lines.append(f"- {label}: `{source}`")
    (output_dir / "cross_dataset_binding.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def build_manuscript_binding(output_dir: Path, paper_root: Path, artifact_inputs: Optional[List[str]] = None) -> Path:
    multi = load_json(multi_dataset_sources(paper_root)["summary_json"])
    stability = load_json(stability_sources(paper_root)["summary_json"])
    comparison = comparison_sources(paper_root)
    accepted = accepted_ticket_ids(paper_root)
    resolved_inputs = resolve_artifact_inputs(paper_root, artifact_inputs or [])
    accepted_set = set(accepted)

    explainability_bundle = explainability_sources(paper_root)
    explainability_quant = None
    truth_sync = None
    open_issues: List[str] = []

    try:
        explainability_quant = explainability_quant_sources(paper_root)
    except FileNotFoundError:
        open_issues.append("quantitative explainability metrics are missing")

    try:
        truth_sync = truth_sync_sources(paper_root)
    except FileNotFoundError:
        open_issues.append("manuscript truth sync summary is missing")

    if "1d2d-multi-dataset-validation" not in accepted_set:
        open_issues.append("multi-dataset validation is not accepted in the queue")
    if "1d2d-stability-three-seed" not in accepted_set:
        open_issues.append("three-seed stability is not accepted in the queue")
    if "1d2d-comparison-suite" not in accepted_set:
        open_issues.append("comparison suite is not accepted in the queue")
    if "1d2d-cross-dataset-generalization" not in accepted_set:
        open_issues.append("cross-dataset generalization is still pending accepted evidence")
    if truth_sync and truth_sync["summary"].get("unsupported_claims_remaining", 1) != 0:
        open_issues.append("unsupported manuscript claims remain after truth sync")
    if truth_sync and not truth_sync["summary"].get("all_targets_synced", False):
        open_issues.append("manuscript targets are not fully synchronized")

    internal_binding_ready = not open_issues
    parent_gate_blockers = [
        "parent UXFD accepted-run artifact gate not satisfied",
        "parent UXFD 2x4090 GPU queue not accepted",
        "parent UXFD cross-paper submission gate not passed",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    binding_summary = {
        "bound": True,
        "internal_binding_ready": internal_binding_ready,
        "submission_ready": False,
        "submission_ready_policy": (
            "This script binds paper-local 1D-2D evidence only. External IEEE "
            "submission readiness is controlled by the parent UXFD gate."
        ),
        "parent_gate_blockers": parent_gate_blockers,
        "accuracy": multi.get("mean_test_acc"),
        "accepted_ticket_ids": accepted,
        "datasets": multi.get("successful_datasets") or multi.get("requested_datasets") or [],
        "cross_dataset": {
            "success_count": multi.get("success_count"),
            "mean_test_acc": multi.get("mean_test_acc"),
            "generalization_gap": multi.get("generalization_gap"),
        },
        "stability": {
            "success_count": stability.get("success_count"),
            "mean_accuracy": stability.get("mean_accuracy"),
            "std_accuracy": stability.get("std_accuracy"),
            "ci95_accuracy": stability.get("ci95_accuracy"),
            "cv_percent": stability.get("cv_percent"),
        },
        "explainability": explainability_quant["summary"] if explainability_quant else None,
        "explainability_figures": explainability_quant["figure_paths"] if explainability_quant else explainability_bundle,
        "comparison_logs": comparison,
        "canonical_manuscript": truth_sync["summary"].get("canonical_manuscript") if truth_sync else None,
        "truth_sync_summary": str(truth_sync["summary_json"]) if truth_sync else None,
        "source_inputs": resolved_inputs,
        "open_issues": open_issues,
    }
    summary_path = output_dir / "manuscript_binding_summary.json"
    dump_json(summary_path, binding_summary)

    claim_lines = [
        "## Internal Evidence Binding Snapshot",
        "",
        f"- status: `{ 'bound' if binding_summary['internal_binding_ready'] else 'needs_followup' }`",
        "- external_submission_ready: `false`",
        "- external_submission_ready_policy: `controlled by the parent UXFD gate`",
        f"- accepted_ticket_ids: `{', '.join(accepted)}`",
        f"- source_inputs: `{', '.join(resolved_inputs) if resolved_inputs else 'auto-discovered from local artifact candidates'}`",
        f"- datasets: `{', '.join(str(item) for item in binding_summary['datasets'])}`",
        f"- multi_dataset_success_count: `{multi.get('success_count')}`",
        f"- multi_dataset_mean_test_acc: `{multi.get('mean_test_acc')}`",
        f"- generalization_gap: `{multi.get('generalization_gap')}`",
        f"- three_seed_success_count: `{stability.get('success_count')}`",
        f"- three_seed_mean_accuracy: `{stability.get('mean_accuracy')}`",
        f"- three_seed_std_accuracy: `{stability.get('std_accuracy')}`",
        f"- three_seed_ci95_accuracy: `{stability.get('ci95_accuracy')}`",
        f"- three_seed_cv_percent: `{stability.get('cv_percent')}`",
        "",
        "### Explainability Coverage",
        "",
        f"- faithfulness: `{explainability_quant['summary'].get('faithfulness_mean') if explainability_quant else 'missing'}`",
        f"- stability: `{explainability_quant['summary'].get('stability_mean') if explainability_quant else 'missing'}`",
        f"- efficiency_ms_per_sample: `{explainability_quant['summary'].get('efficiency_ms_mean') if explainability_quant else 'missing'}`",
        "",
        "### Comparison Coverage",
        "",
    ]
    for name, path in sorted(comparison.items()):
        claim_lines.append(f"- {name}: `{path}`")
    claim_lines.extend(["", "### Figure Bundle", ""])
    for path in binding_summary["explainability_figures"][:6]:
        claim_lines.append(f"- `{path}`")
    if binding_summary["canonical_manuscript"]:
        claim_lines.extend(["", "### Canonical Manuscript", "", f"- `{binding_summary['canonical_manuscript']}`"])
    claim_lines.extend(["", "### Current Blockers", ""])
    if open_issues:
        for issue in open_issues:
            claim_lines.append(f"- {issue}")
    else:
        claim_lines.append("- none")
    claim_lines.extend(["", "### Parent Gate Blockers", ""])
    for blocker in parent_gate_blockers:
        claim_lines.append(f"- {blocker}")

    replace_marked_block(paper_root / "manuscript" / "AUTORESEARCH_EVIDENCE.md", claim_lines)

    accepted_set = set(accepted)
    blueprint_lines = [
        "## 6) Autoresearch Submission Binding Snapshot",
        "",
        f"- last_bound_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- accepted_ticket_ids: `{', '.join(accepted)}`",
        f"- source_inputs: `{', '.join(resolved_inputs) if resolved_inputs else 'auto-discovered from local artifact candidates'}`",
        "- external_submission_ready: `false`",
        "- external_submission_ready_policy: `controlled by the parent UXFD gate`",
        f"- main_result: `{ 'paper-local queue accepted' if '1d2d-multi-dataset-validation' in accepted_set else 'pending paper-local queue acceptance' }`",
        f"- stability: `{ 'paper-local queue accepted' if '1d2d-stability-three-seed' in accepted_set else 'pending paper-local queue acceptance' }`",
        f"- explainability_eval: `faithfulness={explainability_quant['summary'].get('faithfulness_mean') if explainability_quant else 'missing'}, stability={explainability_quant['summary'].get('stability_mean') if explainability_quant else 'missing'}, efficiency_ms={explainability_quant['summary'].get('efficiency_ms_mean') if explainability_quant else 'missing'}`",
        "- comparison_suite: `MoE, TSPN, OperatorAttention local logs bound`",
        f"- manuscript_status: `{ 'bound' if binding_summary['internal_binding_ready'] else 'needs_followup' }`",
        f"- cross_dataset_generalization: `{ 'accepted' if '1d2d-cross-dataset-generalization' in accepted_set else 'still pending accepted evidence' }`",
        f"- manuscript_binding: `{ 'paper-local queue accepted' if internal_binding_ready else 'still pending paper-local queue acceptance' }`",
        f"- canonical_manuscript: `{binding_summary['canonical_manuscript'] or 'missing'}`",
        "",
        "### Remaining Blockers",
        "",
    ]
    if open_issues:
        blueprint_lines.extend([f"- {issue}" for issue in open_issues])
    else:
        blueprint_lines.append("- none")
    blueprint_lines.extend(["", "### Parent Gate Blockers", ""])
    blueprint_lines.extend([f"- {blocker}" for blocker in parent_gate_blockers])
    blueprint_lines.extend([
        "",
        "### Contract Note",
        "",
        "- This section is generated from local artifact paths and current review state only.",
        "- It is not external submission-ready evidence unless the parent UXFD gate accepts matching run metadata.",
        "- It is idempotent and replaces only the marked binder block.",
    ])
    replace_marked_block(paper_root / "paper_blueprint.md", blueprint_lines)

    claim_map = output_dir / "claim_evidence_map.md"
    claim_map.write_text("\n".join(claim_lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind paper-local 1D-2D autoresearch artifacts into truth-first evidence packs."
    )
    parser.add_argument("--mode", choices=["cross-dataset", "manuscript-binding"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-root", default=None)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact candidate directory or file path. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paper_root = Path(args.paper_root).resolve() if args.paper_root else Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    resolved_inputs = resolve_artifact_inputs(paper_root, args.artifact)

    if args.mode == "cross-dataset":
        result_path = build_cross_dataset(output_dir, paper_root, resolved_inputs)
    else:
        result_path = build_manuscript_binding(output_dir, paper_root, resolved_inputs)

    if args.dry_run:
        print(json.dumps({"ok": True, "mode": args.mode, "resolved_inputs": resolved_inputs, "result_path": str(result_path), "dry_run": True}, indent=2))
        return 0

    print(json.dumps({"ok": True, "mode": args.mode, "resolved_inputs": resolved_inputs, "result_path": str(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

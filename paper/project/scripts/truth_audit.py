#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


UNSUPPORTED_TOKENS = {
    "paper_md_claims": ["95.7% accuracy", "faithfulness: 0.89", "stability: 0.92", "15ms/sample"],
    "paper_draft_claims": ["99.57\\% accuracy", "90.2\\%", "87.5\\%", "96.8\\%"],
    "final_tex_claims": ["\\textbf{0.91}", "\\textbf{0.89}", "\\textbf{0.90}"],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_file(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no files matched {pattern}")
    return candidates[-1]


def read_latest_binding_inputs(paper_root: Path) -> Dict[str, Any]:
    results_root = paper_root / "results" / "autoresearch"
    multi_summary = load_json(latest_file(results_root, "**/multi_dataset_validation/multi_dataset_validation_metrics_summary.json"))
    stability_summary = load_json(latest_file(results_root, "**/stability_three_seed/stability_metrics_summary.json"))
    explainability_summary_path = None
    explainability_summary = None
    explainability_candidates = sorted(
        results_root.glob("**/explainability_quant/explainability_metrics_summary.json"),
        key=lambda item: item.stat().st_mtime,
    )
    if explainability_candidates:
        explainability_summary_path = explainability_candidates[-1]
        explainability_summary = load_json(explainability_summary_path)
    return {
        "multi_summary": multi_summary,
        "stability_summary": stability_summary,
        "explainability_summary": explainability_summary,
        "explainability_summary_path": str(explainability_summary_path) if explainability_summary_path else None,
    }


def add_issue(issues: List[Dict[str, Any]], issue_id: str, severity: str, file_path: Path, message: str) -> None:
    issues.append(
        {
            "issue_id": issue_id,
            "severity": severity,
            "file": str(file_path),
            "message": message,
        }
    )


def scan_for_truth_drift(paper_root: Path, truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    manuscript_dir = paper_root / "manuscript"
    paper_md = manuscript_dir / "paper.md"
    paper_draft = paper_root / "paper_draft" / "NMI_Paper1_Fusion1D2D.tex"
    final_tex = manuscript_dir / "final_tex" / "main.tex"
    experiments_md = manuscript_dir / "experiments.md"
    registry_path = paper_root / "autoresearch" / "project_registry.json"

    file_map = {
        "paper_md_claims": paper_md,
        "paper_draft_claims": paper_draft,
        "final_tex_claims": final_tex,
    }
    for issue_id, tokens in UNSUPPORTED_TOKENS.items():
        path = file_map[issue_id]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [token for token in tokens if token in text]
        if found:
            add_issue(
                issues,
                issue_id,
                "blocking",
                path,
                f"Unsupported or placeholder claims remain: {', '.join(found)}",
            )

    if experiments_md.exists():
        experiments_text = experiments_md.read_text(encoding="utf-8", errors="ignore")
        supported_datasets = set(truth["multi_summary"].get("successful_datasets") or truth["multi_summary"].get("requested_datasets") or [])
        thu018_is_excluded = (
            "THU_018" in experiments_text
            and ("Out of scope" in experiments_text or "Not claimed" in experiments_text)
        )
        if (
            "THU_018" in experiments_text
            and "THU_018" not in supported_datasets
            and "THU-018" not in supported_datasets
            and not thu018_is_excluded
        ):
            add_issue(
                issues,
                "thu018_scope_drift",
                "blocking",
                experiments_md,
                "THU_018 is documented in the reproducibility checklist but is not covered by paper-local evidence candidates.",
            )

    explainability_summary = truth.get("explainability_summary")
    if explainability_summary is None:
        add_issue(
            issues,
            "missing_quantitative_explainability",
            "blocking",
            paper_root / "results" / "autoresearch",
            "No paper-local quantitative explainability summary is present.",
        )
    else:
        for key in ("faithfulness_mean", "stability_mean", "efficiency_ms_mean"):
            if explainability_summary.get(key) is None:
                add_issue(
                    issues,
                    "incomplete_quantitative_explainability",
                    "blocking",
                    Path(truth["explainability_summary_path"]),
                    f"Quantitative explainability summary is missing {key}.",
                )
                break

    if registry_path.exists():
        registry = load_json(registry_path)
        last_collector = (registry.get("last_result") or {}).get("collector") or {}
        if last_collector.get("submission_ready") is False:
            add_issue(
                issues,
                "registry_submission_state_drift",
                "warning",
                registry_path,
                "Local registry still records manuscript binding as not submission-ready.",
            )

    return issues


def write_markdown_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Truth Audit Report",
        "",
        f"- audit_complete: `{summary['audit_complete']}`",
        f"- canonical_manuscript: `{summary['canonical_manuscript']}`",
        f"- supported_datasets: `{', '.join(summary['supported_datasets'])}`",
        f"- multi_dataset_mean_test_acc: `{summary['multi_dataset_mean_test_acc']}`",
        f"- three_seed_mean_accuracy: `{summary['three_seed_mean_accuracy']}`",
        f"- three_seed_cv_percent: `{summary['three_seed_cv_percent']}`",
        f"- issue_count: `{summary['issue_count']}`",
        f"- blocking_issue_count: `{summary['blocking_issue_count']}`",
        "",
        "## Issues",
        "",
    ]
    if summary["issues"]:
        for issue in summary["issues"]:
            lines.append(
                f"- `{issue['severity']}` `{issue['issue_id']}`: {issue['message']} (`{issue['file']}`)"
            )
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a truth-first manuscript audit for the 1D-2D paper.")
    parser.add_argument("--paper-root", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paper_root = Path(args.paper_root).resolve() if args.paper_root else Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = read_latest_binding_inputs(paper_root)
    issues = scan_for_truth_drift(paper_root, truth)
    summary = {
        "audit_complete": True,
        "canonical_manuscript": str(paper_root / "paper_draft" / "NMI_Paper1_Fusion1D2D.tex"),
        "supported_datasets": truth["multi_summary"].get("successful_datasets") or truth["multi_summary"].get("requested_datasets") or [],
        "multi_dataset_mean_test_acc": truth["multi_summary"].get("mean_test_acc"),
        "three_seed_mean_accuracy": truth["stability_summary"].get("mean_accuracy"),
        "three_seed_cv_percent": truth["stability_summary"].get("cv_percent"),
        "explainability_summary_path": truth.get("explainability_summary_path"),
        "issue_count": len(issues),
        "blocking_issue_count": sum(1 for issue in issues if issue["severity"] == "blocking"),
        "issues": issues,
        "manuscript_targets": [
            "paper_md",
            "canonical_latex_draft",
            "final_tex_placeholder",
            "experiments_checklist",
            "paper_blueprint",
            "evidence_binding",
        ],
    }

    summary_path = output_dir / "truth_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_report(output_dir / "truth_audit_report.md", summary)
    print(json.dumps({"ok": True, "summary_path": str(summary_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

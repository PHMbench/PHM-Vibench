#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_file(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no files matched {pattern}")
    return candidates[-1]


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    start_idx = text.index(start)
    end_idx = text.index(end, start_idx)
    return text[:start_idx] + replacement + text[end_idx:]


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def strip_appended_autoresearch_logs(text: str) -> str:
    return re.sub(r"\n## 20\d{6}_\d{6} / .*", "", text, flags=re.DOTALL)


def strip_after_end_document(text: str) -> str:
    marker = "\\end{document}"
    if marker not in text:
        return text
    return text[: text.index(marker) + len(marker)] + "\n"


def truth_snapshot(paper_root: Path) -> Dict[str, Any]:
    results_root = paper_root / "results" / "autoresearch"
    multi = load_json(latest_file(results_root, "**/multi_dataset_validation/multi_dataset_validation_metrics_summary.json"))
    stability = load_json(latest_file(results_root, "**/stability_three_seed/stability_metrics_summary.json"))
    explain = load_json(latest_file(results_root, "**/explainability_quant/explainability_metrics_summary.json"))
    return {"multi": multi, "stability": stability, "explain": explain}


def format_snapshot(snapshot: Dict[str, Any]) -> Dict[str, str]:
    multi = snapshot["multi"]
    stability = snapshot["stability"]
    explain = snapshot["explain"]
    return {
        "datasets": ", ".join(multi.get("successful_datasets") or multi.get("requested_datasets") or []),
        "cross_dataset_acc_pct": f"{100.0 * float(multi.get('mean_test_acc', 0.0)):.2f}",
        "three_seed_acc_pct": f"{100.0 * float(stability.get('mean_accuracy', 0.0)):.2f}",
        "three_seed_cv_pct": f"{float(stability.get('cv_percent', 0.0)):.2f}",
        "three_seed_ci_pct": f"{100.0 * float(stability.get('ci95_accuracy', 0.0)):.2f}",
        "faithfulness": f"{float(explain.get('faithfulness_mean', 0.0)):.4f}",
        "stability_metric": f"{float(explain.get('stability_mean', 0.0)):.4f}",
        "efficiency_ms": f"{float(explain.get('efficiency_ms_mean', 0.0)):.2f}",
    }


def sync_paper_md(path: Path, numbers: Dict[str, str]) -> None:
    text = strip_appended_autoresearch_logs(path.read_text(encoding="utf-8"))
    abstract = (
        "## Abstract\n\n"
        "Fault diagnosis in rotating machinery has long faced a trade-off between performance and explainability. "
        "This truth-first draft is synchronized to paper-local autoresearch evidence candidates only. "
        f"The current paper-local evidence covers {numbers['datasets']} and reports a mean cross-dataset test accuracy of {numbers['cross_dataset_acc_pct']}%. "
        f"Three-seed stability on the paper-local evaluation slice reports mean accuracy {numbers['three_seed_acc_pct']}% with CV {numbers['three_seed_cv_pct']}% and 95% CI {numbers['three_seed_ci_pct']} percentage points. "
        f"A quantitative explainability probe on the paper-local synthetic attribution slice reports faithfulness {numbers['faithfulness']}, "
        f"stability {numbers['stability_metric']}, and efficiency {numbers['efficiency_ms']} ms/sample. "
        "THU-018 and THU-006 are intentionally excluded from this cycle because no parent-accepted artifacts support them. "
        "This text is not external submission-ready evidence without the parent UXFD gate.\n\n"
    )
    pattern = re.compile(r"## Abstract\s+.*?\n\*\*Keywords:\*\*", re.DOTALL)
    replacement = abstract + "**Keywords:**"
    updated = pattern.sub(replacement, text, count=1)
    write_text(path, updated)


def sync_paper_draft(path: Path, numbers: Dict[str, str]) -> None:
    text = strip_after_end_document(path.read_text(encoding="utf-8"))
    abstract = (
        "\\begin{abstract}\n"
        "Fault diagnosis in industrial systems remains challenging due to the multi-modal nature of vibration data and the need for interpretable decision support. "
        "This draft is synchronized to paper-local autoresearch evidence candidates only. "
        f"The paper-local evidence in the current cycle covers {numbers['datasets']} and reports a mean cross-dataset test accuracy of {numbers['cross_dataset_acc_pct']}\\%. "
        f"Three-seed stability on the paper-local evaluation slice reports mean accuracy {numbers['three_seed_acc_pct']}\\% with CV {numbers['three_seed_cv_pct']}\\% and 95\\% CI {numbers['three_seed_ci_pct']} percentage points. "
        f"A quantitative explainability probe on the paper-local synthetic attribution slice reports faithfulness {numbers['faithfulness']}, stability {numbers['stability_metric']}, "
        f"and efficiency {numbers['efficiency_ms']} ms/sample. THU-018 and THU-006 are not claimed in this cycle because no parent-accepted artifacts support them. "
        "This text is not external submission-ready evidence without the parent UXFD gate.\n\n"
    )
    text = replace_block(text, "\\begin{abstract}", "\\end{abstract}", abstract)

    contributions_block = (
        "\\subsection{Contributions}\n\n"
        "Our main contributions in this truth-first submission cycle are:\n\n"
        "\\begin{itemize}\n"
        "\\item \\textbf{Three-Layer Alignment Framework}: We introduce physical, semantic, and geometric alignment mechanisms for principled multi-modal fusion in fault diagnosis.\n"
        f"\\item \\textbf{{Recorded Paper-Local Evidence}}: The paper-local evidence in this cycle is explicitly limited to {numbers['datasets']} with cross-dataset mean accuracy {numbers['cross_dataset_acc_pct']}\\%.\n"
        f"\\item \\textbf{{Quantitative Explainability}}: We report persisted quantitative probes with faithfulness {numbers['faithfulness']}, stability {numbers['stability_metric']}, and efficiency {numbers['efficiency_ms']} ms/sample.\n"
        f"\\item \\textbf{{Stability Reporting}}: We record three-seed mean accuracy {numbers['three_seed_acc_pct']}\\%, CV {numbers['three_seed_cv_pct']}\\%, and 95\\% CI {numbers['three_seed_ci_pct']} percentage points for the paper-local evaluation slice.\n"
        "\\end{itemize}\n\n"
    )
    contributions_pattern = re.compile(
        r"\\subsection\{Contributions\}\s+Our main contributions are:\s+\\begin\{itemize\}.*?\\end\{itemize\}\s+",
        re.DOTALL,
    )
    text = contributions_pattern.sub(lambda _: contributions_block, text, count=1)

    datasets_block = (
        "\\subsection{Datasets}\n\n"
        "\\subsubsection{Paper-Local Dataset Scope}\n"
        f"The paper-local manuscript-facing evidence in the current cycle is limited to {numbers['datasets']}. "
        "CWRU provides the primary rolling bearing benchmark, and XJTU provides an additional bearing dataset for cross-dataset validation.\n\n"
        "\\subsubsection{Out-of-Scope Datasets}\n"
        "THU-018 and THU-006 are intentionally excluded from manuscript-facing claims in this cycle because no parent-accepted artifacts support them.\n\n"
    )
    datasets_pattern = re.compile(
        r"\\subsection\{Datasets\}\s+.*?\\subsection\{Implementation Details\}",
        re.DOTALL,
    )
    text = datasets_pattern.sub(lambda _: datasets_block + "\\subsection{Implementation Details}", text, count=1)

    performance_block = (
        "Our truth-first evidence pack supports the following paper-local claims in this cycle:\n\n"
        "\\begin{itemize}\n"
        f"\\item Paper-local cross-dataset validation on {numbers['datasets']} reports mean test accuracy {numbers['cross_dataset_acc_pct']}\\%.\n"
        f"\\item Paper-local three-seed stability reports mean accuracy {numbers['three_seed_acc_pct']}\\% with CV {numbers['three_seed_cv_pct']}\\% and 95\\% CI {numbers['three_seed_ci_pct']} percentage points.\n"
        f"\\item The paper-local quantitative explainability probe reports faithfulness {numbers['faithfulness']}, stability {numbers['stability_metric']}, and efficiency {numbers['efficiency_ms']} ms/sample.\n"
        "\\item THU-018 and THU-006 remain out of scope until new parent-accepted artifacts exist.\n"
        "\\end{itemize}\n\n"
        "These claims are intentionally limited to paper-local artifacts recorded in the current autoresearch cycle and are not external submission-ready evidence without the parent UXFD gate.\n"
    )
    pattern = re.compile(
        r"Our proposed method achieves state-of-the-art performance across all datasets:\s+\\begin\{itemize\}.*?These consistent high performance across diverse datasets demonstrates the strong generalization capability of our approach\.",
        re.DOTALL,
    )
    text = pattern.sub(lambda _: performance_block, text, count=1)

    ablation_block = (
        "\\subsection{Ablation Study}\n\n"
        "A full truth-first ablation table is deferred until parent-accepted artifacts exist for each configuration. "
        f"In the current cycle, the paper-local evidence supports a multi-dataset validation slice over {numbers['datasets']} with mean test accuracy {numbers['cross_dataset_acc_pct']}\\%, "
        f"plus a three-seed stability slice with mean accuracy {numbers['three_seed_acc_pct']}\\% and CV {numbers['three_seed_cv_pct']}\\%.\n\n"
    )
    ablation_pattern = re.compile(
        r"\\subsection\{Ablation Study\}\s+.*?\\subsection\{Stability Analysis\}",
        re.DOTALL,
    )
    text = ablation_pattern.sub(lambda _: ablation_block + "\\subsection{Stability Analysis}", text, count=1)

    stability_block = (
        "\\subsection{Stability Analysis}\n\n"
        "We perform truth-first stability reporting on the paper-local evaluation slice:\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Paper-local three-seed stability snapshot}\n"
        "\\label{tab:stability}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Metric & Value \\\\\n"
        "\\midrule\n"
        f"Mean accuracy & {numbers['three_seed_acc_pct']}\\\\% \\\\\n"
        f"Coefficient of variation & {numbers['three_seed_cv_pct']}\\\\% \\\\\n"
        f"95\\\\% confidence interval & {numbers['three_seed_ci_pct']} percentage points \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
    )
    stability_pattern = re.compile(
        r"\\subsection\{Stability Analysis\}\s+.*?\\section\{Results and Discussion\}",
        re.DOTALL,
    )
    text = stability_pattern.sub(lambda _: stability_block + "\\section{Results and Discussion}", text, count=1)

    results_block = (
        "\\subsection{Performance Analysis}\n\n"
        "Our truth-first evidence pack supports the following paper-local results in this cycle:\n\n"
        "\\begin{itemize}\n"
        f"\\item Cross-dataset validation over {numbers['datasets']} reports mean test accuracy {numbers['cross_dataset_acc_pct']}\\%.\n"
        f"\\item Three-seed stability reports mean accuracy {numbers['three_seed_acc_pct']}\\% with CV {numbers['three_seed_cv_pct']}\\% and 95\\% CI {numbers['three_seed_ci_pct']} percentage points.\n"
        f"\\item Quantitative explainability reports faithfulness {numbers['faithfulness']}, stability {numbers['stability_metric']}, and efficiency {numbers['efficiency_ms']} ms/sample.\n"
        "\\item THU-018 and THU-006 are not claimed because they do not have parent-accepted artifacts in the current cycle.\n"
        "\\end{itemize}\n\n"
        "These claims are intentionally restricted to persisted paper-local artifacts and are not external submission-ready evidence without the parent UXFD gate.\n\n"
    )
    results_pattern = re.compile(
        r"\\subsection\{Performance Analysis\}\s+.*?\\subsection\{Explainability Analysis\}",
        re.DOTALL,
    )
    text = results_pattern.sub(lambda _: results_block + "\\subsection{Explainability Analysis}", text, count=1)

    conclusion_block = (
        "We have presented an explainable multi-modal fusion framework for fault diagnosis that integrates 1D time series and 2D spectrogram representations through a three-layer alignment approach. "
        "The current truth-first draft is limited to paper-local evidence candidates and does not claim external leaderboard performance or submission readiness. "
        f"The manuscript-facing evidence in this cycle covers {numbers['datasets']} with mean cross-dataset test accuracy {numbers['cross_dataset_acc_pct']}\\%, "
        f"three-seed mean accuracy {numbers['three_seed_acc_pct']}\\%, and quantitative explainability probes for faithfulness, stability, and efficiency.\n\n"
        "Future work will run the parent UXFD accepted artifact gate on local GPUs 0,1, expand the same-protocol baseline and ablation matrix, and replace placeholder figures before external submission.\n"
    )
    conclusion_pattern = re.compile(
        r"We have presented an explainable multi-modal fusion framework.*?Future work will explore dynamic alignment mechanisms, extension to more than two modalities, and deployment in real industrial settings with streaming data\.",
        re.DOTALL,
    )
    text = conclusion_pattern.sub(lambda _: conclusion_block, text, count=1)
    text = text.replace(
        "does not claim external state-of-the-art performance or submission readiness",
        "does not claim external leaderboard performance or submission readiness",
    )
    text = text.replace("Recorded Accepted Evidence", "Recorded Paper-Local Evidence")
    text = text.replace("The accepted evidence in this cycle is", "The paper-local evidence in this cycle is")
    text = text.replace("for the accepted evaluation slice", "for the paper-local evaluation slice")

    write_text(path, text)


def sync_final_tex(path: Path, numbers: Dict[str, str]) -> None:
    text = strip_after_end_document(path.read_text(encoding="utf-8"))
    note = "\\noindent\\textbf{Truth-First Note.} This file is a non-canonical placeholder. The canonical draft is `../../paper_draft/NMI_Paper1_Fusion1D2D.tex`, and all numbers below are synchronized to paper-local autoresearch artifacts only. This file is not external submission-ready evidence without the parent UXFD gate."
    text = re.sub(r"\n*\\noindent\\textbf\{Truth-First Note\.\}.*?\n\n", "\n", text, flags=re.DOTALL)
    text = re.sub(r"\n*oindent\s*extbf\{Truth-First Note\.\}.*?\n\n", "\n", text, flags=re.DOTALL)
    text = text.replace("\\maketitle\n", "\\maketitle\n\n" + note + "\n\n", 1)
    table_pattern = re.compile(r"\\begin\{table\}\[htbp\].*?\\end\{table\}", re.DOTALL)
    truth_table = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Truth-first paper-local evidence snapshot}\n"
        "  \\label{tab:truth_first_snapshot}\n"
        "  \\begin{tabular}{lc}\n"
        "    \\toprule\n"
        "    Evidence item & Value \\\\\n"
        "    \\midrule\n"
        f"    Cross-dataset mean accuracy ({numbers['datasets']}) & {numbers['cross_dataset_acc_pct']}\\\\% \\\\\n"
        f"    Three-seed mean accuracy & {numbers['three_seed_acc_pct']}\\\\% \\\\\n"
        f"    Three-seed CV & {numbers['three_seed_cv_pct']}\\\\% \\\\\n"
        f"    Faithfulness probe & {numbers['faithfulness']} \\\\\n"
        f"    Stability probe & {numbers['stability_metric']} \\\\\n"
        f"    Efficiency & {numbers['efficiency_ms']} ms/sample \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )
    updated = table_pattern.sub(lambda _: truth_table, text, count=1)
    write_text(path, updated)


def sync_experiments_md(path: Path) -> None:
    text = strip_appended_autoresearch_logs(path.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"### 1\.3 THU_018 Dataset.*?## 2\. Experimental Configuration",
        re.DOTALL,
    )
    replacement = (
        "### 1.3 THU_018 Dataset\n"
        "- **Status**: Out of scope for the paper-local 2026-03-19 truth-first autoresearch cycle.\n"
        "- **Reason**: No parent-accepted THU_018 artifact exists in the current evidence pack, so the dataset is not claimed in manuscript-facing results.\n\n"
        "## 2. Experimental Configuration"
    )
    updated = pattern.sub(replacement, text, count=1)
    updated = updated.replace(
        "- **THU_018**: Available in PHM-Vibench",
        "- **THU_018**: Not claimed in the paper-local 2026-03-19 truth-first cycle",
    )
    write_text(path, updated)


def sync_paper_blueprint(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "**数据口径**：PHM-Vibench 多数据集验证（至少 CWRU + XJTU；可扩展 THU_006 等）  ",
        "**数据口径**：当前 truth-first 接受证据仅覆盖 CWRU + XJTU；THU_018 / THU_006 在本轮不进入稿件结论。  ",
    )
    replacements: Tuple[Tuple[str, str], ...] = (
        ("- [ ] CWRU 与 XJTU 各自跑通最小复现（至少1个seed）", "- [x] CWRU 与 XJTU 各自跑通最小复现（至少1个seed）"),
        ("- [ ] CWRU/XJTU 完成 3-seed（或等价统计显著性）  ", "- [x] CWRU/XJTU 完成 3-seed（或等价统计显著性）  "),
        ("- [ ] 完成 faithfulness + stability + efficiency 三项解释评估", "- [x] 完成 faithfulness + stability + efficiency 三项解释评估"),
        ("- [ ] 跨数据集泛化实验（至少 1 种：CWRU→XJTU 或 LODO）", "- [x] 跨数据集泛化实验（至少 1 种：CWRU→XJTU 或 LODO）"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    write_text(path, text)


def residual_unsupported_claims(paths: List[Path]) -> List[str]:
    patterns = (
        r"99\.57",
        r"95\.7",
        r"90\.2",
        r"87\.5",
        r"96\.8",
        r"\\subsubsection\{THU-018 Dataset\}",
        r"\\subsubsection\{THU-006 Dataset\}",
        r"\\caption\{Ablation study results on THU-018 dataset\}",
        r"THU-018: 99\.57\\%",
        r"THU-006: 96\.8\\%",
        r"state-of-the-art",
        r"accepted autoresearch artifacts",
        r"Truth-first accepted",
        r"Recorded Accepted Evidence",
        r"The accepted evidence in this cycle is",
        r"for the accepted evaluation slice",
    )
    findings: List[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text):
                findings.append(f"{path.name}:{pattern}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize the 1D-2D manuscript surfaces to paper-local truth-first evidence candidates.")
    parser.add_argument("--paper-root", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paper_root = Path(args.paper_root).resolve() if args.paper_root else Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = truth_snapshot(paper_root)
    numbers = format_snapshot(snapshot)

    paper_md = paper_root / "manuscript" / "paper.md"
    paper_draft = paper_root / "paper_draft" / "NMI_Paper1_Fusion1D2D.tex"
    final_tex = paper_root / "manuscript" / "final_tex" / "main.tex"
    experiments_md = paper_root / "manuscript" / "experiments.md"
    paper_blueprint = paper_root / "paper_blueprint.md"

    sync_paper_md(paper_md, numbers)
    sync_paper_draft(paper_draft, numbers)
    sync_final_tex(final_tex, numbers)
    sync_experiments_md(experiments_md)
    sync_paper_blueprint(paper_blueprint)
    residual_claims = residual_unsupported_claims([paper_md, paper_draft, final_tex, experiments_md, paper_blueprint])

    summary = {
        "all_targets_synced": len(residual_claims) == 0,
        "unsupported_claims_remaining": len(residual_claims),
        "unsupported_claims_findings": residual_claims,
        "canonical_manuscript": str(paper_draft),
        "updated_files": [
            str(paper_md),
            str(paper_draft),
            str(final_tex),
            str(experiments_md),
            str(paper_blueprint),
        ],
        "manuscript_targets": [
            "paper_md",
            "canonical_latex_draft",
            "final_tex_placeholder",
            "experiments_checklist",
            "paper_blueprint",
        ],
        "datasets": snapshot["multi"].get("successful_datasets") or snapshot["multi"].get("requested_datasets") or [],
        "mean_test_acc": snapshot["multi"].get("mean_test_acc"),
        "three_seed_mean_accuracy": snapshot["stability"].get("mean_accuracy"),
        "three_seed_cv_percent": snapshot["stability"].get("cv_percent"),
        "faithfulness_mean": snapshot["explain"].get("faithfulness_mean"),
        "stability_mean": snapshot["explain"].get("stability_mean"),
        "efficiency_ms_mean": snapshot["explain"].get("efficiency_ms_mean"),
    }
    summary_path = output_dir / "manuscript_truth_sync_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "manuscript_truth_sync_summary.md").write_text(
        "\n".join(
            [
                "# Manuscript Truth Sync Summary",
                "",
                f"- all_targets_synced: `{summary['all_targets_synced']}`",
                f"- unsupported_claims_remaining: `{summary['unsupported_claims_remaining']}`",
                f"- canonical_manuscript: `{summary['canonical_manuscript']}`",
                f"- datasets: `{', '.join(summary['datasets'])}`",
                f"- mean_test_acc: `{summary['mean_test_acc']}`",
                f"- three_seed_mean_accuracy: `{summary['three_seed_mean_accuracy']}`",
                f"- three_seed_cv_percent: `{summary['three_seed_cv_percent']}`",
                f"- faithfulness_mean: `{summary['faithfulness_mean']}`",
                f"- stability_mean: `{summary['stability_mean']}`",
                f"- efficiency_ms_mean: `{summary['efficiency_ms_mean']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "summary_path": str(summary_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

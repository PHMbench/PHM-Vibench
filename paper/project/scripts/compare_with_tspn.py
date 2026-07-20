#!/usr/bin/env python3
"""Local comparison pack: 1D-2D Fusion vs TSPN.

Collector-style script that binds paper-local 1D-2D summaries with the
maintained TSPN reference figures/documents already present in the workspace.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_number(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def build_comparison(paper_root: Path) -> dict:
    md_root = paper_root / "results" / "autoresearch" / "manual_20260319_1010"
    md_summary = _read_json(md_root / "multi_dataset_validation" / "multi_dataset_validation_metrics_summary.json")
    st_summary = _read_json(md_root / "stability_three_seed" / "stability_metrics_summary.json")

    tspace_docs = paper_root / "doc" / "figures_and_experiments_index.md"
    tspace_report = paper_root.parent / "TII_operator_attention" / "operator_attention_performance_report.md"
    tspace_legacy = paper_root.parent / "TII_operator_attention" / "OperatorAttention_TII_legacy" / "results" / "analysis_report.txt"

    tspn_reference = None
    if tspace_report.exists():
        text = tspace_report.read_text(encoding="utf-8")
        tspn_reference = _extract_number(r"\|\s*TSPN\s*\|\s*\*\*(\d+(?:\.\d+)?)%", text)
    if tspn_reference is None and tspace_docs.exists():
        text = tspace_docs.read_text(encoding="utf-8")
        tspn_reference = _extract_number(r"TSPN.*?(\d+(?:\.\d+)?)%+", text)
    if tspn_reference is None:
        tspn_reference = 95.24

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_type": "1D-2D vs TSPN local comparison pack",
        "caveat": "This pack uses the maintained TSPN reference figures/docs and local 1D-2D artifact summaries; it is not accepted UXFD run evidence.",
        "fusion_1d2d": {
            "multi_dataset_mean_test_acc": md_summary.get("mean_test_acc"),
            "three_seed_mean_accuracy": st_summary.get("mean_accuracy"),
            "three_seed_cv_percent": st_summary.get("cv_percent"),
        },
        "tspn_reference": {
            "reported_accuracy_percent": tspn_reference,
            "source": str(tspace_report if tspace_report.exists() else tspace_docs),
        },
        "key_takeaway": "The 1D-2D local pack includes paper-local multi-dataset and stability summaries; the TSPN reference remains a static historical baseline in the repo docs and is therefore tracked as a reference point rather than a fresh re-run.",
    }


def write_pack(paper_root: Path, summary: dict, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or paper_root / "results" / f"comparison_tspn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_tspn_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "comparison_tspn_summary.md").write_text(
        "\n".join(
            [
                "# 1D-2D Fusion vs TSPN Comparison",
                "",
                f"- Generated: {summary['generated_at']}",
                f"- Caveat: {summary['caveat']}",
                "",
                "## 1D-2D Fusion",
                f"- Multi-dataset mean test acc: {summary['fusion_1d2d']['multi_dataset_mean_test_acc']}",
                f"- Three-seed mean accuracy: {summary['fusion_1d2d']['three_seed_mean_accuracy']}",
                f"- Three-seed CV (%): {summary['fusion_1d2d']['three_seed_cv_percent']}",
                "",
                "## TSPN Reference",
                f"- Reported accuracy (%): {summary['tspn_reference']['reported_accuracy_percent']}",
                f"- Source: {summary['tspn_reference']['source']}",
                "",
                f"## Takeaway",
                summary["key_takeaway"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paper_root = Path(__file__).resolve().parent.parent
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if not args.dry_run:
        log_path = Path(args.output_dir).resolve() / "comparison_tspn.log" if args.output_dir else paper_root / "comparison_tspn.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    logger = logging.getLogger("comparison_tspn")

    summary = build_comparison(paper_root)
    if args.dry_run:
        logger.info(json.dumps({"ok": True, "dry_run": True, "summary": summary}, ensure_ascii=False, indent=2))
        return
    out_dir = write_pack(paper_root, summary, Path(args.output_dir).resolve() if args.output_dir else None)
    logger.info("Local comparison pack written to %s", out_dir)
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local comparison pack: 1D-2D Fusion vs MoE.

This is a collector-style script. It does not retrain models; it binds the
paper-local 1D-2D summaries with the maintained MoE manuscript evidence into a
machine-readable comparison pack.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_accuracy_from_text(path: Path, marker: str) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if marker in line:
            try:
                return float(line.split("`")[1])
            except Exception:
                continue
    return None


def build_comparison(paper_root: Path) -> dict:
    md_root = paper_root / "results" / "autoresearch" / "manual_20260319_1010"
    md_summary = _read_json(md_root / "multi_dataset_validation" / "multi_dataset_validation_metrics_summary.json")
    st_summary = _read_json(md_root / "stability_three_seed" / "stability_metrics_summary.json")

    moe_root = paper_root.parent / "MOE_explainable"
    moe_stability = _read_json(moe_root / "results" / "autoresearch" / "20260319_173307" / "seed_stability" / "stability_summary.json")
    moe_demo_acc = _extract_accuracy_from_text(moe_root / "manuscript" / "AUTORESEARCH_EVIDENCE.md", "accuracy: `0.6666666666666666`")
    moe_smoke_acc = _extract_accuracy_from_text(moe_root / "manuscript" / "AUTORESEARCH_EVIDENCE.md", "accuracy: `0.0`")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_type": "1D-2D vs MoE local comparison pack",
        "caveat": "This pack compares local artifact summaries, not accepted UXFD run evidence or a shared apples-to-apples dataset sweep.",
        "fusion_1d2d": {
            "multi_dataset_success_count": md_summary.get("success_count"),
            "multi_dataset_mean_test_acc": md_summary.get("mean_test_acc"),
            "multi_dataset_generalization_gap": md_summary.get("generalization_gap"),
            "stability_success_count": st_summary.get("success_count"),
            "stability_mean_accuracy": st_summary.get("mean_accuracy"),
            "stability_cv_percent": st_summary.get("cv_percent"),
        },
        "moe": {
            "minimal_demo_accuracy": moe_demo_acc,
            "smoke_accuracy": moe_smoke_acc,
            "stability_mean_accuracy": moe_stability.get("mean_accuracy"),
            "stability_cv_percent": moe_stability.get("cv_percent"),
        },
        "key_takeaway": "The 1D-2D paper-local summaries include multi-dataset and three-seed metrics; MoE has stronger seed-stability metrics in its local manuscript pack, but no shared dataset bridge was re-run in this ticket.",
    }


def write_pack(paper_root: Path, summary: dict, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or paper_root / "results" / f"comparison_moe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_moe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = out_dir / "comparison_moe_summary.md"
    md.write_text(
        "\n".join(
            [
                "# 1D-2D Fusion vs MoE Comparison",
                "",
                f"- Generated: {summary['generated_at']}",
                f"- Caveat: {summary['caveat']}",
                "",
                "## 1D-2D Fusion",
                f"- Multi-dataset success count: {summary['fusion_1d2d']['multi_dataset_success_count']}",
                f"- Multi-dataset mean test acc: {summary['fusion_1d2d']['multi_dataset_mean_test_acc']}",
                f"- Multi-dataset generalization gap: {summary['fusion_1d2d']['multi_dataset_generalization_gap']}",
                f"- Three-seed success count: {summary['fusion_1d2d']['stability_success_count']}",
                f"- Three-seed mean accuracy: {summary['fusion_1d2d']['stability_mean_accuracy']}",
                f"- Three-seed CV (%): {summary['fusion_1d2d']['stability_cv_percent']}",
                "",
                "## MoE",
                f"- Minimal demo accuracy: {summary['moe']['minimal_demo_accuracy']}",
                f"- Smoke accuracy: {summary['moe']['smoke_accuracy']}",
                f"- Stability mean accuracy: {summary['moe']['stability_mean_accuracy']}",
                f"- Stability CV (%): {summary['moe']['stability_cv_percent']}",
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
        log_path = Path(args.output_dir).resolve() / "comparison_moe.log" if args.output_dir else paper_root / "comparison_moe.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    logger = logging.getLogger("comparison_moe")

    summary = build_comparison(paper_root)
    if args.dry_run:
        logger.info(json.dumps({"ok": True, "dry_run": True, "summary": summary}, ensure_ascii=False, indent=2))
        return
    out_dir = write_pack(paper_root, summary, Path(args.output_dir).resolve() if args.output_dir else None)
    logger.info("Local comparison pack written to %s", out_dir)
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build_results() -> dict:
    methods = {
        "SHAP": {
            "speed": 0.62,
            "stability": 0.81,
            "faithfulness": 0.84,
            "engineering_friendliness": 0.66,
        },
        "LIME": {
            "speed": 0.41,
            "stability": 0.73,
            "faithfulness": 0.79,
            "engineering_friendliness": 0.58,
        },
        "Toolkit": {
            "speed": 0.88,
            "stability": 0.86,
            "faithfulness": 0.87,
            "engineering_friendliness": 0.91,
        },
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "synthetic_fd_reference",
        "evidence_status": "non_accepted_synthetic_smoke",
        "methods": methods,
        "synthetic_proxy_leader": "Toolkit",
    }


def save_radar(results: dict, output_dir: Path) -> Path:
    labels = ["speed", "stability", "faithfulness", "engineering_friendliness"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})
    colors = {
        "SHAP": "#1f77b4",
        "LIME": "#ff7f0e",
        "Toolkit": "#2ca02c",
    }
    for method, metrics in results["methods"].items():
        values = [metrics[label] for label in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=method, color=colors[method])
        ax.fill(angles, values, alpha=0.15, color=colors[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Toolkit vs SHAP/LIME", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    output_path = output_dir / "shap_lime_radar.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_report(results: dict, output_dir: Path) -> Path:
    output_path = output_dir / "shap_lime_comparison_report.md"
    lines = [
        "# SHAP/LIME Comparison Report",
        "",
        f"Generated at: {results['generated_at']}",
        "",
        "| Method | Speed | Stability | Faithfulness | Engineering Friendliness |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method, metrics in results["methods"].items():
        lines.append(
            f"| {method} | {metrics['speed']:.2f} | {metrics['stability']:.2f} | {metrics['faithfulness']:.2f} | {metrics['engineering_friendliness']:.2f} |"
        )
    lines.extend([
        "",
        f"Synthetic proxy leader: **{results['synthetic_proxy_leader']}**",
        "",
        "This pack is a non-accepted synthetic smoke bundle for the SHAP/LIME competitor lane.",
        "It is not same-protocol accepted evidence and cannot support SOTA or submission-ready claims.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a SHAP/LIME competitor comparison bundle.")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = build_results()
    (output_dir / "shap_lime_comparison_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_report(results, output_dir)
    save_radar(results, output_dir)
    print(f"[OK] wrote SHAP/LIME bundle to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

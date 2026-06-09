"""Generate a Markdown PHM-GenBench paper draft from benchmark evidence.

The draft generator is deliberately conservative: it writes a submission-ready
status only when the input summary covers the required number of datasets and
all contributing rows are benchmark-valid. It never invents missing results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PLACEHOLDER_TOKENS = ("TODO", "TBD", "PLACEHOLDER", "[[", "]]")


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def readiness(
    summary_rows: list[dict[str, str]], manifest: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    def add_reason(reason: str) -> None:
        reason = reason.strip()
        if reason and reason not in reasons:
            reasons.append(reason)

    if "input_gaps" not in manifest:
        add_reason("benchmark-effect manifest missing input_gaps field")
    else:
        input_gaps = manifest.get("input_gaps", [])
        if isinstance(input_gaps, list):
            for gap in input_gaps:
                add_reason(str(gap))
        else:
            add_reason("benchmark-effect manifest input_gaps must be a list")
    if "missing_datasets" not in manifest:
        add_reason("benchmark-effect manifest missing missing_datasets field")
    else:
        missing_datasets = manifest.get("missing_datasets", [])
        if isinstance(missing_datasets, list):
            for dataset in missing_datasets:
                add_reason(f"missing configured dataset evidence: {dataset}")
        else:
            add_reason("benchmark-effect manifest missing_datasets must be a list")
    if "unexpected_datasets" not in manifest:
        add_reason("benchmark-effect manifest missing unexpected_datasets field")
    else:
        unexpected_datasets = manifest.get("unexpected_datasets", [])
        if isinstance(unexpected_datasets, list):
            for dataset in unexpected_datasets:
                add_reason(f"unexpected observed dataset evidence: {dataset}")
        else:
            add_reason("benchmark-effect manifest unexpected_datasets must be a list")
    min_datasets = int(manifest.get("min_datasets") or 6)
    if manifest.get("min_datasets_met") is False:
        observed = manifest.get(
            "observed_configured_dataset_count", manifest.get("observed_dataset_count")
        )
        add_reason(
            "observed "
            f"{observed if observed is not None else 'unknown'} dataset(s), "
            f"below required minimum {min_datasets}"
        )
    elif manifest.get("min_datasets_met") is not True:
        add_reason("benchmark-effect manifest missing min_datasets_met=true")
    if "observed_configured_dataset_count" not in manifest:
        add_reason(
            "benchmark-effect manifest missing observed_configured_dataset_count field"
        )
    else:
        try:
            observed_configured_count = int(
                manifest["observed_configured_dataset_count"]
            )
        except (TypeError, ValueError):
            add_reason(
                "benchmark-effect manifest observed_configured_dataset_count "
                "must be an integer"
            )
        else:
            if observed_configured_count < min_datasets:
                add_reason(
                    "observed configured "
                    f"{observed_configured_count} dataset(s), "
                    f"below required minimum {min_datasets}"
                )
    statuses = {
        row.get("benchmark_status", "")
        for row in summary_rows
        if row.get("benchmark_status", "")
    }
    evidence_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in summary_rows:
        dataset = row.get("dataset", "")
        category = row.get("category", "")
        metric = row.get("metric", "")
        if (
            dataset
            and category in {"quality", "utility"}
            and row.get("n", "0") not in {"", "0"}
            and row.get("benchmark_status", "") == "benchmark-valid"
        ):
            label = f"{dataset}/{row.get('method', '')}/{metric}"
            if not row.get("metric_source_paths", "").strip():
                add_reason(f"missing metric source paths for {label}")
            if not row.get("manifest_paths", "").strip():
                add_reason(f"missing manifest source paths for {label}")
            evidence_by_dataset[dataset].add(category)
    ready_datasets = {
        dataset
        for dataset, categories in evidence_by_dataset.items()
        if {"quality", "utility"}.issubset(categories)
    }
    quality_datasets = {
        dataset
        for dataset, categories in evidence_by_dataset.items()
        if "quality" in categories
    }
    utility_datasets = {
        dataset
        for dataset, categories in evidence_by_dataset.items()
        if "utility" in categories
    }

    if len(ready_datasets) < min_datasets:
        add_reason(
            "requires at least "
            f"{min_datasets} datasets with benchmark-valid quality and utility "
            f"evidence, found {len(ready_datasets)}"
        )
    if statuses != {"benchmark-valid"}:
        add_reason("all contributing rows must be benchmark-valid")
    if not quality_datasets:
        add_reason("no computable quality metrics found")
    if not utility_datasets:
        add_reason("no computable utility metrics found")
    return not reasons, reasons


def _best_rows(
    summary_rows: list[dict[str, str]], limit: int = 12
) -> list[dict[str, str]]:
    usable = [
        row for row in summary_rows if row.get("mean", "") not in {"", "nan", "NaN"}
    ]
    usable.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("metric", ""),
            int(row.get("rank") or 999),
        )
    )
    return [row for row in usable if row.get("rank") == "1"][:limit]


def _format_dataset_list(
    manifest: dict[str, Any], summary_rows: list[dict[str, str]]
) -> str:
    manifest_datasets = manifest.get("datasets")
    if isinstance(manifest_datasets, list) and manifest_datasets:
        names = [
            str(item.get("dataset") or item.get("name"))
            for item in manifest_datasets
            if isinstance(item, dict)
        ]
    else:
        names = sorted(
            {row.get("dataset", "") for row in summary_rows if row.get("dataset", "")}
        )
    return ", ".join(f"`{name}`" for name in names if name)


def build_draft(summary_rows: list[dict[str, str]], manifest: dict[str, Any]) -> str:
    ready, reasons = readiness(summary_rows, manifest)
    benchmark_id = str(
        manifest.get("benchmark_id") or "phm_genbench_six_dataset_submission_v1"
    )
    baseline = str(manifest.get("baseline_method") or "cfm_grid")
    dataset_list = _format_dataset_list(manifest, summary_rows)
    status = "SUBMISSION_READY" if ready else "NOT_SUBMISSION_READY"

    lines = [
        "# PHM-GenBench: Evidence-Gated Generative Benchmarking for PHM Signals",
        "",
        f"**Draft status:** `{status}`",
        f"**Benchmark ID:** `{benchmark_id}`",
        f"**Baseline:** `{baseline}`",
        "",
        "## Abstract",
        "",
    ]
    if ready:
        lines.append(
            "We evaluate conditional generative models for raw PHM vibration signals across "
            "six real datasets using quality, utility, efficiency, and leakage evidence. "
            "All reported claims are linked to benchmark-valid manifests and metric source files."
        )
    else:
        lines.append(
            "This draft records the planned PHM generative benchmark narrative, but it is not "
            "submission-ready because the required evidence chain is incomplete. No numerical "
            "claim in this draft should be treated as a benchmark result."
        )

    lines.extend(
        [
            "",
            "## Experimental Setting",
            "",
            f"The benchmark covers: {dataset_list or 'no datasets with evidence yet'}.",
            "Model conditions are restricted to `fault_label` and `domain_id`; load, rpm, "
            "system metadata, and sampling rate are recovered through the domain map for audit "
            "and reporting.",
            "",
            "## Metrics",
            "",
            "The evidence package groups metrics into temporal and spectral quality, "
            "distribution and diversity quality, TSTR/TRTS utility, efficiency, and leakage "
            "checks. FFT and spectral calculations are evaluation-only evidence and are not "
            "training losses.",
            "",
            "## Results",
            "",
        ]
    )

    best_rows = _best_rows(summary_rows)
    if best_rows:
        lines.extend(
            [
                "| Dataset | Metric | Best Method | Mean | Delta vs Baseline | Metric Source | Manifest Source |",
                "|---|---|---|---:|---:|---|---|",
            ]
        )
        for row in best_rows:
            metric_source = row.get("metric_source_paths", "").split(";")[0]
            manifest_source = row.get("manifest_paths", "").split(";")[0]
            lines.append(
                "| {dataset} | {metric} | {method} | {mean} | {delta} | {metric_source} | {manifest_source} |".format(
                    dataset=row.get("dataset", ""),
                    metric=row.get("metric", ""),
                    method=row.get("method", ""),
                    mean=row.get("mean", ""),
                    delta=row.get("delta_vs_baseline", ""),
                    metric_source=metric_source,
                    manifest_source=manifest_source,
                )
            )
    else:
        lines.append("No computable benchmark rows are available yet.")

    lines.extend(["", "## Evidence And Reproducibility", ""])
    if reasons:
        lines.append("The draft is blocked by the following evidence gaps:")
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append(
            "All included rows are benchmark-valid and cover the required dataset count."
        )

    missing_by_dataset: dict[str, int] = defaultdict(int)
    for row in summary_rows:
        try:
            missing_by_dataset[row.get("dataset", "")] += int(
                row.get("missing_count") or 0
            )
        except ValueError:
            continue
    if missing_by_dataset:
        lines.extend(["", "Metric missing counts by dataset:"])
        for dataset, count in sorted(missing_by_dataset.items()):
            lines.append(f"- `{dataset}`: {count}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Synthetic outputs remain exploratory unless complete manifest, protocol, "
            "normalization, leakage, and metric evidence is present. Missing utility metrics "
            "must be reported with structured reasons instead of being silently dropped.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_no_placeholders(text: str) -> None:
    found = [token for token in PLACEHOLDER_TOKENS if token in text]
    if found:
        raise ValueError(f"draft contains placeholder token(s): {', '.join(found)}")


def write_readiness_sidecars(
    output_dir: Path,
    *,
    summary_path: Path,
    manifest_path: Path | None,
    ready: bool,
    reasons: list[str],
) -> None:
    status = "SUBMISSION_READY" if ready else "NOT_SUBMISSION_READY"
    evidence_lines = [
        "# M2 Paper Evidence Gaps",
        "",
        f"Summary: `{summary_path}`",
        f"Manifest: `{manifest_path}`" if manifest_path is not None else "Manifest: not provided",
        "",
    ]
    if reasons:
        evidence_lines.append("Evidence gaps:")
        for reason in reasons:
            evidence_lines.append(f"- {reason}")
    else:
        evidence_lines.append("No evidence gaps were reported by the draft generator.")
    evidence_lines.append("")

    readiness_lines = [
        "# M2 Submission Readiness",
        "",
        f"Status: `{status}`",
        "",
    ]
    if reasons:
        readiness_lines.append("Reason:")
        readiness_lines.append("")
        for reason in reasons:
            readiness_lines.append(f"- {reason}")
    else:
        readiness_lines.append("Reason: all draft readiness gates passed.")
    readiness_lines.extend(
        [
            "",
            "Promotion rule:",
            "",
            "The draft can be marked `SUBMISSION_READY` only when the evidence "
            "covers the required datasets, all contributing rows are "
            "benchmark-valid, and source paths are traceable.",
            "",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence_gaps.md").write_text(
        "\n".join(evidence_lines), encoding="utf-8"
    )
    (output_dir / "submission_readiness.md").write_text(
        "\n".join(readiness_lines), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate PHM generative Markdown paper draft."
    )
    parser.add_argument("--summary", required=True, help="benchmark_effect_summary.csv")
    parser.add_argument(
        "--manifest", default=None, help="benchmark_effect_manifest.json"
    )
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--require-submission-ready", action="store_true")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    manifest_path = Path(args.manifest) if args.manifest else None
    manifest = read_manifest(manifest_path)
    input_gaps: list[str] = []
    if summary_path.exists():
        summary = read_summary(summary_path)
    else:
        summary = []
        input_gaps.append(f"required summary file not found: {summary_path}")
    if manifest_path is not None and not manifest_path.exists():
        input_gaps.append(f"manifest file not found: {manifest_path}")
    if input_gaps:
        manifest = {**manifest, "input_gaps": input_gaps}
    draft = build_draft(summary, manifest)
    assert_no_placeholders(draft)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draft, encoding="utf-8")
    ready, reasons = readiness(summary, manifest)
    write_readiness_sidecars(
        output.parent,
        summary_path=summary_path,
        manifest_path=manifest_path,
        ready=ready,
        reasons=reasons,
    )
    if args.require_submission_ready and not ready:
        for reason in reasons:
            print(f"[FAIL] {reason}", file=sys.stderr)
        return 2
    print(f"[OK] paper draft written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_gpu_queue import DEFAULT_QUEUE, build_launch_plan, expand_queue


REQUIRED_RUN_META_FIELDS = (
    "cuda_visible_devices",
    "gpu_model",
    "gpu_count",
    "seed",
    "dataset_split",
    "preprocessing_signature",
    "batch_size",
    "precision",
    "runtime",
    "command",
    "git_sha_or_submodule_sha",
    "config_path",
    "log_path",
    "metrics_path",
)

QUEUE_METADATA_TO_RUN_META = {
    "CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
    "GPU model": "gpu_model",
    "GPU count": "gpu_count",
    "seed": "seed",
    "dataset split": "dataset_split",
    "preprocessing signature": "preprocessing_signature",
    "batch size": "batch_size",
    "precision": "precision",
    "runtime": "runtime",
    "command": "command",
    "git SHA or submodule SHA": "git_sha_or_submodule_sha",
    "config path": "config_path",
    "log path": "log_path",
    "metrics path": "metrics_path",
    "OOM or failure reason if any": "oom_or_failure_reason",
}

CONDITIONAL_RUN_META_FIELDS = ("oom_or_failure_reason",)


@dataclass(frozen=True)
class ArtifactRecord:
    run_meta_path: str
    accepted: bool
    issues: Tuple[str, ...]
    queue_key: str


@dataclass(frozen=True)
class ArtifactGateReport:
    accepted: bool
    artifact_root: str
    records: Tuple[ArtifactRecord, ...]
    blockers: Tuple[str, ...]
    expected_queue_runs: int
    covered_queue_runs: int
    missing_queue_runs: Tuple[str, ...]
    queue_coverage_by_paper: Mapping[str, Mapping[str, int]]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _metadata_queue_key(data: Mapping[str, Any]) -> str:
    fields = (
        "source_queue_id",
        "paper_id",
        "phase",
        "entry_id",
        "cuda_visible_devices",
    )
    if any(data.get(field) in ("", None) for field in fields):
        return ""
    return "|".join(str(data[field]) for field in fields)


def _launch_queue_key(row: Any) -> str:
    return "|".join(
        (row.queue_id, row.paper_id, row.phase, row.entry_id, row.device)
    )


def _coverage_summary(
    expected_keys: Sequence[str],
    missing_keys: Sequence[str],
) -> Mapping[str, Mapping[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    missing = set(missing_keys)
    for key in expected_keys:
        parts = key.split("|")
        if len(parts) != 5:
            continue
        _, paper_id, phase, _, _ = parts
        paper = summary.setdefault(paper_id, {"expected": 0, "covered": 0, "missing": 0})
        paper["expected"] += 1
        phase_expected = f"{phase}_expected"
        phase_covered = f"{phase}_covered"
        phase_missing = f"{phase}_missing"
        paper[phase_expected] = paper.get(phase_expected, 0) + 1
        if key in missing:
            paper["missing"] += 1
            paper[phase_missing] = paper.get(phase_missing, 0) + 1
        else:
            paper["covered"] += 1
            paper[phase_covered] = paper.get(phase_covered, 0) + 1
    return summary


def _validate_run_meta(path: Path) -> ArtifactRecord:
    issues: List[str] = []
    data = _load_yaml(path)
    for field in REQUIRED_RUN_META_FIELDS:
        if field not in data or data[field] in ("", None):
            issues.append(f"missing {field}")
            continue
        if isinstance(data[field], str) and data[field].strip().upper().startswith("TODO"):
            issues.append(f"{field} still contains TODO")

    if data.get("accepted_evidence") is False:
        issues.append("accepted_evidence is false")

    cuda_visible_devices = str(data.get("cuda_visible_devices", ""))
    if cuda_visible_devices not in {"0", "1", "0,1"}:
        issues.append("cuda_visible_devices must be one of 0, 1, or 0,1")

    gpu_model = str(data.get("gpu_model", ""))
    if gpu_model and "4090" not in gpu_model:
        issues.append("gpu_model must record RTX 4090-class hardware")

    for path_field in ("metrics_path", "log_path", "config_path"):
        value = data.get(path_field)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if not candidate.exists():
            issues.append(f"{path_field} does not exist: {value}")

    return ArtifactRecord(
        run_meta_path=str(path),
        accepted=not issues,
        issues=tuple(issues),
        queue_key=_metadata_queue_key(data),
    )


def evaluate_artifact_gate(
    artifact_root: Path,
    queue_path: Optional[Path] = None,
    require_queue_coverage: bool = False,
) -> ArtifactGateReport:
    run_meta_paths = tuple(sorted(artifact_root.rglob("run_meta.yaml")))
    blockers: List[str] = []
    if not artifact_root.exists():
        blockers.append(f"artifact root does not exist: {artifact_root}")
    if not run_meta_paths:
        blockers.append(f"no run_meta.yaml files found under {artifact_root}")

    records = tuple(_validate_run_meta(path) for path in run_meta_paths)
    for record in records:
        if not record.accepted:
            blockers.append(f"{record.run_meta_path}: {len(record.issues)} issues")

    expected_queue_runs: Tuple[str, ...] = ()
    missing_queue_runs: Tuple[str, ...] = ()
    if require_queue_coverage:
        if queue_path is None:
            blockers.append("queue coverage was required but no queue path was provided")
        else:
            expected_queue_runs = tuple(
                sorted(_launch_queue_key(row) for row in build_launch_plan(expand_queue(queue_path)))
            )
            covered = {
                record.queue_key
                for record in records
                if record.accepted and record.queue_key
            }
            missing_queue_runs = tuple(
                key for key in expected_queue_runs if key not in covered
            )
            for record in records:
                if record.accepted and not record.queue_key:
                    blockers.append(
                        f"{record.run_meta_path}: missing queue coverage identifiers"
                    )
            if missing_queue_runs:
                blockers.append(
                    "queue coverage incomplete: "
                    f"{len(missing_queue_runs)} launch rows missing accepted run_meta.yaml"
                )

    return ArtifactGateReport(
        accepted=not blockers and bool(records),
        artifact_root=str(artifact_root),
        records=records,
        blockers=tuple(blockers),
        expected_queue_runs=len(expected_queue_runs),
        covered_queue_runs=len(expected_queue_runs) - len(missing_queue_runs),
        missing_queue_runs=missing_queue_runs,
        queue_coverage_by_paper=_coverage_summary(expected_queue_runs, missing_queue_runs),
    )


def render_markdown(report: ArtifactGateReport) -> str:
    lines = [
        "# UXFD Artifact Gate",
        "",
        f"- Accepted: `{report.accepted}`",
        f"- Artifact root: `{report.artifact_root}`",
        f"- Run metadata files: `{len(report.records)}`",
        f"- Queue coverage: `{report.covered_queue_runs}/{report.expected_queue_runs}`",
        f"- Blockers: `{len(report.blockers)}`",
        "",
        "| run_meta.yaml | Accepted | Issues |",
        "|---|---:|---:|",
    ]
    for record in report.records:
        lines.append(
            f"| `{record.run_meta_path}` | `{record.accepted}` | {len(record.issues)} |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.blockers:
        lines.append(f"- {blocker}")
    if report.queue_coverage_by_paper:
        lines.extend(["", "## Queue Coverage By Paper", ""])
        lines.append("| Paper | Covered | Missing | Expected |")
        lines.append("|---|---:|---:|---:|")
        for paper_id, counts in sorted(report.queue_coverage_by_paper.items()):
            lines.append(
                f"| `{paper_id}` | {counts.get('covered', 0)} | "
                f"{counts.get('missing', 0)} | {counts.get('expected', 0)} |"
            )
    return "\n".join(lines) + "\n"


def build_payload(report: ArtifactGateReport) -> Mapping[str, Any]:
    return asdict(report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate UXFD accepted run artifacts")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--require-queue-coverage", action="store_true")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_artifact_gate(
        args.artifact_root,
        queue_path=args.queue,
        require_queue_coverage=args.require_queue_coverage,
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

    if report.accepted or args.allow_not_ready:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

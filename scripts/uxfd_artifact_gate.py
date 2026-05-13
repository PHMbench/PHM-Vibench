from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_gpu_queue import DEFAULT_QUEUE, build_launch_plan, expand_queue


REQUIRED_RUN_META_FIELDS = (
    "source_queue_id",
    "paper_id",
    "phase",
    "entry_id",
    "cuda_visible_devices",
    "gpu_model",
    "gpu_count",
    "seed",
    "dataset_split",
    "preprocessing_signature",
    "batch_size",
    "precision",
    "runtime",
    "evidence_level",
    "command",
    "git_sha_or_submodule_sha",
    "source_tree_status",
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
    "evidence level": "evidence_level",
    "command": "command",
    "git SHA or submodule SHA": "git_sha_or_submodule_sha",
    "source tree status": "source_tree_status",
    "config path": "config_path",
    "log path": "log_path",
    "metrics path": "metrics_path",
    "OOM or failure reason if any": "oom_or_failure_reason",
}

CONDITIONAL_RUN_META_FIELDS = ("oom_or_failure_reason",)
DISALLOWED_ACCEPTED_EVIDENCE_MARKERS = (
    "smoke",
    "demo",
    "dummy",
    "template",
    "pending",
)
DISALLOWED_SHA_PROVENANCE_MARKERS = (
    "dirty",
    "modified",
    "unknown",
    "uncommitted",
)
PREPROCESSING_SIGNATURE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
RUNTIME_PATTERN = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")
ACCEPTED_PRECISION_VALUES = ("fp32", "tf32", "fp16", "bf16", "amp")
ACCEPTED_EVIDENCE_LEVEL_VALUES = ("accepted_same_protocol",)
PROTOCOL_EVIDENCE_FIELDS = (
    "dataset_split",
    "preprocessing_signature",
    "evidence_level",
    "command",
    "config_path",
    "log_path",
    "metrics_path",
)


@dataclass(frozen=True)
class ArtifactRecord:
    run_meta_path: str
    accepted: bool
    issues: Tuple[str, ...]
    queue_key: str
    queue_seed_key: str


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


def _metadata_queue_seed_key(data: Mapping[str, Any]) -> str:
    queue_key = _metadata_queue_key(data)
    seed = _coerce_integer(data.get("seed"))
    if not queue_key or seed is None:
        return ""
    return f"{queue_key}|seed={seed}"


def _launch_queue_key(row: Any) -> str:
    return "|".join((row.queue_id, row.paper_id, row.phase, row.entry_id, row.device))


def _top_representative_queue_key(row: Any) -> str:
    return "|".join((row.queue_id, row.paper_id, row.phase, row.entry_id, "0,1"))


def _expected_queue_commands(queue_path: Path) -> Mapping[str, str]:
    rows = expand_queue(queue_path)
    expected: Dict[str, str] = {
        _launch_queue_key(row): row.command for row in build_launch_plan(rows)
    }
    expected.update(
        {
            _top_representative_queue_key(row): row.command
            for row in rows
            if row.phase == "top_representatives"
        }
    )
    return expected


def _minimum_seeds_by_queue_key(queue_path: Path) -> Mapping[str, int]:
    queue = _load_yaml(queue_path)
    paper_minimums = {
        str(item.get("paper_id", "")): int(item.get("minimum_seeds", 1))
        for item in queue.get("paper_queue", ())
        if isinstance(item, Mapping)
    }
    return {
        key: max(1, paper_minimums.get(key.split("|")[1], 1))
        for key in _expected_queue_commands(queue_path)
    }


def _expected_queue_keys(queue_path: Path) -> Tuple[str, ...]:
    return tuple(sorted(_expected_queue_commands(queue_path)))


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
        paper = summary.setdefault(
            paper_id, {"expected": 0, "covered": 0, "missing": 0}
        )
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


def _validate_metrics_file(path: Path) -> Tuple[str, ...]:
    issues: List[str] = []
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return (f"metrics_path JSON is not parseable: {exc.msg}",)
        if payload in ({}, [], None):
            issues.append("metrics_path JSON must contain at least one metric")
        elif not _contains_numeric_value(payload):
            issues.append("metrics_path JSON must contain at least one numeric metric")
    elif path.suffix == ".csv":
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        nonempty_rows = [row for row in rows if any(cell.strip() for cell in row)]
        if len(nonempty_rows) < 2:
            issues.append(
                "metrics_path CSV must contain a header and at least one data row"
            )
        elif not any(
            _is_numeric_cell(cell) for row in nonempty_rows[1:] for cell in row
        ):
            issues.append("metrics_path CSV must contain at least one numeric metric")
    else:
        issues.append("metrics_path must point to .json or .csv")
    return tuple(issues)


def _contains_numeric_value(payload: Any) -> bool:
    if isinstance(payload, bool):
        return False
    if isinstance(payload, (int, float)):
        return True
    if isinstance(payload, Mapping):
        return any(_contains_numeric_value(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_numeric_value(value) for value in payload)
    return False


def _is_numeric_cell(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _coerce_integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _runtime_seconds(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    match = RUNTIME_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _resolve_referenced_artifact_path(
    run_meta_path: Path,
    field: str,
    value: Any,
) -> Tuple[Optional[Path], Tuple[str, ...]]:
    issues: List[str] = []
    reference = Path(str(value))
    if reference.is_absolute():
        return None, (f"{field} must be relative to the run_meta.yaml directory",)

    run_dir = run_meta_path.parent.resolve()
    candidate = (run_meta_path.parent / reference).resolve()
    if not candidate.is_relative_to(run_dir):
        return None, (f"{field} must stay inside the run_meta.yaml directory",)
    return candidate, tuple(issues)


def _command_cuda_visible_devices(command: str) -> str:
    match = re.search(r"(?:^|\s)CUDA_VISIBLE_DEVICES=([0-9,]+)(?:\s|$)", command)
    return match.group(1) if match else ""


def _validate_run_meta(path: Path) -> ArtifactRecord:
    issues: List[str] = []
    data = _load_yaml(path)
    for field in REQUIRED_RUN_META_FIELDS:
        if field not in data or data[field] in ("", None):
            issues.append(f"missing {field}")
            continue
        if isinstance(data[field], str) and data[field].strip().upper().startswith(
            "TODO"
        ):
            issues.append(f"{field} still contains TODO")

    if data.get("accepted_evidence") is not True:
        issues.append("accepted_evidence must be true")

    cuda_visible_devices = str(data.get("cuda_visible_devices", ""))
    if cuda_visible_devices not in {"0", "1", "0,1"}:
        issues.append("cuda_visible_devices must be one of 0, 1, or 0,1")

    phase = str(data.get("phase", ""))
    command = str(data.get("command", ""))
    command_devices = _command_cuda_visible_devices(command)
    if phase != "top_representatives" and cuda_visible_devices in {"0", "1", "0,1"}:
        if not command_devices:
            issues.append("command must include CUDA_VISIBLE_DEVICES for non-top run")
        elif command_devices != cuda_visible_devices:
            issues.append(
                "command CUDA_VISIBLE_DEVICES="
                f"{command_devices} does not match cuda_visible_devices={cuda_visible_devices}"
            )

    seed = _coerce_integer(data.get("seed"))
    if seed is None or seed < 0:
        issues.append("seed must be a non-negative integer")

    batch_size = _coerce_integer(data.get("batch_size"))
    if batch_size is None or batch_size <= 0:
        issues.append("batch_size must be a positive integer")

    runtime_seconds = _runtime_seconds(data.get("runtime"))
    if runtime_seconds is None or runtime_seconds <= 0:
        issues.append("runtime must be positive HH:MM:SS")

    precision = str(data.get("precision", "")).strip().lower()
    if precision and precision not in ACCEPTED_PRECISION_VALUES:
        issues.append("precision must be one of fp32, tf32, fp16, bf16, amp")

    evidence_level = str(data.get("evidence_level", "")).strip().lower()
    if evidence_level and evidence_level not in ACCEPTED_EVIDENCE_LEVEL_VALUES:
        issues.append("evidence_level must be accepted_same_protocol")

    gpu_count = _coerce_integer(data.get("gpu_count"))
    if gpu_count is None:
        issues.append("gpu_count must be an integer")
    else:
        expected_gpu_count = 2 if cuda_visible_devices == "0,1" else 1
        if (
            cuda_visible_devices in {"0", "1", "0,1"}
            and gpu_count != expected_gpu_count
        ):
            issues.append(
                f"gpu_count must be {expected_gpu_count} for cuda_visible_devices={cuda_visible_devices}"
            )

    gpu_model = str(data.get("gpu_model", ""))
    if gpu_model and "RTX 4090" not in gpu_model:
        issues.append("gpu_model must record RTX 4090-class hardware")

    source_tree_status = str(data.get("source_tree_status", "")).strip().lower()
    if source_tree_status and source_tree_status != "clean":
        issues.append("source_tree_status must be clean")

    sha_provenance = str(data.get("git_sha_or_submodule_sha", "")).strip().lower()
    sha_marker = next(
        (item for item in DISALLOWED_SHA_PROVENANCE_MARKERS if item in sha_provenance),
        "",
    )
    if sha_marker:
        issues.append(f"git_sha_or_submodule_sha must not contain {sha_marker}")

    preprocessing_signature = str(data.get("preprocessing_signature", "")).strip()
    if preprocessing_signature and not PREPROCESSING_SIGNATURE_PATTERN.fullmatch(
        preprocessing_signature
    ):
        issues.append("preprocessing_signature must match sha256:<64 lowercase hex>")

    for field in PROTOCOL_EVIDENCE_FIELDS:
        value = str(data.get(field, ""))
        lowered = value.lower()
        marker = next(
            (item for item in DISALLOWED_ACCEPTED_EVIDENCE_MARKERS if item in lowered),
            "",
        )
        if marker:
            issues.append(f"{field} must not reference {marker} evidence")

    for path_field in ("metrics_path", "log_path", "config_path"):
        value = data.get(path_field)
        if not value:
            continue
        candidate, path_issues = _resolve_referenced_artifact_path(
            path, path_field, value
        )
        issues.extend(path_issues)
        if candidate is None:
            continue
        if not candidate.exists():
            issues.append(f"{path_field} does not exist: {value}")
            continue
        if path_field == "metrics_path":
            issues.extend(_validate_metrics_file(candidate))

    return ArtifactRecord(
        run_meta_path=str(path),
        accepted=not issues,
        issues=tuple(issues),
        queue_key=_metadata_queue_key(data),
        queue_seed_key=_metadata_queue_seed_key(data),
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
            blockers.append(
                "queue coverage was required but no queue path was provided"
            )
        else:
            expected_commands = _expected_queue_commands(queue_path)
            minimum_seeds_by_key = _minimum_seeds_by_queue_key(queue_path)
            expected_queue_runs = tuple(sorted(expected_commands))
            expected = set(expected_queue_runs)
            accepted_keys = [
                record.queue_key
                for record in records
                if record.accepted and record.queue_key
            ]
            covered = {
                record.queue_key
                for record in records
                if record.accepted and record.queue_key in expected
            }
            missing_queue_runs = tuple(
                key for key in expected_queue_runs if key not in covered
            )
            unknown_keys = tuple(
                sorted(key for key in accepted_keys if key not in expected)
            )
            accepted_seed_keys = [
                record.queue_seed_key
                for record in records
                if record.accepted and record.queue_seed_key
            ]
            seen: Dict[str, int] = {}
            for key in accepted_seed_keys:
                seen[key] = seen.get(key, 0) + 1
            duplicate_keys = tuple(
                sorted(key for key, count in seen.items() if count > 1)
            )
            seeds_by_key: Dict[str, set[int]] = {}
            for record in records:
                if record.accepted and not record.queue_key:
                    blockers.append(
                        f"{record.run_meta_path}: missing queue coverage identifiers"
                    )
                if record.accepted and record.queue_key in expected:
                    run_meta = _load_yaml(Path(record.run_meta_path))
                    command = str(run_meta.get("command", ""))
                    expected_command = expected_commands[record.queue_key]
                    if command != expected_command:
                        blockers.append(
                            f"{record.run_meta_path}: command does not match queue command"
                        )
                    seed = _coerce_integer(run_meta.get("seed"))
                    if seed is not None:
                        seeds_by_key.setdefault(record.queue_key, set()).add(seed)
            seed_shortfall_keys = tuple(
                key
                for key in expected_queue_runs
                if key in covered
                and len(seeds_by_key.get(key, set())) < minimum_seeds_by_key[key]
            )
            if unknown_keys:
                blockers.append(
                    "queue coverage contains unknown accepted run_meta.yaml keys: "
                    f"{len(unknown_keys)}"
                )
            if duplicate_keys:
                blockers.append(
                    "queue coverage contains duplicate accepted run_meta.yaml queue+seed keys: "
                    f"{len(duplicate_keys)}"
                )
            if seed_shortfall_keys:
                blockers.append(
                    "queue seed coverage incomplete: "
                    f"{len(seed_shortfall_keys)} queue coverage rows below minimum_seeds"
                )
            if missing_queue_runs:
                blockers.append(
                    "queue coverage incomplete: "
                    f"{len(missing_queue_runs)} queue coverage rows missing accepted run_meta.yaml"
                )

    return ArtifactGateReport(
        accepted=not blockers and bool(records),
        artifact_root=str(artifact_root),
        records=records,
        blockers=tuple(blockers),
        expected_queue_runs=len(expected_queue_runs),
        covered_queue_runs=len(expected_queue_runs) - len(missing_queue_runs),
        missing_queue_runs=missing_queue_runs,
        queue_coverage_by_paper=_coverage_summary(
            expected_queue_runs, missing_queue_runs
        ),
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

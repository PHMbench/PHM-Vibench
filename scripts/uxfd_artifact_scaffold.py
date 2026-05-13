from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_gpu_queue import (
    DEFAULT_QUEUE,
    QueueLaunchCommand,
    build_launch_plan,
    expand_queue,
    validate_queue,
)


DEFAULT_TEMPLATE_ROOT = Path("paper/UXFD_paper/results/accepted_run_templates")


@dataclass(frozen=True)
class ArtifactTemplateRecord:
    queue_id: str
    paper_id: str
    phase: str
    entry_id: str
    device: str
    workdir: str
    template_path: str
    command: str


@dataclass(frozen=True)
class ArtifactScaffoldReport:
    template_root: str
    records: Tuple[ArtifactTemplateRecord, ...]
    validation_can_execute: bool
    validation_resource_reason: str
    note: str


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "unnamed"


def _extract_config_path(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return "TODO: record config path"
    for index, part in enumerate(parts):
        if part == "--config" and index + 1 < len(parts):
            return parts[index + 1]
    return "TODO: record config path"


def _template_dir(root: Path, row: QueueLaunchCommand) -> Path:
    return (
        root
        / _slug(row.paper_id)
        / _slug(row.phase)
        / f"{_slug(row.entry_id)}__gpu{_slug(row.device)}"
    )


def _run_meta_template(row: QueueLaunchCommand) -> Mapping[str, Any]:
    queue_config_path = _extract_config_path(row.command)
    return {
        "accepted_evidence": False,
        "source_queue_id": row.queue_id,
        "paper_id": row.paper_id,
        "phase": row.phase,
        "entry_id": row.entry_id,
        "label": row.label,
        "workdir": row.workdir,
        "cuda_visible_devices": row.device,
        "gpu_model": "TODO: NVIDIA GeForce RTX 4090",
        "gpu_count": 2 if row.device == "0,1" else 1,
        "seed": "TODO: record seed",
        "dataset_split": "TODO: record dataset split",
        "preprocessing_signature": "TODO: record preprocessing signature",
        "batch_size": "TODO: record batch size",
        "precision": "TODO: record precision",
        "runtime": "TODO: record wall-clock runtime",
        "command": row.command,
        "original_command": row.original_command,
        "queue_config_path": queue_config_path,
        "git_sha_or_submodule_sha": "TODO: record parent git SHA and paper submodule SHA",
        "source_tree_status": "TODO: clean after git status --porcelain",
        "config_path": "config.yaml",
        "log_path": "run.log",
        "metrics_path": "metrics.json",
        "oom_or_failure_reason": "",
    }


def create_scaffold(
    output_root: Path = DEFAULT_TEMPLATE_ROOT,
    queue_path: Path = DEFAULT_QUEUE,
) -> ArtifactScaffoldReport:
    queue_rows = expand_queue(queue_path)
    rows = list(build_launch_plan(queue_rows))
    rows.extend(
        QueueLaunchCommand(
            queue_id=row.queue_id,
            paper_id=row.paper_id,
            phase=row.phase,
            entry_id=row.entry_id,
            label=row.label,
            device="0,1",
            workdir=".",
            command=row.command,
            original_command=row.command,
            status=row.status,
        )
        for row in queue_rows
        if row.phase == "top_representatives"
    )
    validation = validate_queue(queue_path)
    records: List[ArtifactTemplateRecord] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        run_dir = _template_dir(output_root, row)
        run_dir.mkdir(parents=True, exist_ok=True)
        template_path = run_dir / "run_meta.template.yaml"
        template_path.write_text(
            yaml.safe_dump(_run_meta_template(row), sort_keys=False),
            encoding="utf-8",
        )
        (run_dir / "README.md").write_text(
            "\n".join(
                [
                    "# UXFD Accepted Run Template",
                    "",
                    "This directory is a scaffold, not accepted evidence.",
                    "After a real run, fill the template, rename it to `run_meta.yaml`,",
                    "place the referenced log/metrics/config files beside it, and run",
                    "`python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs`.",
                    "",
                    f"- Queue: `{row.queue_id}`",
                    f"- Paper: `{row.paper_id}`",
                    f"- Phase: `{row.phase}`",
                    f"- Entry: `{row.entry_id}`",
                    f"- Device: `{row.device}`",
                    f"- Workdir: `{row.workdir}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        records.append(
            ArtifactTemplateRecord(
                queue_id=row.queue_id,
                paper_id=row.paper_id,
                phase=row.phase,
                entry_id=row.entry_id,
                device=row.device,
                workdir=row.workdir,
                template_path=str(template_path),
                command=row.command,
            )
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        render_markdown(records, output_root), encoding="utf-8"
    )

    return ArtifactScaffoldReport(
        template_root=str(output_root),
        records=tuple(records),
        validation_can_execute=validation.can_execute,
        validation_resource_reason=validation.resource_reason,
        note="templates only; not accepted evidence",
    )


def render_markdown(records: Sequence[ArtifactTemplateRecord], root: Path) -> str:
    lines = [
        "# UXFD Accepted Run Artifact Templates",
        "",
        f"- Template root: `{root}`",
        f"- Templates: `{len(records)}`",
        "- Status: templates only; not accepted evidence.",
        (
            "- Accepted metrics rule: `metrics.json` or `metrics.csv` must include "
            "at least one numeric metric; status-only payloads are rejected."
        ),
        "- Source-tree rule: accepted runs must set `source_tree_status: clean`.",
        (
            "- Run-control rule: `seed` must be a non-negative integer and "
            "`batch_size` must be a positive integer."
        ),
        "- Runtime rule: `runtime` must be a positive `HH:MM:SS` duration.",
        (
            "- Protocol-signature rule: `preprocessing_signature` must match "
            "`sha256:<64 lowercase hex>`."
        ),
        (
            "- Provenance rule: `git_sha_or_submodule_sha` must be a concrete SHA "
            "record without dirty, modified, unknown, or uncommitted markers."
        ),
        "",
        "| Queue | Paper | Phase | Entry | GPU | Template |",
        "|---|---|---|---|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record.queue_id}` | `{record.paper_id}` | `{record.phase}` | "
            f"`{record.entry_id}` | `{record.device}` | `{record.template_path}` |"
        )
    return "\n".join(lines) + "\n"


def build_payload(report: ArtifactScaffoldReport) -> Mapping[str, Any]:
    return asdict(report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create UXFD accepted-run metadata templates"
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = create_scaffold(output_root=args.output_root, queue_path=args.queue)
    if args.format == "json":
        output = json.dumps(build_payload(report), indent=2) + "\n"
    else:
        output = render_markdown(report.records, Path(report.template_root))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

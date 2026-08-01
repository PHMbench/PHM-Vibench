from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from phmfactory.config import resolve_config
from phmfactory.pipelines import PIPELINE_DESCRIPTORS
from scripts.gen_config_atlas import RegistryRow, read_registry


@dataclass(frozen=True)
class SupportedDemo:
    config_id: str
    path: str
    description: str
    pipeline: str
    data_base: str
    model: str
    embedding: str
    backbone: str
    task_head: str
    task: str
    trainer: str
    status: str


def _value(mapping: dict[str, Any], key: str, default: str = "-") -> str:
    value = mapping.get(key)
    return str(value) if value not in (None, "") else default


def supported_demos(rows: Iterable[RegistryRow]) -> list[SupportedDemo]:
    result: list[SupportedDemo] = []
    for row in rows:
        if row.category != "demo" or row.status != "sanity_ok":
            continue
        resolved = resolve_config(row.path)
        if resolved.pipeline != row.pipeline:
            raise ValueError(
                f"registry pipeline mismatch for {row.config_id}: "
                f"registry={row.pipeline!r}, resolved={resolved.pipeline!r}"
            )
        model = resolved.data.get("model") or {}
        task = resolved.data.get("task") or {}
        trainer = resolved.data.get("trainer") or {}
        result.append(
            SupportedDemo(
                config_id=row.config_id,
                path=row.path,
                description=row.description,
                pipeline=resolved.pipeline,
                data_base=Path(row.base_data).stem,
                model=f"{_value(model, 'type')}/{_value(model, 'name')}",
                embedding=_value(model, "embedding"),
                backbone=_value(model, "backbone"),
                task_head=_value(model, "task_head"),
                task=f"{_value(task, 'type')}/{_value(task, 'name')}",
                trainer=_value(trainer, "name"),
                status=row.status,
            )
        )
    if not result:
        raise ValueError("no category=demo,status=sanity_ok rows found")
    return sorted(result, key=lambda item: item.config_id)


def _code_list(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in sorted(set(values)))


def render_components(demos: list[SupportedDemo]) -> str:
    lines = [
        "# Supported Components for the PHMFactory v0.3 Pre-release",
        "",
        "> Generated from `phmfactory.pipelines.PIPELINE_DESCRIPTORS`, "
        "`configs/config_registry.csv`, and resolved maintained configs.",
        "",
        "Re-generate:",
        "",
        "```bash",
        "python -m scripts.gen_support_matrix",
        "```",
        "",
        "PHMFactory distinguishes three claims:",
        "",
        "```text",
        "discoverable  = a canonical Pipeline or registry entry exists",
        "runnable      = the public control plane permits execution",
        "supported     = a maintained combination has current smoke evidence",
        "```",
        "",
        "The required relationship is:",
        "",
        "```text",
        "supported ⊆ runnable ⊆ discoverable",
        "```",
        "",
        "A source file, importable module, registry row, or explicit experimental "
        "opt-in is not a release-support claim.",
        "",
        "## Pipeline maturity",
        "",
        "| Pipeline | Maturity | Default public access | Reason |",
        "|---|---|---:|---|",
    ]
    for name, descriptor in PIPELINE_DESCRIPTORS.items():
        access = "explicit opt-in" if descriptor.opt_in_required else "yes"
        reason = descriptor.reason or "-"
        lines.append(
            f"| `{name}` | `{descriptor.maturity}` | {access} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## Evidence-derived maintained surface",
            "",
            "| Surface | Values derived from `sanity_ok` demos |",
            "|---|---|",
            f"| Pipelines | {_code_list(d.pipeline for d in demos)} |",
            f"| Data bases | {_code_list(d.data_base for d in demos)} |",
            f"| Models | {_code_list(d.model for d in demos)} |",
            f"| Embeddings | {_code_list(d.embedding for d in demos)} |",
            f"| Backbones | {_code_list(d.backbone for d in demos)} |",
            f"| Task heads | {_code_list(d.task_head for d in demos)} |",
            f"| Tasks | {_code_list(d.task for d in demos)} |",
            f"| Trainers | {_code_list(d.trainer for d in demos)} |",
            "",
            "Exact supported executions are generated in `SUPPORTED_COMBINATIONS.md`.",
            "",
            "## Support boundaries",
            "",
            "- `sanity_ok` is bounded smoke evidence, not benchmark performance.",
            "- Model/task registry discovery does not imply Cartesian-product compatibility.",
            "- Pipeline 03 and Pipeline 04 are not release-supported.",
            "- Pipeline 05, Pipeline 06, and Pipeline_ID remain outside the maintained "
            "release combination table unless a `sanity_ok` demo is added.",
            "- Historical and paper-only configs are not promoted by this generator.",
            "- External dataset redistribution and availability are separate source-license "
            "questions.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_combinations(demos: list[SupportedDemo]) -> str:
    lines = [
        "# Supported Combinations for the PHMFactory v0.3 Pre-release",
        "",
        "> Generated from `configs/config_registry.csv` rows with "
        "`category=demo,status=sanity_ok` and their fully resolved configurations.",
        "",
        "Re-generate:",
        "",
        "```bash",
        "python -m scripts.gen_support_matrix",
        "```",
        "",
        "| Registry id | Config | Pipeline | Data base | Model | Task | Trainer | Evidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for demo in demos:
        lines.append(
            f"| `{demo.config_id}` | `{demo.path}` | `{demo.pipeline}` | "
            f"`{demo.data_base}` | `{demo.model}` | `{demo.task}` | "
            f"`{demo.trainer}` | `{demo.status}` |"
        )
    lines.extend(
        [
            "",
            "Current evidence is one-epoch or otherwise bounded smoke evidence for the "
            "exact registered path. It validates configuration resolution, factory "
            "assembly, runtime execution, checkpoint/test flow where applicable, and the "
            "current invocation manifest contract. It does not claim benchmark performance.",
            "",
            "## Interpretation",
            "",
            "A combination is release-supported only when the registry row remains "
            "`sanity_ok`, the path resolves, the registry Pipeline matches the resolved "
            "Pipeline, and repository gates continue to pass. Any unlisted combination is "
            "discoverable or experimental at most until it receives its own reviewed evidence.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PHMFactory support matrices")
    parser.add_argument("--registry", default="configs/config_registry.csv")
    parser.add_argument("--components-out", default="SUPPORTED_COMPONENTS.md")
    parser.add_argument("--combinations-out", default="SUPPORTED_COMBINATIONS.md")
    args = parser.parse_args(argv)

    rows = read_registry(Path(args.registry))
    demos = supported_demos(rows)
    Path(args.components_out).write_text(render_components(demos), encoding="utf-8")
    Path(args.combinations_out).write_text(render_combinations(demos), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

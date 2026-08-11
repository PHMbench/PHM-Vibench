from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from phmfactory.config import resolve_config
from phmfactory.pipelines import PIPELINE_DESCRIPTORS
from scripts.gen_config_atlas import RegistryRow, read_registry


@dataclass(frozen=True)
class VerifiedDemo:
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
    execution_status: str
    protocol_status: str


def _value(mapping: dict[str, Any], key: str, default: str = "-") -> str:
    value = mapping.get(key)
    return str(value) if value not in (None, "") else default


def verified_demos(rows: Iterable[RegistryRow]) -> list[VerifiedDemo]:
    result: list[VerifiedDemo] = []
    for row in rows:
        if row.category != "demo" or row.status != "sanity_ok":
            continue
        if not row.protocol_status:
            raise ValueError(
                f"demo {row.config_id} is missing protocol_status; "
                "execution smoke evidence must not imply scientific validity"
            )
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
            VerifiedDemo(
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
                execution_status=row.status,
                protocol_status=row.protocol_status,
            )
        )
    if not result:
        raise ValueError("no category=demo,status=sanity_ok rows found")
    return sorted(result, key=lambda item: item.config_id)


def _code_list(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in sorted(set(values)))


def render_components(demos: list[VerifiedDemo]) -> str:
    lines = [
        "# Execution-Verified Components for the PHMFactory v0.3 Pre-release",
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
        "PHMFactory separates software execution evidence from scientific protocol validity:",
        "",
        "```text",
        "discoverable     = a canonical Pipeline or registry entry exists",
        "runnable         = the public control plane permits execution",
        "smoke-verified   = the exact maintained command has bounded execution evidence",
        "protocol-valid   = the complete data/split/task/metric combination satisfies its scientific protocol",
        "```",
        "",
        "The software relationship is:",
        "",
        "```text",
        "smoke-verified ⊆ runnable ⊆ discoverable",
        "```",
        "",
        "Protocol validity is a separate property of a complete experiment combination. "
        "It is not inferred from component importability, Pipeline maturity, or a successful smoke run.",
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
            "## Execution-verified maintained surface",
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
            f"| Protocol statuses | {_code_list(d.protocol_status for d in demos)} |",
            "",
            "Exact execution-smoke combinations are generated in `SUPPORTED_COMBINATIONS.md`.",
            "",
            "## Boundaries",
            "",
            "- `sanity_ok` means bounded execution smoke only.",
            "- `protocol_status=smoke_only` forbids benchmark or algorithm-validity claims.",
            "- Model/task registry discovery does not imply Cartesian-product compatibility.",
            "- Pipeline 03 and Pipeline 04 remain experimental rather than maintained execution paths.",
            "- Pipeline 05, Pipeline 06, and Pipeline_ID remain outside this exact smoke table unless a reviewed demo is added.",
            "- Historical and paper-only configs are not promoted by this generator.",
            "- External dataset redistribution and availability are separate source-license questions.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_combinations(demos: list[VerifiedDemo]) -> str:
    lines = [
        "# Execution-Verified Combinations for the PHMFactory v0.3 Pre-release",
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
        "| Registry id | Config | Pipeline | Data base | Model | Task | Trainer | Execution evidence | Protocol status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for demo in demos:
        lines.append(
            f"| `{demo.config_id}` | `{demo.path}` | `{demo.pipeline}` | "
            f"`{demo.data_base}` | `{demo.model}` | `{demo.task}` | "
            f"`{demo.trainer}` | `{demo.execution_status}` | "
            f"`{demo.protocol_status}` |"
        )
    lines.extend(
        [
            "",
            "Current evidence is one-epoch or otherwise bounded execution evidence for the "
            "exact registered path. It validates configuration resolution, factory "
            "assembly, runtime execution, checkpoint/test flow where applicable, and the "
            "current run-record contract. It does not establish benchmark validity.",
            "",
            "## Interpretation",
            "",
            "`execution_status=sanity_ok` says that the exact command has current smoke "
            "evidence. `protocol_status=smoke_only` says that its split, statistical "
            "independence, task semantics, and metric protocol have not yet been promoted "
            "to a scientific baseline. The two statuses are independent and must not be "
            "collapsed into one support claim.",
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
    demos = verified_demos(rows)
    Path(args.components_out).write_text(render_components(demos), encoding="utf-8")
    Path(args.combinations_out).write_text(render_combinations(demos), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

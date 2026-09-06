from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from phmfactory.config import resolve_config
from phmfactory.pipelines import PIPELINE_DESCRIPTORS
from scripts.gen_config_atlas import RegistryRow, read_registry


_MAINTAINED_CATEGORIES = frozenset({"demo", "baseline"})


@dataclass(frozen=True)
class VerifiedDemo:
    config_id: str
    category: str
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
    """Resolve every maintained config with current execution evidence."""

    result: list[VerifiedDemo] = []
    for row in rows:
        if row.category not in _MAINTAINED_CATEGORIES or row.status != "sanity_ok":
            continue
        if not row.protocol_status:
            raise ValueError(
                f"maintained config {row.config_id} is missing protocol_status; "
                "execution evidence must not imply scientific validity"
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
                category=row.category,
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
        raise ValueError(
            "no category in {demo, baseline}, status=sanity_ok rows found"
        )
    return sorted(result, key=lambda item: item.config_id)


def _code_list(values: Iterable[str]) -> str:
    concrete = sorted({value for value in values if value not in ("", "-")})
    if not concrete:
        return "-"
    return ", ".join(f"`{value}`" for value in concrete)


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
        "execution-verified = the exact maintained command has current bounded or baseline evidence",
        "baseline-valid   = the exact complete experiment passed its declared scientific protocol",
        "```",
        "",
        "The software relationship is:",
        "",
        "```text",
        "execution-verified ⊆ runnable ⊆ discoverable",
        "```",
        "",
        "`baseline-valid` is a separate property of one complete experiment combination. "
        "It is not inferred from component importability, Pipeline maturity, or another "
        "configuration's successful run.",
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
            "| Surface | Values derived from `sanity_ok` maintained configs |",
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
            "Exact execution-verified combinations are generated in `SUPPORTED_COMBINATIONS.md`.",
            "",
            "## Boundaries",
            "",
            "- `execution_status=sanity_ok` means the exact registered command has current evidence; its scientific scope is determined separately by `protocol_status`.",
            "- `protocol_status=smoke_only` forbids benchmark or algorithm-validity claims.",
            "- `protocol_status=baseline_valid` applies only to the exact data, split, model, task, checkpoint, seed, and estimator combination that passed review.",
            "- A baseline-valid result does not imply strong accuracy, state-of-the-art performance, or Cartesian-product support for its individual components.",
            "- Pipeline 03 and Pipeline 04 remain experimental rather than maintained execution paths.",
            "- Pipeline 05, Pipeline 06, and Pipeline_ID remain outside this exact table unless a reviewed maintained config is added.",
            "- Historical and paper-only configs are not promoted by this generator.",
            "- External dataset redistribution and availability remain separate source-license questions.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_combinations(demos: list[VerifiedDemo]) -> str:
    lines = [
        "# Execution-Verified Combinations for the PHMFactory v0.3 Pre-release",
        "",
        "> Generated from `configs/config_registry.csv` rows with "
        "`category in {demo, baseline}` and `status=sanity_ok`, plus their fully "
        "resolved configurations.",
        "",
        "Re-generate:",
        "",
        "```bash",
        "python -m scripts.gen_support_matrix",
        "```",
        "",
        "| Registry id | Kind | Config | Pipeline | Data base | Model | Task | Trainer | Execution evidence | Protocol status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for demo in demos:
        lines.append(
            f"| `{demo.config_id}` | `{demo.category}` | `{demo.path}` | "
            f"`{demo.pipeline}` | `{demo.data_base}` | `{demo.model}` | "
            f"`{demo.task}` | `{demo.trainer}` | `{demo.execution_status}` | "
            f"`{demo.protocol_status}` |"
        )
    lines.extend(
        [
            "",
            "Evidence scope is configuration-specific. Smoke rows establish bounded "
            "execution only. A `baseline_valid` row additionally establishes the "
            "declared data population, disjoint split, objective, checkpoint-selection, "
            "repeated-seed, and estimator contract for that exact configuration.",
            "",
            "## Interpretation",
            "",
            "`execution_status=sanity_ok` says that the exact command has current "
            "execution evidence. `protocol_status=smoke_only` says that its scientific "
            "protocol has not been promoted. `protocol_status=baseline_valid` says that "
            "the exact complete experiment passed its declared scientific gates; it does "
            "not say that the model is accurate, state of the art, or transferable to "
            "other component combinations.",
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

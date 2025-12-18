from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RegistryRow:
    config_id: str
    category: str
    path: str
    description: str
    pipeline: str
    base_environment: str
    base_data: str
    base_model: str
    base_task: str
    base_trainer: str
    status: str
    owner_code: str
    keyspace: str
    minimal_run: str
    common_overrides: str
    outputs: str
    related_docs: str


def read_registry(registry_path: Path) -> List[RegistryRow]:
    with registry_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "id",
            "category",
            "path",
            "description",
            "pipeline",
            "base_environment",
            "base_data",
            "base_model",
            "base_task",
            "base_trainer",
            "status",
            "owner_code",
            "keyspace",
            "minimal_run",
            "common_overrides",
            "outputs",
            "related_docs",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"configs/config_registry.csv missing columns: {sorted(missing)}; "
                "see docs/config_registry_schema.md"
            )

        def _cell(name: str) -> str:
            val = raw.get(name)
            return (val or "").strip()

        rows: List[RegistryRow] = []
        for raw in reader:
            rows.append(
                RegistryRow(
                    config_id=_cell("id"),
                    category=_cell("category"),
                    path=_cell("path"),
                    description=_cell("description"),
                    pipeline=_cell("pipeline"),
                    base_environment=_cell("base_environment"),
                    base_data=_cell("base_data"),
                    base_model=_cell("base_model"),
                    base_task=_cell("base_task"),
                    base_trainer=_cell("base_trainer"),
                    status=_cell("status"),
                    owner_code=_cell("owner_code"),
                    keyspace=_cell("keyspace"),
                    minimal_run=_cell("minimal_run"),
                    common_overrides=_cell("common_overrides"),
                    outputs=_cell("outputs"),
                    related_docs=_cell("related_docs"),
                )
            )
        return rows


def _fmt_list(cell: str) -> str:
    items = [x.strip() for x in cell.split(";") if x.strip()]
    if not items:
        return "-"
    return ", ".join(f"`{x}`" for x in items)


def _fmt_code(cell: str) -> str:
    return f"`{cell}`" if cell else "-"


def group_rows(rows: Iterable[RegistryRow]) -> DefaultDict[str, DefaultDict[str, List[RegistryRow]]]:
    grouped: DefaultDict[str, DefaultDict[str, List[RegistryRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pipeline = row.pipeline or "BASE"
        grouped[pipeline][row.category].append(row)
    for pipeline in grouped:
        for cat in grouped[pipeline]:
            grouped[pipeline][cat].sort(key=lambda r: r.config_id)
    return grouped


def render_atlas(rows: List[RegistryRow], registry_path: Path) -> str:
    grouped = group_rows(rows)

    lines: List[str] = []
    lines.append("# CONFIG_ATLAS")
    lines.append("")
    lines.append("> This file is generated from `configs/config_registry.csv`.")
    lines.append("")
    lines.append("Re-generate:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python -m scripts.gen_config_atlas --registry {registry_path.as_posix()}")
    lines.append("```")
    lines.append("")
    lines.append("## Index")
    for pipeline in sorted(grouped.keys()):
        lines.append(f"- [{pipeline}](#{pipeline.lower().replace('_', '-')})")
    lines.append("")

    for pipeline in sorted(grouped.keys()):
        lines.append(f"## {pipeline}")
        lines.append("")
        for category in sorted(grouped[pipeline].keys()):
            lines.append(f"### {category}")
            lines.append("")
            for row in grouped[pipeline][category]:
                lines.append(f"#### `{row.config_id}`")
                lines.append(f"- Path: `{row.path}`")
                lines.append(f"- Description: {row.description}")
                if row.category == "demo":
                    lines.append("- Base configs:")
                    lines.append(f"  - environment: `{row.base_environment}`")
                    lines.append(f"  - data: `{row.base_data}`")
                    lines.append(f"  - model: `{row.base_model}`")
                    lines.append(f"  - task: `{row.base_task}`")
                    lines.append(f"  - trainer: `{row.base_trainer}`")
                lines.append(f"- Owner code: {_fmt_code(row.owner_code)}")
                lines.append(f"- Keyspace: {_fmt_list(row.keyspace)}")
                lines.append(f"- Minimal run: {_fmt_code(row.minimal_run)}")
                lines.append(f"- Common overrides: {_fmt_list(row.common_overrides)}")
                lines.append(f"- Outputs: {_fmt_code(row.outputs)}")
                lines.append(f"- Related docs: {_fmt_list(row.related_docs)}")
                lines.append(f"- Status: `{row.status or '/'}`")
                lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate docs/CONFIG_ATLAS.md from config_registry.csv")
    parser.add_argument(
        "--registry",
        type=str,
        default="configs/config_registry.csv",
        help="Path to configs/config_registry.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/CONFIG_ATLAS.md",
        help="Output markdown path",
    )
    args = parser.parse_args(argv)

    registry_path = Path(args.registry)
    out_path = Path(args.out)

    rows = read_registry(registry_path)
    content = render_atlas(rows, registry_path=registry_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

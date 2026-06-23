from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.configs.config_utils import load_config


SUPPORT_STATUSES = {"smoke-tested", "dependency-blocked", "unverified", "unsupported", "failed"}
MODEL_REGISTRY_PATH = Path("src/model_factory/model_registry.csv")
ISFM_COMPONENT_REGISTRY_PATH = Path("src/model_factory/ISFM/isfm_components.csv")

OPTIONAL_DEPENDENCIES = {
    ("X_model", "CI_GNN"): ("torch_geometric", "requires torch_geometric for GNN layers"),
}

SMOKE_TESTED_MODELS = {
    ("ISFM", "M_01_ISFM"): "configs/hydra/experiments/00_smoke/dummy_dg.yaml",
}

SMOKE_TESTED_COMPONENTS = {
    ("embedding", "E_01_HSE"),
    ("backbone", "B_04_Dlinear"),
    ("task_head", "H_01_Linear_cla"),
}


@dataclass(frozen=True)
class ModelEntry:
    model_type: str
    model_name: str
    module_path: str
    args: str
    notes: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.model_type, self.model_name)


@dataclass(frozen=True)
class ComponentEntry:
    component_type: str
    component_id: str
    module_path: str
    key_args: str
    notes: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.component_type, self.component_id)


@dataclass(frozen=True)
class SupportRecord:
    key: Tuple[str, str]
    status: str
    reason: str
    evidence: str = ""


@dataclass(frozen=True)
class ModelSupportReport:
    model_statuses: Mapping[Tuple[str, str], SupportRecord]
    component_statuses: Mapping[Tuple[str, str], SupportRecord]
    duplicate_model_keys: Tuple[Tuple[str, str], ...]
    duplicate_component_keys: Tuple[Tuple[str, str], ...]


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _source_path_exists(module_path: str) -> bool:
    return Path(module_path).exists()


def _duplicate_keys(keys: Sequence[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    seen = set()
    duplicates = []
    for key in keys:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return tuple(duplicates)


def load_model_entries(path: Path = MODEL_REGISTRY_PATH) -> Tuple[ModelEntry, ...]:
    entries: List[ModelEntry] = []
    for row in _read_csv(path):
        model_type = (row.get("model.type") or "").strip()
        model_name = (row.get("model.name") or "").strip()
        if not model_type or not model_name:
            continue
        entries.append(
            ModelEntry(
                model_type=model_type,
                model_name=model_name,
                module_path=(row.get("module_path") or "").strip(),
                args=(row.get("args") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
        )
    return tuple(entries)


def load_component_entries(path: Path = ISFM_COMPONENT_REGISTRY_PATH) -> Tuple[ComponentEntry, ...]:
    entries: List[ComponentEntry] = []
    for row in _read_csv(path):
        component_type = (row.get("component_type") or "").strip()
        component_id = (row.get("component_id") or "").strip()
        if not component_type or not component_id:
            continue
        entries.append(
            ComponentEntry(
                component_type=component_type,
                component_id=component_id,
                module_path=(row.get("module_path") or "").strip(),
                key_args=(row.get("key_args") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
        )
    return tuple(entries)


def _dependency_blocker(key: Tuple[str, str]) -> Optional[str]:
    dep = OPTIONAL_DEPENDENCIES.get(key)
    if dep is None:
        return None
    package, reason = dep
    if importlib.util.find_spec(package) is None:
        return f"{package}: {reason}"
    return None


def _model_status(entry: ModelEntry) -> SupportRecord:
    blocker = _dependency_blocker(entry.key)
    if blocker is not None:
        return SupportRecord(entry.key, "dependency-blocked", blocker)

    if not _source_path_exists(entry.module_path):
        return SupportRecord(entry.key, "failed", f"missing module path: {entry.module_path}")

    if entry.key in SMOKE_TESTED_MODELS:
        config_path = SMOKE_TESTED_MODELS[entry.key]
        return SupportRecord(entry.key, "smoke-tested", f"smoke config: {config_path}", config_path)

    if entry.model_type == "X_model":
        return SupportRecord(
            entry.key,
            "smoke-tested",
            "covered by test/test_x_model_smoke.py",
            "python -m pytest -q test/test_x_model_smoke.py",
        )

    return SupportRecord(entry.key, "unverified", "registry-backed, but no smoke evidence recorded")


def _component_status(entry: ComponentEntry) -> SupportRecord:
    if not _source_path_exists(entry.module_path):
        return SupportRecord(entry.key, "failed", f"missing module path: {entry.module_path}")
    if entry.key in SMOKE_TESTED_COMPONENTS:
        return SupportRecord(entry.key, "smoke-tested", "referenced by maintained ISFM smoke config")
    if entry.key_args in {"", "/"}:
        return SupportRecord(entry.key, "unverified", "component lacks key_args documentation")
    return SupportRecord(entry.key, "unverified", "registry-backed, but no smoke evidence recorded")


def maintained_isfm_component_keys(config_path: str) -> Tuple[Tuple[str, str], ...]:
    cfg = load_config(config_path)
    return (
        ("embedding", str(cfg.model.embedding)),
        ("backbone", str(cfg.model.backbone)),
        ("task_head", str(cfg.model.task_head)),
    )


def derive_model_support(
    *,
    model_registry_path: Path = MODEL_REGISTRY_PATH,
    component_registry_path: Path = ISFM_COMPONENT_REGISTRY_PATH,
) -> ModelSupportReport:
    models = load_model_entries(model_registry_path)
    components = load_component_entries(component_registry_path)
    return ModelSupportReport(
        model_statuses={entry.key: _model_status(entry) for entry in models},
        component_statuses={entry.key: _component_status(entry) for entry in components},
        duplicate_model_keys=_duplicate_keys([entry.key for entry in models]),
        duplicate_component_keys=_duplicate_keys([entry.key for entry in components]),
    )


def render_markdown(report: ModelSupportReport) -> str:
    lines = [
        "# Model, Loss, And Baseline Registry",
        "",
        "## Model Support",
        "",
        "| Model type | Model name | Status | Reason |",
        "|---|---|---|---|",
    ]
    for key in sorted(report.model_statuses):
        item = report.model_statuses[key]
        lines.append(f"| `{key[0]}` | `{key[1]}` | `{item.status}` | {item.reason} |")

    lines.extend(["", "## ISFM Component Support", "", "| Type | ID | Status | Reason |", "|---|---|---|---|"])
    for key in sorted(report.component_statuses):
        item = report.component_statuses[key]
        lines.append(f"| `{key[0]}` | `{key[1]}` | `{item.status}` | {item.reason} |")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive model/component support from registries")
    parser.add_argument("--model-registry", default=str(MODEL_REGISTRY_PATH))
    parser.add_argument("--component-registry", default=str(ISFM_COMPONENT_REGISTRY_PATH))
    args = parser.parse_args(argv)

    report = derive_model_support(
        model_registry_path=Path(args.model_registry),
        component_registry_path=Path(args.component_registry),
    )
    print(render_markdown(report))
    failed = [
        item
        for item in list(report.model_statuses.values()) + list(report.component_statuses.values())
        if item.status == "failed"
    ]
    if report.duplicate_model_keys or report.duplicate_component_keys or failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

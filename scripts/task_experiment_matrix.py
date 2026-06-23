from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.configs.config_utils import load_config


SUPPORT_STATUSES = {"smoke-tested", "real-data-ready", "unverified", "unsupported"}
TASK_REGISTRY_PATH = Path("src/task_factory/task_registry.csv")
CONFIG_REGISTRY_PATH = Path("configs/config_registry.csv")

SMOKE_CONFIGS = {
    "configs/hydra/experiments/00_smoke/dummy_dg.yaml",
}

FULL_CONFIGS = {
    "configs/hydra/experiments/01_cross_domain/cwru_dg.yaml",
    "configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml",
    "configs/hydra/experiments/03_fewshot/cwru_protonet.yaml",
    "configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml",
    "configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml",
    "configs/hydra/experiments/06_pretrain_cddg/pretrain_hse_cddg.yaml",
}

FOCUSED_TEST_FAMILIES = {
    ("pretrain", "hse_contrastive"): (
        "python -m pytest -q "
        "test/test_hse_contrastive_failfast.py::test_hse_contrastive_flow_has_nonzero_signal"
    ),
}

EXPECTED_ABSENT_CAPABILITIES = {
    "regression": "No registry-backed regression task family is currently exposed.",
    "multi-task": "Multi-task code exists experimentally but has no task-registry row.",
}


@dataclass(frozen=True)
class TaskFamily:
    task_type: str
    task_name: str
    path: str
    dataset_path: str
    batch_format: str
    notes: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.task_type, self.task_name)


@dataclass(frozen=True)
class ConfigEntry:
    entry_id: str
    path: str
    status: str
    minimal_run: str
    common_overrides: str


@dataclass(frozen=True)
class FamilyStatus:
    family: TaskFamily
    status: str
    reason: str
    config_paths: Tuple[str, ...]
    command: str = ""


@dataclass(frozen=True)
class MissingConfigTaskRef:
    entry_id: str
    path: str
    task_type: str
    task_name: str


@dataclass(frozen=True)
class MatrixReport:
    family_statuses: Mapping[Tuple[str, str], FamilyStatus]
    missing_config_task_refs: Tuple[MissingConfigTaskRef, ...]
    duplicate_task_keys: Tuple[Tuple[str, str], ...]
    absent_capabilities: Mapping[str, str]


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_task_families(path: Path = TASK_REGISTRY_PATH) -> Tuple[TaskFamily, ...]:
    families: List[TaskFamily] = []
    for row in _read_csv(path):
        task_type = (row.get("task.type") or "").strip()
        task_name = (row.get("task.name") or "").strip()
        if not task_type or not task_name:
            continue
        families.append(
            TaskFamily(
                task_type=task_type,
                task_name=task_name,
                path=(row.get("path") or "").strip(),
                dataset_path=(row.get("dataset_path") or "").strip(),
                batch_format=(row.get("batch_format") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
        )
    return tuple(families)


def load_active_config_entries(path: Path = CONFIG_REGISTRY_PATH) -> Tuple[ConfigEntry, ...]:
    entries: List[ConfigEntry] = []
    for row in _read_csv(path):
        status = (row.get("status") or "").strip()
        config_path = (row.get("path") or "").strip()
        if not config_path or status == "/":
            continue
        entries.append(
            ConfigEntry(
                entry_id=(row.get("id") or config_path).strip(),
                path=config_path,
                status=status,
                minimal_run=(row.get("minimal_run") or "").strip(),
                common_overrides=(row.get("common_overrides") or "").strip(),
            )
        )
    return tuple(entries)


def resolve_config_task(config_path: str) -> Tuple[str, str]:
    cfg = load_config(config_path)
    return (str(cfg.task.type), str(cfg.task.name))


def _duplicate_keys(families: Sequence[TaskFamily]) -> Tuple[Tuple[str, str], ...]:
    seen = set()
    duplicates = []
    for family in families:
        if family.key in seen and family.key not in duplicates:
            duplicates.append(family.key)
        seen.add(family.key)
    return tuple(duplicates)


def derive_matrix(
    *,
    task_registry_path: Path = TASK_REGISTRY_PATH,
    config_registry_path: Path = CONFIG_REGISTRY_PATH,
) -> MatrixReport:
    families = load_task_families(task_registry_path)
    family_by_key = {family.key: family for family in families}
    configs_by_key: Dict[Tuple[str, str], List[str]] = {family.key: [] for family in families}
    missing_refs: List[MissingConfigTaskRef] = []

    for entry in load_active_config_entries(config_registry_path):
        if not Path(entry.path).exists():
            continue
        task_type, task_name = resolve_config_task(entry.path)
        key = (task_type, task_name)
        if key not in family_by_key:
            missing_refs.append(
                MissingConfigTaskRef(
                    entry_id=entry.entry_id,
                    path=entry.path,
                    task_type=task_type,
                    task_name=task_name,
                )
            )
            continue
        configs_by_key.setdefault(key, []).append(entry.path)

    statuses: Dict[Tuple[str, str], FamilyStatus] = {}
    for family in families:
        config_paths = tuple(sorted(configs_by_key.get(family.key, [])))
        smoke_paths = tuple(path for path in config_paths if path in SMOKE_CONFIGS)
        full_paths = tuple(path for path in config_paths if path in FULL_CONFIGS)
        focused_command = FOCUSED_TEST_FAMILIES.get(family.key, "")

        if smoke_paths:
            status = "smoke-tested"
            reason = f"offline smoke config: {smoke_paths[0]}"
            command = f"python main.py --config {smoke_paths[0]}"
        elif focused_command:
            status = "smoke-tested"
            reason = "focused offline test covers this family"
            command = focused_command
        elif full_paths:
            status = "real-data-ready"
            reason = f"full matrix config requires PHM_VIBENCH_DATA: {full_paths[0]}"
            command = f"PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full"
        elif config_paths:
            status = "unverified"
            reason = "config-backed, but no smoke/full evidence is recorded"
            command = ""
        else:
            status = "unverified"
            reason = "registry-backed, but no maintained config is recorded"
            command = ""

        statuses[family.key] = FamilyStatus(
            family=family,
            status=status,
            reason=reason,
            config_paths=config_paths,
            command=command,
        )

    return MatrixReport(
        family_statuses=statuses,
        missing_config_task_refs=tuple(missing_refs),
        duplicate_task_keys=_duplicate_keys(families),
        absent_capabilities=EXPECTED_ABSENT_CAPABILITIES,
    )


def render_markdown(report: MatrixReport) -> str:
    lines = [
        "# PHM Task Experiment Matrix",
        "",
        "| Task type | Task name | Status | Evidence / reason |",
        "|---|---|---|---|",
    ]
    for key in sorted(report.family_statuses):
        item = report.family_statuses[key]
        command = f" Command: `{item.command}`." if item.command else ""
        lines.append(
            "| "
            f"`{item.family.task_type}` | `{item.family.task_name}` | `{item.status}` | "
            f"{item.reason}{command} |"
        )

    lines.extend(["", "## Absent Capabilities", ""])
    for name, reason in sorted(report.absent_capabilities.items()):
        lines.append(f"- `{name}`: unsupported. {reason}")

    if report.missing_config_task_refs:
        lines.extend(["", "## Config-To-Task Gaps", ""])
        for ref in report.missing_config_task_refs:
            lines.append(f"- `{ref.entry_id}` -> `{ref.task_type}.{ref.task_name}` from `{ref.path}`")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive PHM task experiment matrix from registries")
    parser.add_argument("--task-registry", default=str(TASK_REGISTRY_PATH))
    parser.add_argument("--config-registry", default=str(CONFIG_REGISTRY_PATH))
    args = parser.parse_args(argv)

    report = derive_matrix(
        task_registry_path=Path(args.task_registry),
        config_registry_path=Path(args.config_registry),
    )
    print(render_markdown(report))
    if report.duplicate_task_keys or report.missing_config_task_refs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

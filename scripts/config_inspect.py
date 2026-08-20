"""Inspect a PHMFactory configuration without defining a second resolver.

The script is a presentation adapter over :func:`phmfactory.config.analyze_config`.
Composition, explicit local configuration, CLI overrides, schema validation, and
canonical Pipeline naming therefore match the real runtime and preflight.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml

from phmfactory.config import ConfigAnalysis, analyze_config
from phmfactory.pipelines import pipeline_module_name


DumpMode = Literal["resolved", "sources", "targets", "all"]
OutFormat = Literal["yaml", "json", "md"]


@dataclass(frozen=True)
class InspectResult:
    """Resolved config plus human-facing discovery and sanity information."""

    resolved: Dict[str, Any]
    sources: Dict[str, str]
    targets: Dict[str, Any]
    sanity: List[Dict[str, Any]]
    local_config_path: str | None


def _load_model_registry() -> Dict[Tuple[str, str], str]:
    path = Path("src/model_factory/model_registry.csv")
    mapping: Dict[Tuple[str, str], str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model_type = (row.get("model.type") or "").strip()
            model_name = (row.get("model.name") or "").strip()
            module_path = (row.get("module_path") or "").strip()
            if model_type and model_name and module_path:
                mapping[(model_type, model_name)] = module_path
    return mapping


def _load_isfm_components() -> Dict[Tuple[str, str], str]:
    path = Path("src/model_factory/ISFM/isfm_components.csv")
    mapping: Dict[Tuple[str, str], str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            component_type = (row.get("component_type") or "").strip()
            component_id = (row.get("component_id") or "").strip()
            module_path = (row.get("module_path") or "").strip()
            if component_type and component_id and module_path:
                mapping[(component_type, component_id)] = module_path
    return mapping


def _load_task_registry() -> Dict[Tuple[str, str], Dict[str, str]]:
    path = Path("src/task_factory/task_registry.csv")
    mapping: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_type = (row.get("task.type") or "").strip()
            task_name = (row.get("task.name") or "").strip()
            if task_type and task_name:
                mapping[(task_type, task_name)] = {
                    "task_path": (row.get("path") or "").strip(),
                    "dataset_path": (row.get("dataset_path") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
    return mapping


def _module_discoverable(module_name: str) -> Tuple[bool, str]:
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"
    return found, "" if found else f"module not found: {module_name}"


def _instantiation_targets(analysis: ConfigAnalysis) -> Dict[str, Any]:
    resolved = analysis.effective_config
    pipeline_module = pipeline_module_name(analysis.pipeline, warn=False)
    pipeline_found, pipeline_error = _module_discoverable(pipeline_module)
    targets: Dict[str, Any] = {
        "pipeline": {
            "name": analysis.pipeline,
            "module": pipeline_module,
            "symbol": "pipeline",
            "discoverable": pipeline_found,
            "error": pipeline_error,
        },
        "factories": {
            "data_factory": "src.data_factory:build_data",
            "model_factory": "src.model_factory:build_model",
            "task_factory": "src.task_factory:build_task",
            "trainer_factory": "src.trainer_factory:build_trainer",
        },
    }

    model = resolved.get("model") if isinstance(resolved.get("model"), dict) else {}
    model_type = str(model.get("type") or "")
    model_name = str(model.get("name") or "")
    model_path = _load_model_registry().get((model_type, model_name), "")
    targets["model"] = {
        "type": model_type,
        "name": model_name,
        "module_path": model_path,
        "registered": bool(model_path),
    }
    if model_type == "ISFM":
        components = _load_isfm_components()
        embedding = str(model.get("embedding") or "")
        backbone = str(model.get("backbone") or "")
        task_head = str(model.get("task_head") or "")
        targets["model"]["components"] = {
            "embedding": {
                "id": embedding,
                "module_path": components.get(("embedding", embedding), ""),
            },
            "backbone": {
                "id": backbone,
                "module_path": components.get(("backbone", backbone), ""),
            },
            "task_head": {
                "id": task_head,
                "module_path": components.get(("task_head", task_head), ""),
            },
        }

    task = resolved.get("task") if isinstance(resolved.get("task"), dict) else {}
    task_type = str(task.get("type") or "")
    task_name = str(task.get("name") or "")
    task_info = _load_task_registry().get((task_type, task_name), {})
    targets["task"] = {
        "type": task_type,
        "name": task_name,
        "task_path": task_info.get("task_path", ""),
        "dataset_path": task_info.get("dataset_path", ""),
        "notes": task_info.get("notes", ""),
        "registered": bool(task_info),
    }

    trainer = resolved.get("trainer") if isinstance(resolved.get("trainer"), dict) else {}
    trainer_name = str(
        trainer.get("trainer_name") or trainer.get("name") or "Default_trainer"
    )
    trainer_module = f"src.trainer_factory.{trainer_name}"
    trainer_found, trainer_error = _module_discoverable(trainer_module)
    targets["trainer"] = {
        "name": trainer_name,
        "module": trainer_module,
        "discoverable": trainer_found,
        "error": trainer_error,
    }
    return targets


def _sanity_checks(
    analysis: ConfigAnalysis,
    targets: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, message: str, fix: str = "") -> None:
        checks.append({"check": name, "ok": ok, "message": message, "fix": fix})

    resolved = analysis.effective_config
    pipeline = targets["pipeline"]
    add(
        "pipeline_discoverable",
        bool(pipeline["discoverable"]),
        f"pipeline={analysis.pipeline}, module={pipeline['module']}",
        "Correct the top-level pipeline value or install its required package.",
    )

    environment = resolved["environment"]
    add(
        "seed_type",
        isinstance(environment.get("seed"), int)
        and not isinstance(environment.get("seed"), bool),
        f"environment.seed={environment.get('seed')!r}",
        "Set environment.seed to an integer.",
    )
    add(
        "output_dir_set",
        isinstance(environment.get("output_dir"), str)
        and bool(environment.get("output_dir", "").strip()),
        f"environment.output_dir={environment.get('output_dir')!r}",
        "Set environment.output_dir to a writable path.",
    )

    trainer = resolved["trainer"]
    epochs = trainer.get("num_epochs")
    add(
        "num_epochs_positive",
        isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0,
        f"trainer.num_epochs={epochs!r}",
        "Set trainer.num_epochs to a positive integer.",
    )
    return checks


def inspect_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    local_config: Optional[str] = None,
) -> InspectResult:
    """Inspect the exact config used by validate, preflight, and run."""

    analysis = analyze_config(
        config_path,
        override_values=overrides,
        local_config=local_config,
    )
    targets = _instantiation_targets(analysis)
    return InspectResult(
        resolved=analysis.runtime_config(),
        sources=dict(analysis.sources),
        targets=targets,
        sanity=_sanity_checks(analysis, targets),
        local_config_path=(
            str(analysis.local_config_path)
            if analysis.local_config_path is not None
            else None
        ),
    )


def _has_failed_sanity(result: InspectResult) -> bool:
    return any(not bool(item.get("ok")) for item in result.sanity)


def _payload(result: InspectResult, dump: DumpMode) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"local_config_path": result.local_config_path}
    if dump in ("resolved", "all"):
        payload["resolved"] = result.resolved
    if dump in ("sources", "all"):
        payload["sources"] = result.sources
    if dump in ("targets", "all"):
        payload["targets"] = result.targets
    if dump == "all":
        payload["sanity"] = result.sanity
    return payload


def _render_md(result: InspectResult, dump: DumpMode) -> str:
    parts = [f"explicit_local_config: `{result.local_config_path or 'none'}`", ""]
    if dump in ("resolved", "all"):
        parts.extend(
            [
                "## RESOLVED CONFIG",
                "```yaml",
                yaml.safe_dump(result.resolved, allow_unicode=True, sort_keys=False),
                "```",
                "",
            ]
        )
    if dump in ("sources", "all"):
        parts.extend(["## FIELD SOURCES", "", "| Field | Source |", "|---|---|"])
        for key in sorted(result.sources):
            parts.append(f"| `{key}` | {result.sources[key]} |")
        parts.append("")
    if dump in ("targets", "all"):
        parts.extend(
            [
                "## INSTANTIATION TARGETS",
                "```yaml",
                yaml.safe_dump(result.targets, allow_unicode=True, sort_keys=False),
                "```",
                "",
            ]
        )
    if dump == "all":
        parts.extend(
            [
                "## SANITY CHECK",
                "",
                "| Check | OK | Message | Fix |",
                "|---|---:|---|---|",
            ]
        )
        for item in result.sanity:
            status = "PASS" if item["ok"] else "FAIL"
            parts.append(
                f"| `{item['check']}` | {status} | {item['message']} | "
                f"{item.get('fix', '')} |"
            )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the effective config, sources, targets, and sanity checks"
    )
    parser.add_argument("--config", required=True, help="Preset name or YAML config path")
    parser.add_argument(
        "--override", action="append", default=None, help="Override key=value (repeatable)"
    )
    parser.add_argument(
        "--local-config",
        "--local_config",
        dest="local_config",
        default=None,
        help="Optional explicit machine-local YAML; no file is auto-discovered.",
    )
    parser.add_argument(
        "--dump", choices=["resolved", "sources", "targets", "all"], default="all"
    )
    parser.add_argument("--format", choices=["yaml", "json", "md"], default="md")
    args = parser.parse_args(argv)

    result = inspect_config(
        args.config,
        overrides=args.override,
        local_config=args.local_config,
    )
    payload = _payload(result, args.dump)
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.format == "yaml":
        print(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    else:
        print(_render_md(result, dump=args.dump))
    return 1 if _has_failed_sanity(result) else 0


if __name__ == "__main__":
    raise SystemExit(main())

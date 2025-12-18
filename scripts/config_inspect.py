from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import yaml

from src.configs.config_utils import merge_with_local_override
from src.utils.config_utils import apply_overrides_to_config, parse_overrides


DumpMode = Literal["resolved", "sources", "targets", "all"]
OutFormat = Literal["yaml", "json", "md"]


def _namespace_to_dict(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(v) for v in value]
    return value


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _flatten(d: Dict[str, Any], prefix: str) -> Iterable[Tuple[str, Any]]:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten(v, key)
        else:
            yield key, v


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return raw


@dataclass(frozen=True)
class InspectResult:
    resolved: Dict[str, Any]
    sources: Dict[str, str]
    targets: Dict[str, Any]
    sanity: List[Dict[str, Any]]


def _find_local_override_path(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    default_local = Path("configs/local/local.yaml")
    return default_local if default_local.exists() else None


def _collect_sources(
    config_path: Path,
    local_override: Optional[Path],
    cli_overrides: Dict[str, Any],
) -> Dict[str, str]:
    cfg = _load_yaml_dict(config_path)
    base_cfgs = cfg.get("base_configs") if isinstance(cfg.get("base_configs"), dict) else {}

    sources: Dict[str, str] = {}
    resolved_blocks: Dict[str, Dict[str, Any]] = {k: {} for k in ["environment", "data", "model", "task", "trainer"]}

    for block in resolved_blocks.keys():
        base_path = base_cfgs.get(block)
        if isinstance(base_path, str) and base_path:
            base_yaml = _load_yaml_dict(Path(base_path))
            base_block = base_yaml.get(block, {})
            if isinstance(base_block, dict):
                resolved_blocks[block] = _deep_merge(resolved_blocks[block], base_block)
                for key, _ in _flatten(base_block, prefix=block):
                    sources[key] = f"base:{base_path}"

    for block in resolved_blocks.keys():
        override_block = cfg.get(block, {})
        if isinstance(override_block, dict) and override_block:
            _deep_merge(resolved_blocks[block], override_block)
            for key, _ in _flatten(override_block, prefix=block):
                sources[key] = f"config:{config_path.as_posix()}"

    if local_override is not None:
        local_yaml = _load_yaml_dict(local_override)
        for block in resolved_blocks.keys():
            local_block = local_yaml.get(block, {})
            if isinstance(local_block, dict) and local_block:
                _deep_merge(resolved_blocks[block], local_block)
                for key, _ in _flatten(local_block, prefix=block):
                    sources[key] = f"local:{local_override.as_posix()}"

    for key, value in _flatten(cli_overrides, prefix=""):
        sources[key] = "cli:--override"

    if isinstance(cfg.get("pipeline"), str):
        sources["pipeline"] = f"config:{config_path.as_posix()}"
    if base_cfgs:
        for block, base_path in base_cfgs.items():
            if isinstance(base_path, str) and base_path:
                sources[f"base_configs.{block}"] = f"config:{config_path.as_posix()}"

    return sources


def _load_model_registry() -> Dict[Tuple[str, str], str]:
    registry_path = Path("src/model_factory/model_registry.csv")
    mapping: Dict[Tuple[str, str], str] = {}
    if not registry_path.exists():
        return mapping
    with registry_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_type = (row.get("model.type") or "").strip()
            model_name = (row.get("model.name") or "").strip()
            module_path = (row.get("module_path") or "").strip()
            if model_type and model_name and module_path:
                mapping[(model_type, model_name)] = module_path
    return mapping


def _load_isfm_components() -> Dict[Tuple[str, str], str]:
    registry_path = Path("src/model_factory/ISFM/isfm_components.csv")
    mapping: Dict[Tuple[str, str], str] = {}
    if not registry_path.exists():
        return mapping
    with registry_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_type = (row.get("component_type") or "").strip()
            comp_id = (row.get("component_id") or "").strip()
            module_path = (row.get("module_path") or "").strip()
            if comp_type and comp_id and module_path:
                mapping[(comp_type, comp_id)] = module_path
    return mapping


def _load_task_registry() -> Dict[Tuple[str, str], Dict[str, str]]:
    registry_path = Path("src/task_factory/task_registry.csv")
    mapping: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not registry_path.exists():
        return mapping
    with registry_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_type = (row.get("task.type") or "").strip()
            task_name = (row.get("task.name") or "").strip()
            if not task_type or not task_name:
                continue
            mapping[(task_type, task_name)] = {
                "task_path": (row.get("path") or "").strip(),
                "dataset_path": (row.get("dataset_path") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
    return mapping


def _maybe_import(module_name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as e:
        return False, repr(e)


def _instantiation_targets(resolved: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = (resolved.get("pipeline") or "Pipeline_01_default").strip()

    targets: Dict[str, Any] = {
        "pipeline": {"name": pipeline, "module": f"src.{pipeline}", "symbol": "pipeline"},
        "factories": {
            "data_factory": "src/data_factory/__init__.py:build_data",
            "model_factory": "src/model_factory/__init__.py:build_model",
            "task_factory": "src/task_factory/__init__.py:build_task",
            "trainer_factory": "src/trainer_factory/__init__.py:build_trainer",
        },
    }

    model_cfg = resolved.get("model") if isinstance(resolved.get("model"), dict) else {}
    model_type = str(model_cfg.get("type") or "")
    model_name = str(model_cfg.get("name") or "")
    model_registry = _load_model_registry()
    model_module_path = model_registry.get((model_type, model_name))
    targets["model"] = {
        "type": model_type,
        "name": model_name,
        "module_path": model_module_path or "",
    }

    if model_type == "ISFM":
        comps = _load_isfm_components()
        emb = str(model_cfg.get("embedding") or "")
        backbone = str(model_cfg.get("backbone") or "")
        head = str(model_cfg.get("task_head") or "")
        targets["model"]["components"] = {
            "embedding": {"id": emb, "module_path": comps.get(("embedding", emb), "")},
            "backbone": {"id": backbone, "module_path": comps.get(("backbone", backbone), "")},
            "task_head": {"id": head, "module_path": comps.get(("task_head", head), "")},
        }

    task_cfg = resolved.get("task") if isinstance(resolved.get("task"), dict) else {}
    task_type = str(task_cfg.get("type") or "")
    task_name = str(task_cfg.get("name") or "")
    task_registry = _load_task_registry()
    task_info = task_registry.get((task_type, task_name), {})
    targets["task"] = {
        "type": task_type,
        "name": task_name,
        "task_path": task_info.get("task_path", ""),
        "dataset_path": task_info.get("dataset_path", ""),
        "notes": task_info.get("notes", ""),
    }

    trainer_cfg = resolved.get("trainer") if isinstance(resolved.get("trainer"), dict) else {}
    trainer_name = str(trainer_cfg.get("name") or "")
    trainer_module = f"src.trainer_factory.{trainer_name}"
    ok, err = _maybe_import(trainer_module)
    if not ok:
        trainer_module = "src.trainer_factory.Default_trainer"
    targets["trainer"] = {"name": trainer_name, "module": trainer_module, "import_ok": ok, "import_error": err}

    return targets


def _sanity_checks(resolved: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, message: str, fix: str = "") -> None:
        checks.append({"check": name, "ok": ok, "message": message, "fix": fix})

    for block in ["environment", "data", "model", "task", "trainer"]:
        add(
            f"has_{block}",
            isinstance(resolved.get(block), dict),
            f"{block} present: {block in resolved}",
            fix=f"Ensure `{block}: ...` exists after base_configs merge.",
        )

    pipeline = (resolved.get("pipeline") or "Pipeline_01_default").strip()
    ok, err = _maybe_import(f"src.{pipeline}")
    add(
        "pipeline_import",
        ok,
        f"pipeline={pipeline}",
        fix="Set YAML top-level `pipeline:` to an existing src/Pipeline_*.py module.",
    )
    if not ok:
        add("pipeline_import_error", False, err, fix="Fix missing dependency or typo in pipeline name.")

    env = resolved.get("environment") or {}
    seed = env.get("seed")
    add(
        "seed_type",
        isinstance(seed, int),
        f"environment.seed={seed!r}",
        fix="Set `environment.seed` to an integer.",
    )
    output_dir = env.get("output_dir")
    add(
        "output_dir_set",
        isinstance(output_dir, str) and bool(output_dir.strip()),
        f"environment.output_dir={output_dir!r}",
        fix="Set `environment.output_dir` to a relative path like `results/demo/<exp>`.",
    )
    if isinstance(output_dir, str) and output_dir.startswith("/"):
        add(
            "output_dir_not_absolute",
            False,
            f"environment.output_dir is absolute: {output_dir}",
            fix="Prefer a repo-relative output_dir or use configs/local/local.yaml for machine paths.",
        )

    model = resolved.get("model") or {}
    add(
        "model_type_name",
        bool(model.get("type")) and bool(model.get("name")),
        f"model.type={model.get('type')!r}, model.name={model.get('name')!r}",
        fix="Set `model.type` and `model.name`.",
    )

    task = resolved.get("task") or {}
    add(
        "task_type_name",
        bool(task.get("type")) and bool(task.get("name")),
        f"task.type={task.get('type')!r}, task.name={task.get('name')!r}",
        fix="Set `task.type` and `task.name`.",
    )

    trainer = resolved.get("trainer") or {}
    num_epochs = trainer.get("num_epochs")
    add(
        "num_epochs_int",
        isinstance(num_epochs, int) or num_epochs is None,
        f"trainer.num_epochs={num_epochs!r}",
        fix="Set `trainer.num_epochs` to an integer (or override via CLI).",
    )

    return checks


def inspect_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    local_config: Optional[str] = None,
) -> InspectResult:
    config_path_p = Path(config_path)
    if not config_path_p.exists():
        raise FileNotFoundError(f"--config not found: {config_path}")

    local_override_path = _find_local_override_path(local_config)

    cfg = merge_with_local_override(config_path_p, local_override_path)
    if overrides:
        overrides_dict = parse_overrides(overrides)
        cfg = apply_overrides_to_config(cfg, overrides_dict)
    else:
        overrides_dict = {}

    resolved = _namespace_to_dict(cfg)
    sources = _collect_sources(config_path_p, local_override=local_override_path, cli_overrides=overrides_dict)
    targets = _instantiation_targets(resolved)
    sanity = _sanity_checks(resolved)
    return InspectResult(resolved=resolved, sources=sources, targets=targets, sanity=sanity)


def _render_md(result: InspectResult, dump: DumpMode) -> str:
    parts: List[str] = []
    if dump in ("resolved", "all"):
        parts.append("## RESOLVED CONFIG")
        parts.append("```yaml")
        parts.append(yaml.safe_dump(result.resolved, allow_unicode=True, sort_keys=False))
        parts.append("```")
        parts.append("")
    if dump in ("sources", "all"):
        parts.append("## FIELD SOURCES")
        parts.append("")
        parts.append("| Field | Source |")
        parts.append("|---|---|")
        for k in sorted(result.sources.keys()):
            parts.append(f"| `{k}` | {result.sources[k]} |")
        parts.append("")
    if dump in ("targets", "all"):
        parts.append("## INSTANTIATION TARGETS")
        parts.append("```yaml")
        parts.append(yaml.safe_dump(result.targets, allow_unicode=True, sort_keys=False))
        parts.append("```")
        parts.append("")
    if dump in ("all",):
        parts.append("## SANITY CHECK")
        parts.append("")
        parts.append("| Check | OK | Message | Fix |")
        parts.append("|---|---:|---|---|")
        for item in result.sanity:
            ok = "PASS" if item["ok"] else "FAIL"
            fix = item.get("fix", "")
            parts.append(f"| `{item['check']}` | {ok} | {item['message']} | {fix} |")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a config: resolved config + sources + targets + sanity")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--override", action="append", default=None, help="Override key=value (repeatable)")
    parser.add_argument("--local_config", default=None, help="Optional machine-local override YAML")
    parser.add_argument("--dump", choices=["resolved", "sources", "targets", "all"], default="all")
    parser.add_argument("--format", choices=["yaml", "json", "md"], default="md")
    args = parser.parse_args(argv)

    result = inspect_config(args.config, overrides=args.override, local_config=args.local_config)

    if args.format == "json":
        payload: Dict[str, Any] = {}
        if args.dump in ("resolved", "all"):
            payload["resolved"] = result.resolved
        if args.dump in ("sources", "all"):
            payload["sources"] = result.sources
        if args.dump in ("targets", "all"):
            payload["targets"] = result.targets
        if args.dump in ("all",):
            payload["sanity"] = result.sanity
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.format == "yaml":
        payload_y: Dict[str, Any] = {}
        if args.dump in ("resolved", "all"):
            payload_y["resolved"] = result.resolved
        if args.dump in ("sources", "all"):
            payload_y["sources"] = result.sources
        if args.dump in ("targets", "all"):
            payload_y["targets"] = result.targets
        if args.dump in ("all",):
            payload_y["sanity"] = result.sanity
        print(yaml.safe_dump(payload_y, allow_unicode=True, sort_keys=False))
        return 0

    print(_render_md(result, dump=args.dump))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute installed scientific task operators through the public config resolver."""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phmfactory.commands.common import add_config_arguments, requested_config, requested_local_config
from phmfactory.config import analyze_config
from phmfactory.pipelines import require_pipeline_access


def reject_scientific_factory_task(config: Mapping[str, Any]) -> None:
    """A scientific execute module cannot masquerade as a Lightning task."""
    task = config.get("task", {})
    if "module" in task:
        raise ValueError("task.module requires task.execution=research with an explicit task.phase or research <phase>")
    identifiers = [task.get("type"), task.get("name")]
    if any(not isinstance(value, str) or not value.isidentifier() for value in identifiers):
        return
    name = "src.task_factory.task." + ".".join(identifiers)
    # Inspect only the public exports. Importing task_factory here would import
    # Lightning during ordinary preflight and defeat its non-runtime boundary.
    base = Path(__file__).resolve().parents[2] / "src" / "task_factory" / "task"
    target = base.joinpath(*identifiers)
    source = target.with_suffix(".py")
    if not source.is_file():
        source = target / "__init__.py"
    if not source.is_file():
        return
    names = set()
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            names.update(item.asname or item.name for item in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(item.id for item in targets if isinstance(item, ast.Name))
    if {"execute", "PHASES"} <= names:
        raise ValueError(f"{name} requires task.execution=research with an explicit task.phase or research <phase>")


def task_module(config: Mapping[str, Any]):
    task = config.get("task", {})
    if task.get("execution") != "research":
        raise ValueError("Research execution requires task.execution=research in the visible configuration")
    identifiers = [task.get("type"), task.get("name")]
    if any(not isinstance(value, str) or not value.isidentifier() for value in identifiers):
        raise ValueError("task.type and task.name must be explicit Python identifiers")
    # The downstream configuration selects its module; no study registry,
    # paper name, import fallback or experiment plan belongs to the framework.
    name = task.get("module", "src.task_factory.task." + ".".join(identifiers))
    if not isinstance(name, str) or not name or any(not part.isidentifier() for part in name.split(".")):
        raise ValueError("task.module must be an explicit dotted Python module name")
    module = importlib.import_module(name)
    if not callable(getattr(module, "execute", None)):
        raise TypeError(f"{module.__name__} does not expose scientific execute(config, phase, output)")
    return module


def preflight(analysis) -> Mapping[str, Any]:
    """Resolve the scientific implementation without executing it or creating output."""
    module = task_module(analysis.effective_config)
    result = {"status": "configuration_resolved", "task_module": module.__name__,
              "resolved_config_path": str(analysis.path),
              "phases": list(getattr(module, "PHASES", ())),
              "data_qualified": False, "execution_verified": False}
    print(f"task_module={module.__name__}")
    print("preflight=configuration_resolved; data qualification and execution not performed")
    return result


def execute_analysis(analysis, *, phase: str | None = None,
                     output: Path | None = None,
                     options: Mapping[str, Any] | None = None,
                     allow_experimental: bool = False) -> Mapping[str, Any]:
    """Execute one already-resolved task; both public commands use this path."""
    config = analysis.runtime_config()
    declared_phase = config.get("task", {}).get("phase")
    if phase is not None and declared_phase is not None and phase != declared_phase:
        raise ValueError(
            f"Requested phase {phase!r} conflicts with task.phase={declared_phase!r}; "
            "select one phase explicitly in the configuration or command"
        )
    phase = declared_phase if phase is None else phase
    if not isinstance(phase, str) or not phase or phase != phase.strip():
        raise ValueError("Research execution requires explicit task.phase or research <phase>")
    require_pipeline_access(analysis.pipeline, allow_experimental=allow_experimental, warn=False)
    module = task_module(config)
    phases = getattr(module, "PHASES", ())
    if phase not in phases:
        raise ValueError(f"Requested phase {phase!r}; configured task supports {phases!r}")
    configured_options = config["task"].get("options", {})
    if not isinstance(configured_options, Mapping):
        raise TypeError("task.options must be a mapping of execute keyword arguments")
    kwargs = dict(configured_options)
    kwargs.update(options or {})
    reserved = {"config", "phase", "output"}.intersection(kwargs)
    if reserved:
        raise ValueError(f"task.options cannot replace reserved arguments: {sorted(reserved)}")
    for key in ("data", "checkpoint", "selection"):
        if key in kwargs:
            value = kwargs[key]
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"task.options.{key} must be an explicit nonempty path")
            kwargs[key] = Path(value)
    if output is None:
        root = config.get("environment", {}).get("output_dir")
        if not isinstance(root, str) or not root.strip():
            raise ValueError("Provide --output or environment.output_dir")
        output = Path(root) / phase
    supplied = dict(config=config, phase=phase, output=output, **kwargs)
    # Reject unsupported options before entering a task or creating run outputs.
    inspect.signature(module.execute).bind(**supplied)
    result = module.execute(**supplied)
    if not isinstance(result, Mapping):
        raise TypeError("Scientific task must return a direct result mapping")
    if str(result.get("status", "")).lower() in {"failed", "invalid", "error"}:
        raise RuntimeError(f"Scientific task reported {result['status']}: {result.get('reason', '')}")
    from phmfactory import installed_build_identity
    result = dict(result)
    result["observed_build"] = installed_build_identity()
    from phmfactory.cli import _print_direct_outputs
    _print_direct_outputs(result)
    print("observed_build=" + json.dumps(result["observed_build"], sort_keys=True))
    print(f"research={result.get('status', 'returned')}")
    return result


def run(argv: Sequence[str]) -> Mapping[str, Any]:
    parser = argparse.ArgumentParser(prog="phmfactory research")
    parser.add_argument("phase", help="preflight or a phase supported by the configured scientific task")
    add_config_arguments(parser, include_experimental=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arms", help="Comma-separated comparison arms")
    parser.add_argument("--device", help="Explicit device; never falls back")
    args = parser.parse_args(list(argv))
    analysis = analyze_config(requested_config(args), override_values=args.override,
                              local_config=requested_local_config(args))
    if args.phase == "preflight":
        return preflight(analysis)
    options = {key: getattr(args, key) for key in
               ("data", "checkpoint", "selection", "target", "seed", "device")
               if getattr(args, key) is not None}
    if args.arms is not None:
        options["arms"] = args.arms.split(",")
    return execute_analysis(analysis, phase=args.phase, output=args.output,
                            options=options, allow_experimental=args.allow_experimental)

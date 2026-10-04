"""Installed acquisition-group generalized few-shot adaptation.

``data`` is a JSON object containing ``episodes``. Each episode explicitly binds
source (a qualified export directory), support/query NPZs, classes and draw IDs.
Relative episode paths are resolved against that JSON. ``checkpoint`` explicitly
replaces the source directory for every episode; it never selects a checkpoint.
No source model, physical map or actual acquisition provenance is inferred.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .runtime import run
from .summary import summarize

PHASES = ("adapt", "evaluate", "summarize")


def execute(config: Mapping[str, Any], phase: str, output: str | Path, *,
            data: str | Path | None = None, checkpoint: str | Path | None = None,
            selection: str | Path | None = None, arms: Sequence[str] | None = None,
            device: str | None = None) -> dict[str, Any]:
    """Run the declared scientific operator using explicit installed resources."""
    if phase not in PHASES:
        raise ValueError(f"Unsupported fault_adaptation phase {phase!r}; expected {PHASES}")
    out = Path(output).expanduser().resolve()
    task = config.get("task", {})
    if (task.get("type"), task.get("name"), task.get("execution")) != ("GFS", "fault_adaptation", "research"):
        raise ValueError("Expected task.type=GFS, name=fault_adaptation, execution=research")
    if data is None:
        data = config.get("data", {}).get("manifest")
    if data is None:
        raise ValueError("Provide --data: an episode JSON, or the completed run directory for summarize")
    bound_data = Path(data).expanduser().resolve()
    if phase == "summarize":
        if any(value is not None for value in (checkpoint, selection, arms, device)):
            raise ValueError("summarize consumes saved native outputs only; checkpoint/selection/arms/device are invalid")
        destination = summarize(bound_data, out)
        return {"status": "completed", "result_dir": str(destination),
                "run_summary": str(destination / "summary.json")}
    options = copy.deepcopy(dict(task.get("fault_adaptation", {})))
    allowed = {"experiment", "information_regime", "adaptation_overrides", "source_revision"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported task.fault_adaptation fields: {sorted(unknown)}")
    experiment = options.pop("experiment", None)
    if experiment is None:
        raise ValueError("Declare task.fault_adaptation.experiment=E1, E2 or E3")
    manifest = json.loads(bound_data.read_text())
    if not isinstance(manifest, dict) or set(manifest) != {"episodes"} or not isinstance(manifest["episodes"], list):
        raise ValueError("Episode manifest must contain only an explicit episodes list")
    episodes = copy.deepcopy(manifest["episodes"])
    source_override = Path(checkpoint).expanduser().resolve() if checkpoint is not None else None
    for episode in episodes:
        for field in ("source", "support", "query"):
            value = str(source_override) if field == "source" and source_override is not None else episode.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Episode must explicitly bind {field}")
            path = Path(value).expanduser()
            path = path.resolve() if path.is_absolute() else (bound_data.parent / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Explicit episode {field} does not exist: {path}")
            episode[field] = str(path)
    if selection is not None:
        if options.get("adaptation_overrides"):
            raise ValueError("Bind either a completed selection or direct adaptation_overrides, not both")
        selected = json.loads(Path(selection).expanduser().resolve().read_text())
        field = "mechanism_adaptation_overrides" if experiment == "E3" else "adaptation_overrides"
        options["adaptation_overrides"] = selected[field]
        options["source_selection"] = selected["source_selection"]
    requested_device = device if device is not None else config.get("trainer", {}).get("device")
    if not isinstance(requested_device, str) or not requested_device:
        raise ValueError("Provide --device or trainer.device explicitly")
    cfg = {**options, "episodes": episodes, "device": requested_device,
           "config_origin": str(bound_data), "output": str(out)}
    result = run(cfg, bound_data.parent, out, experiment, arms=arms, evaluate_query=phase == "evaluate")
    return {"status": "completed", "result_dir": str(result),
            "run_summary": str(result / "status.json"),
            "test_metrics": str(result / "metrics.csv") if phase == "evaluate" else None,
            "evidence_kind": options.get("information_regime"),
            "scientific_claim": "No source/data qualification or PHM benefit follows from software execution."}

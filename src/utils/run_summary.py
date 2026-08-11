"""Deterministic summaries for repeated experiment results."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _to_builtin(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _to_builtin(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return _to_builtin(scalar)
    raise TypeError(f"unsupported resolved config value: {type(value).__name__}")


def resolved_config_sha256(config: Any) -> str:
    payload = json.dumps(
        _to_builtin(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, numbers.Real):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("run results contain a non-finite numeric metric")
    return number


def build_run_summary(
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    config: Any,
) -> dict[str, Any]:
    if len(results) != len(seeds):
        raise ValueError("one seed must be recorded for every run result")
    if not results:
        raise ValueError("at least one run result is required")

    metric_names = sorted({str(key) for result in results for key in result})
    metrics: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        values = []
        for result in results:
            if name not in result:
                continue
            value = _numeric_value(result[name])
            if value is not None:
                values.append(value)
        if not values:
            continue
        metrics[name] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        }

    return {
        "schema_version": 1,
        "config_sha256": resolved_config_sha256(config),
        "iterations": len(results),
        "seeds": [int(seed) for seed in seeds],
        "metrics": metrics,
    }


def write_run_summary(
    output_path: str | Path,
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    config: Any,
) -> dict[str, Any]:
    summary = build_run_summary(results=results, seeds=seeds, config=config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary

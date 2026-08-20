"""Deterministic summaries for repeated experiment results."""

from __future__ import annotations

import json
import math
import numbers
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


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
    config: Any = None,
) -> dict[str, Any]:
    """Summarize exactly the completed repeated-run estimator.

    ``config`` remains an ignored call-shape parameter for internal compatibility during
    the RC transition. It is not serialized, hashed, or used to determine success.
    """

    del config
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
        "iterations": len(results),
        "seeds": [int(seed) for seed in seeds],
        "metrics": metrics,
    }


def write_run_summary(
    output_path: str | Path,
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    config: Any = None,
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

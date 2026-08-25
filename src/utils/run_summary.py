"""Complete summaries for repeated experiment results."""

from __future__ import annotations

import json
import math
import numbers
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _numeric_metric(value: Any, *, context: str) -> float:
    """Return one finite scalar metric without silently dropping values."""

    if isinstance(value, bool):
        raise TypeError(f"{context} must be numeric, not boolean")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TypeError(f"{context} must be a scalar numeric value") from exc
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{context} must be a scalar real number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} is not finite: {number!r}")
    return number


def normalize_metric_result(
    result: Mapping[str, Any],
    *,
    context: str = "run result",
) -> dict[str, float]:
    """Validate one complete metric mapping and return plain finite floats."""

    if not isinstance(result, Mapping):
        raise TypeError(f"{context} must be a metric mapping")
    if not result:
        raise ValueError(f"{context} must contain at least one metric")

    normalized: dict[str, float] = {}
    for raw_name, raw_value in result.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise TypeError(
                f"{context} metric names must be non-empty strings, got {raw_name!r}"
            )
        name = raw_name.strip()
        if name != raw_name:
            raise ValueError(
                f"{context} metric name {raw_name!r} contains surrounding whitespace"
            )
        if name in normalized:
            raise ValueError(f"{context} contains duplicate metric name {name!r}")
        normalized[name] = _numeric_metric(
            raw_value,
            context=f"{context} metric {name!r}",
        )
    return normalized


def _normalized_seeds(seeds: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"seed {index} must be an integer, got {seed!r}")
        normalized.append(int(seed))
    return normalized


def build_run_summary(
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Summarize one identical finite metric set across every completed seed."""

    if len(results) != len(seeds):
        raise ValueError("one seed must be recorded for every run result")
    if not results:
        raise ValueError("at least one run result is required")

    normalized_results = [
        normalize_metric_result(result, context=f"run result {index}")
        for index, result in enumerate(results)
    ]
    expected_names = set(normalized_results[0])
    for index, result in enumerate(normalized_results[1:], start=1):
        observed_names = set(result)
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            unexpected = sorted(observed_names - expected_names)
            raise ValueError(
                "every seed must report the same metric set: "
                f"run={index}, missing={missing}, unexpected={unexpected}"
            )

    metrics: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        values = [result[name] for result in normalized_results]
        metrics[name] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        }

    normalized_seeds = _normalized_seeds(seeds)
    return {
        "schema_version": 1,
        "iterations": len(normalized_results),
        "seeds": normalized_seeds,
        "metrics": metrics,
    }


def write_run_summary(
    output_path: str | Path,
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Write a validated repeated-run summary to one explicit path."""

    summary = build_run_summary(results=results, seeds=seeds)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "build_run_summary",
    "normalize_metric_result",
    "write_run_summary",
]

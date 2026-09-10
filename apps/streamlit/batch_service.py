"""Finite experiment-grid planning for the optional Streamlit workspace.

This module plans approved parameter combinations only. It does not launch a process,
select a scientific objective, or inspect test results. Execution remains owned by the
existing single-run service.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence, Tuple

try:
    from .config_service import apply_overrides, dump_yaml
except ImportError:  # pragma: no cover
    from config_service import apply_overrides, dump_yaml  # type: ignore


DEFAULT_MAX_TRIALS = 16
DEFAULT_MAX_FITS = 64


class BatchPlanError(ValueError):
    """Raised when a requested finite batch is ambiguous or exceeds its budget."""


@dataclass(frozen=True)
class TrialPlan:
    """One concrete configuration in a finite user-approved batch."""

    index: int
    trial_id: str
    overrides: Tuple[Tuple[str, Any], ...]
    config_yaml: str
    fit_count: int


@dataclass(frozen=True)
class BatchPlan:
    """A deterministic Cartesian product and its explicit execution cost."""

    varying_paths: Tuple[str, ...]
    trials: Tuple[TrialPlan, ...]
    total_fits: int
    max_trials: int
    max_fits: int


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BatchPlanError(f"{name} must be a positive integer.")
    return value


def _same_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:
        return False
    return isinstance(result, bool) and result


def _normalized_values(path: str, values: Sequence[Any]) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BatchPlanError(f"Grid values for {path} must be a finite sequence.")
    copied = tuple(values)
    if not copied:
        raise BatchPlanError(f"Grid values for {path} cannot be empty.")
    for index, value in enumerate(copied):
        if any(_same_value(value, previous) for previous in copied[:index]):
            raise BatchPlanError(
                f"Grid values for {path} contain a duplicate value: {value!r}."
            )
    return copied


def _fit_count(config: Mapping[str, Any]) -> int:
    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        raise BatchPlanError("Resolved batch configuration must contain environment.")
    return _positive_int(environment.get("iterations"), name="environment.iterations")


def plan_grid(
    base_config: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    *,
    allowed_paths: Iterable[str],
    max_trials: int = DEFAULT_MAX_TRIALS,
    max_fits: int = DEFAULT_MAX_FITS,
) -> BatchPlan:
    """Expand one bounded Cartesian product without executing any experiment.

    ``allowed_paths`` is supplied by the current UI catalogue or a future bounded Agent
    tool. The planner never invents a new config field and never changes seed/iterations
    semantics. ``fit_count`` is the resolved ``environment.iterations`` for each trial,
    so a user can see the real multiplicative cost before authorizing execution.
    """

    max_trials = _positive_int(max_trials, name="max_trials")
    max_fits = _positive_int(max_fits, name="max_fits")
    if not isinstance(base_config, Mapping) or not base_config:
        raise BatchPlanError("base_config must be a non-empty resolved configuration.")
    if not isinstance(grid, Mapping) or not grid:
        raise BatchPlanError("At least one varying parameter is required.")

    allowed = set(allowed_paths)
    varying_paths = tuple(grid)
    if any(not isinstance(path, str) or not path.strip() for path in varying_paths):
        raise BatchPlanError("Every grid path must be a non-empty string.")
    unknown = tuple(path for path in varying_paths if path not in allowed)
    if unknown:
        raise BatchPlanError(
            "Batch grid contains fields that are not approved UI controls: "
            + ", ".join(unknown)
        )

    dimensions = tuple(_normalized_values(path, grid[path]) for path in varying_paths)
    trial_count = 1
    for values in dimensions:
        trial_count *= len(values)
    if trial_count > max_trials:
        raise BatchPlanError(
            f"Batch requests {trial_count} trials, exceeding max_trials={max_trials}."
        )

    trials = []
    total_fits = 0
    for index, combination in enumerate(product(*dimensions), start=1):
        overrides = tuple(zip(varying_paths, combination))
        try:
            resolved = apply_overrides(base_config, overrides)
        except Exception as error:
            raise BatchPlanError(f"Could not apply trial {index} overrides: {error}") from error
        fit_count = _fit_count(resolved)
        total_fits += fit_count
        if total_fits > max_fits:
            raise BatchPlanError(
                f"Batch requires {total_fits} fits by trial {index}, exceeding "
                f"max_fits={max_fits}."
            )
        trials.append(
            TrialPlan(
                index=index,
                trial_id=f"trial-{index:03d}",
                overrides=overrides,
                config_yaml=dump_yaml(resolved),
                fit_count=fit_count,
            )
        )

    return BatchPlan(
        varying_paths=varying_paths,
        trials=tuple(trials),
        total_fits=total_fits,
        max_trials=max_trials,
        max_fits=max_fits,
    )

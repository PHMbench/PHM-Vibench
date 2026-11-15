"""Pretraining/few-shot Pipeline with explicit, non-fallback execution modes."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path
from typing import Any

from src.configs.config_utils import load_config
from src.runtime import run_classification_pipeline
from src.utils.config.pipeline_adapters import adapt_p02
from src.utils.config_utils import apply_overrides_to_config, parse_overrides
from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator


LEGACY_DUAL_YAML_MODE = "legacy_dual_yaml"


def _config_and_stage_overrides(args: Any) -> tuple[Any, list[str]]:
    """Return one config object and only the overrides not already compiled."""

    compiled = getattr(args, "compiled_run_spec", None)
    if compiled is not None:
        return compiled.runtime_config(), []

    config_path = getattr(args, "config_path", None)
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("Pipeline 02 requires args.config_path")
    return load_config(config_path), list(getattr(args, "override", None) or ())


def _has_stages(config: Any) -> bool:
    if isinstance(config, dict):
        stages = config.get("stages")
    else:
        stages = getattr(config, "stages", None)
    return isinstance(stages, (list, tuple)) and bool(stages)


def _metric_scalar(value: Any, *, stage_name: str, metric_name: str) -> float:
    """Return one finite scalar metric without guessing non-scalar reductions."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Pipeline 02 stage {stage_name!r} metric {metric_name!r} must be "
                "a scalar value."
            ) from exc

    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(
            f"Pipeline 02 stage {stage_name!r} metric {metric_name!r} must be "
            f"a numeric scalar, got {type(value).__name__}."
        )

    scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(
            f"Pipeline 02 stage {stage_name!r} metric {metric_name!r} is not finite: "
            f"{scalar!r}."
        )
    return scalar


def _require_completed_stage_evaluation(
    result: Any, mode: str, expected: Mapping[str, bool] | None = None,
) -> Any:
    """Require every requested stage; evaluation gates apply only when declared."""
    if not isinstance(result, Mapping) or not result:
        raise RuntimeError(
            f"Pipeline 02 {mode} must return a non-empty stage result mapping."
        )

    actual_names = [name for name in result if not str(name).startswith("_")]
    if expected is not None and actual_names != list(expected):
        raise RuntimeError(
            f"Pipeline 02 {mode} stage coverage/order mismatch: "
            f"requested={list(expected)!r}, returned={actual_names!r}"
        )

    stage_count = 0
    for stage_name, stage_result in result.items():
        if str(stage_name).startswith("_"):
            continue
        stage_count += 1
        if not isinstance(stage_result, Mapping):
            raise RuntimeError(
                f"Pipeline 02 stage {stage_name!r} returned "
                f"{type(stage_result).__name__}; expected a result mapping."
            )
        if expected is not None and not expected[stage_name]:
            if stage_result.get("status") != "trained_not_evaluated" or "metrics" in stage_result:
                raise RuntimeError(f"Pipeline 02 stage {stage_name!r} must report trained_not_evaluated without test metrics")
            checkpoint = stage_result.get("checkpoint_path")
            if not checkpoint or not Path(checkpoint).is_file():
                raise RuntimeError(f"Pipeline 02 stage {stage_name!r} has no readable selected checkpoint: {checkpoint!r}")
            continue
        metrics = stage_result.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise RuntimeError(
                f"Pipeline 02 stage {stage_name!r} did not complete evaluation: "
                "expected a non-empty metrics mapping from trainer.test."
            )
        for metric_name, value in metrics.items():
            _metric_scalar(
                value,
                stage_name=str(stage_name),
                metric_name=str(metric_name),
            )

    if stage_count == 0:
        raise RuntimeError(
            f"Pipeline 02 {mode} returned no stage results."
        )
    return result


def _run_unified_multistage(args: Any, config: Any, overrides: list[str]) -> Any:
    """Run the one supported multi-stage implementation without fallback."""

    print("[INFO] Pipeline 02 mode: unified_multistage")
    orchestrator = TwoStageOrchestrator(config, cli_overrides=overrides)
    expected = orchestrator.stage_requirements()
    result = orchestrator.run_all_stages()
    return _require_completed_stage_evaluation(result, "unified_multistage", expected)


def _run_legacy_dual_yaml(args: Any) -> Any:
    """Run the isolated dual-YAML adapter only after explicit opt-in."""

    mode = str(getattr(args, "pipeline_mode", "") or "")
    if mode != LEGACY_DUAL_YAML_MODE:
        raise ValueError(
            "fs_config_path is a legacy dual-YAML input; set "
            f"pipeline_mode={LEGACY_DUAL_YAML_MODE!r} explicitly"
        )

    config_path = getattr(args, "config_path", None)
    fs_config_path = getattr(args, "fs_config_path", None)
    if not config_path or not fs_config_path:
        raise ValueError("legacy_dual_yaml requires config_path and fs_config_path")

    unified = adapt_p02(
        config_path,
        fs_config_path,
        getattr(args, "local_config", None),
    )
    overrides = getattr(args, "override", None)
    if overrides:
        unified = apply_overrides_to_config(unified, parse_overrides(overrides))

    print("[INFO] Pipeline 02 mode: legacy_dual_yaml")
    orchestrator = TwoStageOrchestrator(unified)
    expected = {f"stage_{index+1}": evaluate
                for index, evaluate in enumerate(orchestrator.stage_requirements().values())}
    result = orchestrator.run_complete()
    return _require_completed_stage_evaluation(result, "legacy_dual_yaml", expected)


def pipeline(args: Any) -> Any:
    """Select exactly one execution mode from explicit inputs and config structure.

    - ``fs_config_path`` requires explicit ``legacy_dual_yaml`` opt-in;
    - a compiled config containing non-empty ``stages`` uses the unified orchestrator;
    - a config without ``stages`` uses the shared classification runtime.

    Every requested stage must complete. Evaluation stages require finite metrics;
    train-only stages require a selected state and explicitly report no evaluation.
    No exception changes the selected mode or activates a second implementation.
    """

    if getattr(args, "fs_config_path", None):
        return _run_legacy_dual_yaml(args)

    config, stage_overrides = _config_and_stage_overrides(args)
    if _has_stages(config):
        return _run_unified_multistage(args, config, stage_overrides)

    print("[INFO] Pipeline 02 mode: single_stage")
    return run_classification_pipeline(args)

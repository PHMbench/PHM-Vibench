"""Pretraining/few-shot Pipeline with explicit, non-fallback execution modes."""

from __future__ import annotations

from collections.abc import Mapping
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


def _require_completed_stage_evaluation(result: Any, mode: str) -> Any:
    """Reject multi-stage success when a stage has no completed evaluation metrics.

    The current Pipeline 02 multi-stage contract always calls ``trainer.test`` after
    each trained stage. An empty metric mapping therefore means evaluation did not
    complete and the public run must fail rather than report stage success.
    """
    if not isinstance(result, Mapping) or not result:
        raise RuntimeError(
            f"Pipeline 02 {mode} must return a non-empty stage result mapping."
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
        metrics = stage_result.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise RuntimeError(
                f"Pipeline 02 stage {stage_name!r} did not complete evaluation: "
                "expected a non-empty metrics mapping from trainer.test."
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
    result = orchestrator.run_complete()
    return _require_completed_stage_evaluation(result, "unified_multistage")


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
    result = TwoStageOrchestrator(unified).run_complete()
    return _require_completed_stage_evaluation(result, "legacy_dual_yaml")


def pipeline(args: Any) -> Any:
    """Select exactly one execution mode from explicit inputs and config structure.

    - ``fs_config_path`` requires explicit ``legacy_dual_yaml`` opt-in;
    - a compiled config containing non-empty ``stages`` uses the unified orchestrator;
    - a config without ``stages`` uses the shared classification runtime.

    Multi-stage success requires a non-empty evaluation metric mapping for every stage.
    No exception changes the selected mode or activates a second implementation.
    """

    if getattr(args, "fs_config_path", None):
        return _run_legacy_dual_yaml(args)

    config, stage_overrides = _config_and_stage_overrides(args)
    if _has_stages(config):
        return _run_unified_multistage(args, config, stage_overrides)

    print("[INFO] Pipeline 02 mode: single_stage")
    return run_classification_pipeline(args)

"""Guarded runtime shell for PHM generative pipelines.

The public entrypoint remains::

    python main.py --config <yaml> [--override key=value ...]

This first migration slice deliberately implements configuration loading,
preflight validation, stage selection, and iteration dispatch only. Concrete
train/sample/eval implementations are added by later focused PRs so the
unrelated-history source snapshot is never merged wholesale.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STAGE_NAMES = frozenset({"train", "sample", "eval"})
REQUIRED_CONFIG_SECTIONS = ("environment", "data", "model", "task", "trainer")


def _get_attr(value: Any, key: str, default: Any = None) -> Any:
    """Read one config field from mapping- or namespace-style objects."""

    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _validate_required_sections(configs: Any) -> None:
    """Reject incomplete five-block configurations before factory dispatch."""

    missing = [
        section
        for section in REQUIRED_CONFIG_SECTIONS
        if _get_attr(configs, section, None) is None
    ]
    if missing:
        raise ValueError(
            "generative config is missing required section(s): "
            + ", ".join(missing)
        )


def _load_configs(args: Any) -> Any:
    """Load the public config path and apply normal local/CLI overrides."""

    from src.configs.config_utils import merge_with_local_override
    from src.utils.config_utils import apply_overrides_to_config, parse_overrides

    config_path = getattr(args, "config_path", None)
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("Pipeline_06_generative requires args.config_path")

    configs = merge_with_local_override(
        config_path,
        getattr(args, "local_config", None),
    )
    overrides = getattr(args, "override", None)
    if overrides:
        configs = apply_overrides_to_config(configs, parse_overrides(overrides))

    _validate_required_sections(configs)
    return configs


def _generative_cfg(configs: Any) -> Any:
    """Return ``task.generative`` and reject non-generative task configs."""

    task_cfg = _get_attr(configs, "task", None)
    generative_cfg = _get_attr(task_cfg, "generative", None)
    if generative_cfg is None:
        raise ValueError(
            "Pipeline_06_generative requires task.generative configuration"
        )
    return generative_cfg


def _resolve_mode(configs: Any) -> str:
    """Resolve and validate the explicit Pipeline 06 stage."""

    mode = str(_get_attr(_generative_cfg(configs), "mode", "train")).strip().lower()
    if mode not in STAGE_NAMES:
        supported = ", ".join(sorted(STAGE_NAMES))
        raise ValueError(
            f"unsupported generative mode {mode!r}; expected one of: {supported}"
        )
    return mode


def _resolve_iterations(configs: Any) -> int:
    """Return the positive number of independently recorded iterations."""

    environment_cfg = _get_attr(configs, "environment", None)
    iterations = int(_get_attr(environment_cfg, "iterations", 1))
    if iterations <= 0:
        raise ValueError(
            f"environment.iterations must be positive, got {iterations}"
        )
    return iterations


def _validate_stage_inputs(mode: str, generative_cfg: Any) -> None:
    """Fail before deep runtime code when required stage artifacts are absent."""

    if mode == "sample":
        checkpoint_path = _get_attr(generative_cfg, "checkpoint_path", None)
        allow_untrained_smoke = bool(
            _get_attr(generative_cfg, "allow_untrained_smoke", False)
        )
        if not checkpoint_path and not allow_untrained_smoke:
            raise ValueError(
                "generative sample mode requires "
                "task.generative.checkpoint_path; set "
                "allow_untrained_smoke=true only for an explicitly untrained smoke"
            )

    if mode == "eval" and not _get_attr(
        generative_cfg,
        "generated_path",
        None,
    ):
        raise ValueError(
            "generative eval mode requires task.generative.generated_path"
        )


def _stage_not_integrated(mode: str) -> RuntimeError:
    return RuntimeError(
        f"Pipeline 06 {mode} runtime is not integrated in the G1 shell; "
        "use a later reviewed migration slice that provides the concrete stage"
    )


def _run_train_stage(args: Any, configs: Any, iteration: int) -> Any:
    """G1 placeholder replaced by the reviewed CFM/runtime implementation."""

    raise _stage_not_integrated("train")


def _run_sample_stage(args: Any, configs: Any, iteration: int) -> Any:
    """G1 placeholder replaced by the reviewed sampler/artifact implementation."""

    raise _stage_not_integrated("sample")


def _run_eval_stage(args: Any, configs: Any, iteration: int) -> Any:
    """G1 placeholder replaced by the reviewed evaluation implementation."""

    raise _stage_not_integrated("eval")


def pipeline(args: Any) -> list[Any]:
    """Load, preflight, and dispatch one explicit generative stage.

    Train, sample, and eval remain separate invocations. A future optional
    orchestrator may invoke ``main.py`` three times, but it must not bypass this
    public entrypoint or hide intermediate checkpoint/sample paths.
    """

    configs = _load_configs(args)
    mode = _resolve_mode(configs)
    generative_cfg = _generative_cfg(configs)
    _validate_stage_inputs(mode, generative_cfg)

    handlers = {
        "train": _run_train_stage,
        "sample": _run_sample_stage,
        "eval": _run_eval_stage,
    }
    handler = handlers[mode]

    results: list[Any] = []
    for iteration in range(_resolve_iterations(configs)):
        results.append(handler(args, configs, iteration))
    return results

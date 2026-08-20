"""Public resolved-config adapter for Pipeline 06.

The scientific train/sample/eval implementation remains in
``src.Pipeline_06_Generative_Modeling``. This adapter consumes the same
``args.resolved_config_data`` mapping as every other maintained Pipeline and never
re-reads YAML, reapplies CLI overrides, or discovers a machine-local file.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from src.configs.config_utils import dict_to_namespace


PIPELINE_NAME = "Pipeline_06_Generative_Modeling"


def _runtime_config(args: Any, implementation: Any) -> Any:
    """Return the one schema-validated public configuration."""

    resolved = getattr(args, "resolved_config_data", None)
    if not isinstance(resolved, Mapping):
        raise ValueError(
            "Pipeline 06 requires args.resolved_config_data from "
            "phmfactory.config.analyze_config; direct YAML reloading is not supported"
        )

    configured = resolved.get("pipeline")
    expected = getattr(args, "resolved_pipeline", None)
    if configured != PIPELINE_NAME or expected != PIPELINE_NAME:
        raise ValueError(
            "Pipeline 06 resolved Pipeline mismatch: "
            f"config={configured!r}, dispatch={expected!r}"
        )

    configs = dict_to_namespace(deepcopy(dict(resolved)))
    implementation._validate_required_sections(configs)
    return configs


def pipeline(args: Any) -> list[Any]:
    """Dispatch one explicit Pipeline 06 stage from the resolved config contract."""

    import src.Pipeline_06_Generative_Modeling as implementation

    configs = _runtime_config(args, implementation)
    mode = implementation._resolve_mode(configs)
    generative_cfg = implementation._generative_cfg(configs)
    implementation._validate_stage_inputs(mode, generative_cfg)

    handlers = {
        "train": implementation._run_train_stage,
        "sample": implementation._run_sample_stage,
        "eval": implementation._run_eval_stage,
    }
    handler = handlers[mode]

    results: list[Any] = []
    for iteration in range(implementation._resolve_iterations(configs)):
        try:
            results.append(handler(args, configs, iteration))
        except Exception as error:
            try:
                implementation._record_stage(
                    configs,
                    mode,
                    status="failed",
                    iteration=iteration,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            except Exception as ledger_error:
                raise error from ledger_error
            raise
    return results

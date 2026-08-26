"""Public compiled-config adapter for Pipeline 06.

The scientific train/sample/eval implementation remains in
``src.Pipeline_06_Generative_Modeling``. This adapter owns only the public control-plane
boundary: it consumes ``CompiledRunSpec.runtime_config()`` exactly once and delegates the
selected stage without re-reading YAML, reapplying CLI overrides, or discovering a local
machine file.

Direct legacy imports of the original ``src`` module retain its explicit compatibility
loader. The maintained CLI resolves Pipeline 06 to this adapter.
"""

from __future__ import annotations

from typing import Any

from src.configs.config_utils import dict_to_namespace


PIPELINE_NAME = "Pipeline_06_Generative_Modeling"


def _runtime_config(args: Any, implementation: Any) -> Any:
    """Return compiled public config or the original direct-call compatibility config."""

    compiled = getattr(args, "compiled_run_spec", None)
    if compiled is None:
        # Compatibility-only path for code that imports the historical src module contract
        # without going through the PHMFactory public CLI.
        return implementation._load_configs(args)

    expected = getattr(args, "resolved_pipeline", compiled.pipeline)
    if compiled.pipeline != PIPELINE_NAME or expected != PIPELINE_NAME:
        raise ValueError(
            "Pipeline 06 compiled Pipeline mismatch: "
            f"spec={compiled.pipeline!r}, dispatch={expected!r}"
        )
    configs = dict_to_namespace(compiled.runtime_config())
    implementation._validate_required_sections(configs)
    return configs


def pipeline(args: Any) -> dict[str, Any]:
    """Dispatch one explicit Pipeline 06 stage and return a public result mapping."""

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
    return {
        "status": "succeeded",
        "stage": mode,
        "iterations": results,
    }

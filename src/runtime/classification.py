"""Shared classification execution spine for maintained PHMFactory Pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_utils import (
    dict_to_namespace,
    merge_with_local_override,
    path_name,
    transfer_namespace,
)
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer
from src.utils.config_utils import apply_overrides_to_config, parse_overrides
from src.utils.run_summary import (
    build_run_summary,
    normalize_metric_result,
    write_run_summary,
)
from src.utils.utils import close_lab, init_lab, load_best_model_checkpoint


REQUIRED_SECTIONS = ("environment", "data", "model", "task", "trainer")


@dataclass
class ClassificationContext:
    """Mutable context passed to narrow Pipeline-specific hooks."""

    args: Any
    configs: Any
    args_environment: Any
    args_data: Any
    args_model: Any
    args_task: Any
    args_trainer: Any
    iteration: int
    path: Path
    name: str
    data_factory: Any = None
    model: Any = None
    task: Any = None
    trainer: Any = None
    result: dict[str, Any] | None = None


class ClassificationHooks:
    """Optional extension points around the shared classification lifecycle."""

    def on_iteration_start(self, context: ClassificationContext) -> None:
        """Run after the output directory and logger name are resolved."""

    def after_stack_built(self, context: ClassificationContext) -> None:
        """Run after data/model/task/trainer construction and before fitting."""

    def after_test(self, context: ClassificationContext) -> None:
        """Run after the result CSV is written and before cleanup."""


def _required_sections(configs: Any) -> None:
    missing = [section for section in REQUIRED_SECTIONS if not hasattr(configs, section)]
    if missing:
        raise ValueError(
            "classification config is missing required section(s): " + ", ".join(missing)
        )


def load_runtime_config(args: Any) -> Any:
    """Return the exact public compiled config or the explicit legacy fallback.

    Maintained public entrypoints attach ``compiled_run_spec`` and therefore never
    reparse YAML or auto-discover ``configs/local/local.yaml`` here. Direct imports of
    old Pipeline modules retain the historical loader as a compatibility boundary.
    """

    compiled = getattr(args, "compiled_run_spec", None)
    if compiled is not None:
        expected = getattr(args, "resolved_pipeline", compiled.pipeline)
        if compiled.pipeline != expected:
            raise ValueError(
                "compiled Pipeline mismatch: "
                f"spec={compiled.pipeline!r}, dispatch={expected!r}"
            )
        configs = dict_to_namespace(compiled.runtime_config())
    else:
        config_path = getattr(args, "config_path", None)
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("classification Pipeline requires args.config_path")
        configs = merge_with_local_override(
            config_path,
            getattr(args, "local_config", None),
        )
        overrides = getattr(args, "override", None)
        if overrides:
            configs = apply_overrides_to_config(configs, parse_overrides(overrides))

    _required_sections(configs)
    return configs


def _namespaces(configs: Any) -> tuple[Any, Any, Any, Any, Any]:
    return tuple(
        transfer_namespace(getattr(configs, section)) for section in REQUIRED_SECTIONS
    )


def _set_environment(args_environment: Any) -> None:
    for key, value in vars(args_environment).items():
        if str(key).isupper():
            os.environ[str(key)] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")


def _close_data_factory(data_factory: Any) -> None:
    data = getattr(data_factory, "data", None)
    close = getattr(data, "close", None)
    if callable(close):
        close()


def _result_row(result: Any) -> dict[str, float]:
    """Return one complete metric population from ``trainer.test``.

    Lightning returns one mapping per test dataloader. The maintained classification
    estimator currently defines exactly one test population. Multiple mappings are
    ambiguous and require an explicit multi-population protocol.
    """

    if not isinstance(result, (list, tuple)) or len(result) != 1:
        observed = len(result) if isinstance(result, (list, tuple)) else type(result).__name__
        raise RuntimeError(
            "trainer.test must return exactly one metric mapping for the maintained "
            f"classification test population, observed={observed}"
        )
    if not isinstance(result[0], Mapping):
        raise RuntimeError(
            "trainer.test result 0 must be a metric mapping, "
            f"got {type(result[0]).__name__}"
        )
    return normalize_metric_result(result[0], context="trainer.test result 0")


def _best_checkpoint_path(trainer: Any) -> Path:
    callback = next(
        (
            item
            for item in getattr(trainer, "callbacks", ())
            if isinstance(item, ModelCheckpoint)
        ),
        None,
    )
    if callback is None or not callback.best_model_path:
        raise RuntimeError(
            "best checkpoint path is unavailable after checkpoint restoration"
        )
    path = Path(callback.best_model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint does not exist: {path}")
    return path.resolve()


def _write_aggregate_outputs(
    run_root: str | Path,
    last_iteration_path: str | Path | None,
    all_results: list[dict[str, Any]],
    run_seeds: list[int],
) -> dict[str, Any]:
    """Write repeated-run metrics only after the complete estimator validates."""

    if last_iteration_path is None or not all_results:
        raise ValueError("aggregate outputs require at least one completed iteration")

    build_run_summary(results=all_results, seeds=run_seeds)

    run_root_path = Path(run_root)
    iteration_path = Path(last_iteration_path)
    results = pd.DataFrame(all_results)
    results.to_csv(iteration_path / "all_results.csv", index=False)
    results.to_csv(run_root_path / "all_results.csv", index=False)
    return write_run_summary(
        run_root_path / "run_summary.json",
        results=all_results,
        seeds=run_seeds,
    )


def _public_result(
    *,
    final_path: Path,
    best_checkpoints: list[Path],
    all_results: list[dict[str, Any]],
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the direct user-facing outputs of one classification invocation."""

    if not best_checkpoints:
        raise RuntimeError("classification completed without a best checkpoint")

    result_root = final_path.parent.resolve()
    test_metrics = result_root / "all_results.csv"
    run_summary = result_root / "run_summary.json"
    if summary is not None:
        if not test_metrics.is_file():
            raise FileNotFoundError(f"aggregate test metrics are missing: {test_metrics}")
        if not run_summary.is_file():
            raise FileNotFoundError(f"run summary is missing: {run_summary}")

    return {
        "status": "succeeded",
        "result_dir": str(result_root),
        "best_checkpoint": str(best_checkpoints[-1]),
        "best_checkpoints": [str(path) for path in best_checkpoints],
        "test_metrics": str(test_metrics) if summary is not None else None,
        "run_summary": str(run_summary) if summary is not None else None,
        "primary_metrics": dict(summary.get("metrics", {})) if summary else {},
        "iterations": [dict(item) for item in all_results],
    }


def run_classification_pipeline(
    args: Any,
    *,
    hooks: ClassificationHooks | None = None,
) -> dict[str, Any]:
    """Execute the shared train/test lifecycle and return direct output paths."""

    hooks = hooks or ClassificationHooks()
    configs = load_runtime_config(args)
    (
        args_environment,
        args_data,
        args_model,
        args_task,
        args_trainer,
    ) = _namespaces(configs)

    if getattr(args_task, "name", None) == "Multitask":
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list

    if not hasattr(args_environment, "iterations"):
        raise ValueError("environment.iterations is required")
    iterations = args_environment.iterations
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError(
            "environment.iterations must be an integer, "
            f"got {type(iterations).__name__}"
        )
    if iterations <= 0:
        raise ValueError(f"environment.iterations must be positive, got {iterations}")

    if not hasattr(args_environment, "seed"):
        raise ValueError("environment.seed is required")
    base_seed = args_environment.seed
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError(
            "environment.seed must be an integer, "
            f"got {type(base_seed).__name__}"
        )

    if not hasattr(args_trainer, "test_after_fit"):
        raise ValueError(
            "trainer.test_after_fit is required for classification Pipelines"
        )
    test_after_fit = args_trainer.test_after_fit
    if not isinstance(test_after_fit, bool):
        raise TypeError("trainer.test_after_fit must be a boolean")
    _set_environment(args_environment)

    all_results: list[dict[str, Any]] = []
    run_seeds: list[int] = []
    best_checkpoints: list[Path] = []
    final_path: Path | None = None

    for iteration in range(iterations):
        print(
            f"\n{'=' * 50}\n"
            f"[INFO] 开始实验迭代 {iteration + 1}/{iterations}\n"
            f"{'=' * 50}"
        )
        raw_path, name = path_name(configs, iteration)
        path = Path(raw_path)
        path.mkdir(parents=True, exist_ok=True)
        final_path = path
        args_trainer.logger_name = name

        current_seed = base_seed + iteration
        run_seeds.append(current_seed)
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")

        context = ClassificationContext(
            args=args,
            configs=configs,
            args_environment=args_environment,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            iteration=iteration,
            path=path,
            name=str(name),
        )
        lab_started = False
        try:
            init_lab(args_environment, args, name)
            lab_started = True
            hooks.on_iteration_start(context)

            print("[INFO] 构建数据工厂...")
            context.data_factory = build_data(args_data, args_task)
            metadata = context.data_factory.get_metadata()

            print("[INFO] 构建模型...")
            context.model = build_model(args_model, metadata=metadata)

            print("[INFO] 构建任务...")
            context.task = build_task(
                args_task=args_task,
                network=context.model,
                args_data=args_data,
                args_model=args_model,
                args_trainer=args_trainer,
                args_environment=args_environment,
                metadata=metadata,
            )

            print("[INFO] 构建训练器...")
            context.trainer = build_trainer(
                args_environment,
                args_trainer,
                args_data,
                str(path),
            )

            hooks.after_stack_built(context)

            print("[INFO] 开始训练...")
            context.trainer.fit(
                context.task,
                context.data_factory.get_dataloader("train"),
                context.data_factory.get_dataloader("val"),
            )

            print("[INFO] 加载最佳模型并测试...")
            context.task = load_best_model_checkpoint(context.task, context.trainer)
            best_checkpoints.append(_best_checkpoint_path(context.trainer))
            if not test_after_fit:
                continue
            context.result = _result_row(
                context.trainer.test(
                    context.task,
                    context.data_factory.get_dataloader("test"),
                )
            )
            all_results.append(context.result)

            print("[INFO] 保存测试结果...")
            metrics_path = path / f"test_result_{iteration}.csv"
            pd.DataFrame([context.result]).to_csv(metrics_path, index=False)
            hooks.after_test(context)
        finally:
            if context.data_factory is not None:
                _close_data_factory(context.data_factory)
            if lab_started:
                close_lab()

    if final_path is None:
        raise RuntimeError("classification Pipeline produced no iteration path")
    if not test_after_fit:
        print(f"\n{'=' * 50}\n[INFO] 训练完成；配置禁止测试\n{'=' * 50}")
        return _public_result(
            final_path=final_path,
            best_checkpoints=best_checkpoints,
            all_results=all_results,
            summary=None,
        )

    summary = _write_aggregate_outputs(
        final_path.parent,
        final_path,
        all_results,
        run_seeds,
    )
    print(f"\n{'=' * 50}\n[INFO] 所有实验已完成\n{'=' * 50}")
    return _public_result(
        final_path=final_path,
        best_checkpoints=best_checkpoints,
        all_results=all_results,
        summary=summary,
    )

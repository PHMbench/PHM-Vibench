"""Shared classification execution spine for maintained PHMFactory Pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pytorch_lightning import seed_everything

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
from src.utils.run_summary import write_run_summary
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


def _result_row(result: Any) -> dict[str, Any]:
    if not isinstance(result, list) or not result or not isinstance(result[0], dict):
        raise RuntimeError(
            "trainer.test must return a non-empty list whose first item is a mapping"
        )
    return dict(result[0])


def _write_aggregate_outputs(
    run_root: str | Path,
    last_iteration_path: str | Path | None,
    all_results: list[dict[str, Any]],
    run_seeds: list[int],
    configs: Any,
) -> dict[str, Any]:
    """Write repeated-run metrics and their deterministic summary."""

    if last_iteration_path is None or not all_results:
        raise ValueError("aggregate outputs require at least one completed iteration")

    run_root_path = Path(run_root)
    iteration_path = Path(last_iteration_path)
    results = pd.DataFrame(all_results)
    results.to_csv(iteration_path / "all_results.csv", index=False)
    results.to_csv(run_root_path / "all_results.csv", index=False)
    return write_run_summary(
        run_root_path / "run_summary.json",
        results=all_results,
        seeds=run_seeds,
        config=configs,
    )


def _register_iteration_evidence(
    args: Any,
    *,
    iteration: int,
    seed: int,
    path: Path,
    metrics_path: Path,
) -> None:
    attestation = getattr(args, "run_attestation", None)
    if attestation is None:
        return
    artifact = attestation.register_artifact(
        role="classification_test_metrics",
        path=metrics_path,
        metadata={"iteration": iteration, "run_dir": str(path)},
    )
    attestation.append_evidence(
        "classification_iterations",
        {
            "iteration": iteration,
            "seed": seed,
            "run_dir": str(path),
            "metrics_artifact": artifact,
        },
    )


def run_classification_pipeline(
    args: Any,
    *,
    hooks: ClassificationHooks | None = None,
) -> list[dict[str, Any]]:
    """Execute the shared train/test lifecycle for Pipeline 01 and Pipeline 05."""

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

    iterations = int(getattr(args_environment, "iterations", 0))
    if iterations <= 0:
        raise ValueError(f"environment.iterations must be positive, got {iterations}")
    base_seed = int(getattr(args_environment, "seed", 0))
    test_after_fit = getattr(args_trainer, "test_after_fit", True)
    if not isinstance(test_after_fit, bool):
        raise TypeError(
            "trainer.test_after_fit must be a boolean when it is provided"
        )
    _set_environment(args_environment)

    all_results: list[dict[str, Any]] = []
    run_seeds: list[int] = []
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
            _register_iteration_evidence(
                args,
                iteration=iteration,
                seed=current_seed,
                path=path,
                metrics_path=metrics_path,
            )
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
        return []
    _write_aggregate_outputs(
        final_path.parent,
        final_path,
        all_results,
        run_seeds,
        configs,
    )
    aggregate_path = final_path / "all_results.csv"
    attestation = getattr(args, "run_attestation", None)
    if attestation is not None:
        attestation.register_artifact(
            role="classification_aggregate_metrics",
            path=aggregate_path,
            metadata={"iterations": iterations},
        )
    print(f"\n{'=' * 50}\n[INFO] 所有实验已完成\n{'=' * 50}")
    return all_results

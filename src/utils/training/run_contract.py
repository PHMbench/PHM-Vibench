from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.configs.config_utils import ConfigWrapper, path_name, save_config
from src.data_factory import build_data
from src.explain_factory.run_artifacts import (
    write_data_metadata_snapshot_from_data_factory,
    write_run_artifact_sidecars,
)
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer
from src.trainer_factory.extensions import write_run_manifest


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    logger_name: str
    seed: int
    iteration: int


@dataclass(frozen=True)
class RuntimeComponents:
    data_factory: Any
    model: Any
    task: Any
    trainer: Any
    batch_meta: Dict[str, Any]
    meta_source: str
    degraded: bool


def make_config_wrapper(
    args_environment: Any,
    args_data: Any,
    args_model: Any,
    args_task: Any,
    args_trainer: Any,
) -> ConfigWrapper:
    return ConfigWrapper(
        environment=args_environment,
        data=args_data,
        model=args_model,
        task=args_task,
        trainer=args_trainer,
    )


def prepare_run_context(
    configs: Any,
    args_environment: Any,
    args_trainer: Any,
    *,
    iteration: int = 0,
    seed_offset: Optional[int] = None,
) -> RunContext:
    """Create the run directory, attach trainer run fields, and snapshot config."""

    run_dir, logger_name = path_name(configs, iteration)
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    args_trainer.logger_name = logger_name
    args_trainer.run_dir = str(run_dir_path)
    save_config(configs, run_dir_path / "config_snapshot.yaml")

    base_seed = getattr(args_environment, "seed", 42)
    offset = int(iteration) if seed_offset is None else int(seed_offset)
    return RunContext(
        run_dir=run_dir_path,
        logger_name=logger_name,
        seed=int(base_seed) + offset,
        iteration=int(iteration),
    )


def build_training_stack(
    *,
    args_environment: Any,
    args_data: Any,
    args_model: Any,
    args_task: Any,
    args_trainer: Any,
    run_dir: str | Path,
    attach_data_factory: bool = False,
    sidecar_config: Any = None,
) -> RuntimeComponents:
    """Build PHM-Vibench data/model/task/trainer components for one run."""

    data_factory = build_data(args_data, args_task)
    if sidecar_config is None:
        batch_meta, meta_source, degraded = write_data_metadata_snapshot_from_data_factory(
            run_dir=Path(run_dir),
            data_factory=data_factory,
        )
    else:
        batch_meta, meta_source, degraded = write_run_artifact_sidecars(
            run_dir=Path(run_dir),
            cfg=sidecar_config,
            args_trainer=args_trainer,
            data_factory=data_factory,
        )
    metadata = data_factory.get_metadata()
    model = build_model(args_model, metadata=metadata)
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata,
    )
    if attach_data_factory and task is not None:
        setattr(task, "_data_factory", data_factory)
    trainer = build_trainer(args_environment, args_trainer, args_data, str(run_dir))
    return RuntimeComponents(
        data_factory=data_factory,
        model=model,
        task=task,
        trainer=trainer,
        batch_meta=batch_meta,
        meta_source=str(meta_source),
        degraded=bool(degraded),
    )


def write_test_result_and_manifest(
    *,
    run_dir: str | Path,
    metrics: Dict[str, Any],
    iteration: int,
    args_trainer: Any,
    seed: int,
    trainer: Any,
    stage: str = "test",
    paper_id: str = "",
    preset_version: str = "",
    manifest_required: bool = True,
) -> Path:
    """Write metrics CSV and required manifest for a completed run."""

    run_dir_path = Path(run_dir)
    metrics_path = run_dir_path / f"test_result_{int(iteration)}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    if _manifest_enabled(args_trainer) and _is_main_process():
        write_run_manifest(
            run_dir=run_dir_path,
            stage=stage,
            paper_id=paper_id,
            preset_version=preset_version,
            run_id=str(getattr(args_trainer, "logger_name", "") or ""),
            seed=seed,
            trainer=trainer,
            required=manifest_required,
        )
    return metrics_path


def _is_main_process() -> bool:
    return "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0


def _manifest_enabled(args_trainer: Any) -> bool:
    extensions = getattr(args_trainer, "extensions", None)
    report_cfg = getattr(extensions, "report", None) if extensions is not None else None
    report_enable = getattr(report_cfg, "enable", True) if report_cfg is not None else True
    manifest_enable = getattr(report_cfg, "manifest", True) if report_cfg is not None else True
    return bool(report_enable) and bool(manifest_enable)

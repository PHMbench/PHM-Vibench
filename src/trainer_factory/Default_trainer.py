from __future__ import annotations

import os
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import CSVLogger, WandbLogger

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:  # Optional experiment service; required only when enabled.
    SwanLabLogger = None

from phmfactory.device import resolve_device_request
from src.trainer_factory import register_trainer

# Compatibility name for focused tests and historical internal imports.
_resolve_device_request = resolve_device_request

# 获取当前进程的排名
is_main_process = True  # 默认为主进程
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    is_main_process = local_rank == 0


_SELECTION_MODES = frozenset({"min", "max"})


def resolve_epoch_contract(args: Any) -> int:
    """Return the single explicit positive epoch count without alias fallback."""

    if hasattr(args, "max_epochs"):
        raise ValueError(
            "trainer.max_epochs is unsupported; use the single public field "
            "trainer.num_epochs"
        )
    if not hasattr(args, "num_epochs"):
        raise ValueError("trainer.num_epochs is required")
    value = args.num_epochs
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "trainer.num_epochs must be an integer, "
            f"got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"trainer.num_epochs must be positive, got {value}")
    return value


def resolve_selection_contract(args: Any) -> tuple[str, str]:
    """Return the exact checkpoint metric and optimization direction.

    PHMFactory never infers direction from a metric name.  The same explicit pair is
    consumed by ModelCheckpoint and EarlyStopping so checkpoint restoration cannot use a
    different estimator direction from stopping.
    """

    monitor = getattr(args, "monitor", None)
    if not isinstance(monitor, str) or not monitor.strip():
        raise ValueError(
            "trainer.monitor is required and must name one logged validation metric"
        )

    raw_mode = getattr(args, "monitor_mode", None)
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise ValueError(
            "trainer.monitor_mode is required and must be 'min' or 'max'; "
            "PHMFactory does not infer checkpoint direction from the metric name"
        )
    mode = raw_mode.strip().lower()
    if mode not in _SELECTION_MODES:
        raise ValueError(
            f"unsupported trainer.monitor_mode {raw_mode!r}; expected min or max"
        )
    return monitor.strip(), mode


@register_trainer("Default_trainer")
def trainer(args_e, args_t, args_d, path):
    """Build one Lightning Trainer from the explicit trainer configuration."""

    num_epochs = resolve_epoch_contract(args_t)
    if not hasattr(args_t, "pruning"):
        setattr(args_t, "pruning", 0.0)

    accelerator, devices = resolve_device_request(args_t)
    callback_list = call_backs(args_t, path)
    log_list = [CSVLogger(path, name="logs")]
    use_wandb = getattr(args_e, "wandb", False)
    use_swanlab = getattr(args_e, "swanlab", False)

    if use_wandb:
        wandb_logger = WandbLogger(
            project=args_e.project,
            offline=not is_main_process,
        )
        log_list.append(wandb_logger)

    if use_swanlab:
        if SwanLabLogger is None:
            raise RuntimeError(
                "environment.swanlab=true requires the optional 'swanlab' package. "
                "Install it or disable SwanLab logging."
            )
        swanlab_logger = SwanLabLogger(project=args_e.project)
        log_list.append(swanlab_logger)

    if not getattr(args_t, "log_every_n_steps", None):
        args_t.log_every_n_steps = 50

    return pl.Trainer(
        callbacks=callback_list,
        accelerator=accelerator,
        max_epochs=num_epochs,
        devices=devices,
        logger=log_list,
        log_every_n_steps=args_t.log_every_n_steps,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        deterministic=getattr(args_t, "deterministic", None),
    )


def call_backs(args, path):
    """Build checkpoint and stopping callbacks from one selection contract."""

    monitor, mode = resolve_selection_contract(args)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        # Do not embed a hard-coded metric such as val_loss in the filename.  The
        # configured monitor may be any logged scalar and the callback itself owns the
        # selected score.
        filename="model-{epoch:02d}-{step}",
        save_top_k=getattr(args, "save_top_k", 1),
        mode=mode,
        dirpath=path,
    )

    callback_list = [checkpoint_callback]

    if getattr(args, "pruning", 0.0):
        prune_callback = Prune_callback(args)
        callback_list.append(prune_callback)

    if getattr(args, "early_stopping", False):
        callback_list.append(
            create_early_stopping_callback(
                args,
                monitor=monitor,
                mode=mode,
            )
        )

    return callback_list


def Prune_callback(args):
    """根据训练配置，返回模型剪枝回调。"""

    def compute_amount(epoch):
        if epoch == args.num_epochs // 4:
            return args.pruning[0]
        if epoch == args.num_epochs // 2:
            return args.pruning[1]
        if 3 * args.num_epochs // 4 < epoch:
            return args.pruning[2]
        return None

    if isinstance(args.pruning, (int, float)):
        return ModelPruning(
            "l1_unstructured",
            parameter_names=["weight"],
            amount=args.pruning,
        )
    if isinstance(args.pruning, list):
        return ModelPruning(
            "l1_unstructured",
            parameter_names=["weight"],
            amount=compute_amount,
        )
    return None


def create_early_stopping_callback(
    args,
    *,
    monitor: str | None = None,
    mode: str | None = None,
):
    """Build EarlyStopping with the same explicit selection pair as checkpointing."""

    if monitor is None or mode is None:
        monitor, mode = resolve_selection_contract(args)
    return EarlyStopping(
        monitor=monitor,
        min_delta=float(getattr(args, "min_delta", 0.0)),
        patience=getattr(args, "patience", 10),
        verbose=True,
        mode=mode,
        check_finite=True,
    )


__all__ = [
    "call_backs",
    "create_early_stopping_callback",
    "resolve_epoch_contract",
    "resolve_selection_contract",
    "trainer",
]

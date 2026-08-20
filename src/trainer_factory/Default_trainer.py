import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.tensorboard.writer import SummaryWriter

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


@register_trainer("Default_trainer")
def trainer(args_e, args_t, args_d, path):
    """
    设置训练器的配置，包括日志记录、回调函数和数据加载器等。

    参数:
    - args_t: 包含训练配置的对象
    - args_d: 包含数据配置的对象
    - path: 存储日志、检查点的路径

    返回:
    - trainer: 训练器对象
    """
    # 为兼容旧配置，填充 num_epochs / pruning 的合理默认值
    if not hasattr(args_t, "num_epochs"):
        setattr(args_t, "num_epochs", getattr(args_t, "max_epochs", 1))
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

    trainer = pl.Trainer(
        callbacks=callback_list,
        accelerator=accelerator,
        max_epochs=args_t.num_epochs,
        devices=devices,
        logger=log_list,
        log_every_n_steps=args_t.log_every_n_steps,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        deterministic=getattr(args_t, "deterministic", None),
    )
    return trainer


def call_backs(args, path):
    """Build only callbacks that participate in training or checkpoint selection."""

    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=getattr(args, "save_top_k", 1),
        mode="min",
        dirpath=path,
    )

    callback_list = [checkpoint_callback]

    if getattr(args, "pruning", 0.0):
        prune_callback = Prune_callback(args)
        callback_list.append(prune_callback)

    if getattr(args, "early_stopping", False):
        early_stopping = create_early_stopping_callback(args)
        callback_list.append(early_stopping)

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
        prune_callback = ModelPruning(
            "l1_unstructured",
            parameter_names=["weight"],
            amount=args.pruning,
        )
    elif isinstance(args.pruning, list):
        prune_callback = ModelPruning(
            "l1_unstructured",
            parameter_names=["weight"],
            amount=compute_amount,
        )
    else:
        prune_callback = None
    return prune_callback


def create_early_stopping_callback(args):
    """创建并返回早期停止回调。"""
    return EarlyStopping(
        monitor=args.monitor,
        min_delta=float(getattr(args, "min_delta", 0.0)),
        patience=getattr(args, "patience", 10),
        verbose=True,
        mode="min",
        check_finite=True,
    )

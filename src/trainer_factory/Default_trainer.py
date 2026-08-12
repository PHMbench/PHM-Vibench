import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.tensorboard.writer import SummaryWriter

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:  # Optional experiment service; required only when enabled.
    SwanLabLogger = None

from src.trainer_factory import register_trainer
from src.trainer_factory.extensions import ManifestWriterCallback
from src.trainer_factory.p05_pilot_timing import build_p05_pilot_timing_callback
from src.trainer_factory.p05_runtime import prepare_p05_runtime

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
    # Fail closed before callbacks, loggers, or Trainer construction in the
    # explicitly enabled P05 evidence mode. Legacy configs return None here.
    runtime_contract = prepare_p05_runtime(args_t)

    # 为兼容旧配置，填充 num_epochs / gpus / pruning 的合理默认值
    if not hasattr(args_t, "num_epochs"):
        setattr(args_t, "num_epochs", getattr(args_t, "max_epochs", 1))
    if not hasattr(args_t, "gpus"):
        setattr(args_t, "gpus", getattr(args_t, "devices", 1))
    if not hasattr(args_t, "pruning"):
        setattr(args_t, "pruning", 0.0)

    # 获取回调列表
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

    # 如果不存在log_every_n_steps，使用默认值50 # TODO @liq22
    if not getattr(args_t, "log_every_n_steps", None):
        args_t.log_every_n_steps = 50

    if runtime_contract is None:
        runtime_kwargs = {
            "accelerator": "cpu" if args_t.device == "cpu" else "auto",
            "devices": args_t.gpus,
            "strategy": "ddp_find_unused_parameters_true" if args_t.gpus > 1 else "auto",
        }
        if hasattr(args_t, "deterministic"):
            runtime_kwargs["deterministic"] = args_t.deterministic
    else:
        runtime_kwargs = dict(runtime_contract.trainer_kwargs)

    # 初始化训练器
    trainer_instance = pl.Trainer(
        callbacks=callback_list,
        max_epochs=args_t.num_epochs,
        logger=log_list,
        log_every_n_steps=args_t.log_every_n_steps,
        **runtime_kwargs,
    )
    if runtime_contract is not None:
        setattr(
            trainer_instance,
            "p05_runtime_identity",
            dict(runtime_contract.runtime_identity),
        )
    return trainer_instance


def call_backs(args, path):
    """
    配置训练时所需的回调函数，包括检查点保存、模型修剪、早期停止等。

    参数:
    - args: 包含训练配置的对象
    - path: 存储检查点的路径

    返回:
    - callback_list: 配置好的回调列表
    """
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        filename="model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=getattr(args, "save_top_k", 1),
        mode="min",
        dirpath=path,
    )

    callback_list = [checkpoint_callback]

    pilot_timing_callback = build_p05_pilot_timing_callback(args, path)
    if pilot_timing_callback is not None:
        callback_list.append(pilot_timing_callback)

    # UXFD merge: always write an auditable manifest (safe no-op if not main process).
    try:
        extensions = getattr(args, "extensions", None)
        report_cfg = getattr(extensions, "report", None) if extensions is not None else None
        report_enable = getattr(report_cfg, "enable", True) if report_cfg is not None else True
        manifest_enable = getattr(report_cfg, "manifest", True) if report_cfg is not None else True
        enabled = bool(report_enable) and bool(manifest_enable)
    except Exception:
        enabled = True

    callback_list.append(
        ManifestWriterCallback(
            run_dir=path,
            paper_id=str(getattr(args, "paper_id", "") or ""),
            preset_version=str(getattr(args, "preset_version", "") or ""),
            run_id=str(getattr(args, "logger_name", "") or ""),
            enabled=enabled,
            is_main_process=is_main_process,
        )
    )

    # 模型修剪回调（根据需求添加）
    if getattr(args, "pruning", 0.0):
        prune_callback = Prune_callback(args)
        callback_list.append(prune_callback)

    # 早期停止回调（若未配置 early_stopping，则默认为不启用）
    if getattr(args, "early_stopping", False):
        early_stopping = create_early_stopping_callback(args)
        callback_list.append(early_stopping)

    return callback_list


def Prune_callback(args):
    """
    根据训练配置，返回模型修剪回调函数。

    参数:
    - args: 包含训练配置的对象

    返回:
    - prune_callback: 配置好的修剪回调（如果有）
    """

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

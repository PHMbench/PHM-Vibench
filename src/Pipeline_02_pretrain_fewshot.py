import argparse
import os
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_utils import load_config, transfer_namespace, ConfigWrapper
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from src.utils.training.run_contract import (
    build_training_stack,
    prepare_run_context,
    write_test_result_and_manifest,
)
from typing import Optional
from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator
from src.utils.config.pipeline_adapters import adapt_p02
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab


VALID_PIPELINE_MODES = {"single", "staged", "legacy"}


def _run_single_stage_from_cfg(cfg: ConfigWrapper):
    """在无需多阶段编排时，直接按单阶段配置运行一次训练+测试。

    用于：
    - 仅包含单阶段配置的 YAML（无 `stages` 字段），例如实验4–7；
    - 希望沿用 P02 入口但本质是单阶段 GFS/CDDG 训练的情况。
    """
    args_environment = transfer_namespace(getattr(cfg, 'environment', {}))
    args_data = transfer_namespace(getattr(cfg, 'data', {}))
    args_model = transfer_namespace(getattr(cfg, 'model', {}))
    args_task = transfer_namespace(getattr(cfg, 'task', {}))
    args_trainer = transfer_namespace(getattr(cfg, 'trainer', {}))

    # 处理多任务特殊情况
    if getattr(args_task, 'name', None) == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list

    # 设置环境变量（VBENCH_HOME等）
    env_section = getattr(cfg, 'environment', None)
    if env_section is not None:
        env_dict = env_section.__dict__ if hasattr(env_section, '__dict__') else env_section
        if isinstance(env_dict, dict):
            for key, value in env_dict.items():
                if str(key).isupper():
                    os.environ[str(key)] = str(value)

    run_ctx = prepare_run_context(cfg, args_environment, args_trainer, iteration=0, seed_offset=0)
    path = str(run_ctx.run_dir)
    current_seed = run_ctx.seed
    seed_everything(current_seed)
    init_lab(args_environment, cfg, run_ctx.logger_name)

    # 构建 data/model/task/trainer
    components = build_training_stack(
        args_environment=args_environment,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        run_dir=path,
        sidecar_config=cfg,
    )
    data_factory = components.data_factory
    task = components.task
    trainer = components.trainer

    # 运行训练与测试
    trainer.fit(task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))
    task = load_best_model_checkpoint(task, trainer)
    result = trainer.test(task, data_factory.get_dataloader('test'))
    write_test_result_and_manifest(
        run_dir=path,
        metrics=result[0],
        iteration=0,
        args_trainer=args_trainer,
        seed=current_seed,
        trainer=trainer,
        stage="test",
        manifest_required=True,
    )
    if hasattr(data_factory, "data") and hasattr(data_factory.data, "close"):
        data_factory.data.close()
    close_lab()
    return True


def run_stage(config_path, ckpt_path=None, iteration=0, local_config: Optional[str] = None):
    """Run a single training/testing stage given a config path."""
    configs = load_config(config_path, local_config)
    args_environment = transfer_namespace(configs.get('environment', {}))
    args_data = transfer_namespace(configs.get('data', {}))
    args_model = transfer_namespace(configs.get('model', {}))
    args_task = transfer_namespace(configs.get('task', {}))
    args_trainer = transfer_namespace(configs.get('trainer', {}))

    if args_task.name == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list

    if ckpt_path:
        args_model.weights_path = ckpt_path

    for key, value in configs['environment'].items():
        if key.isupper():
            os.environ[key] = str(value)

    run_ctx = prepare_run_context(configs, args_environment, args_trainer, iteration=iteration)
    path = str(run_ctx.run_dir)
    current_seed = run_ctx.seed
    seed_everything(current_seed)
    init_lab(args_environment, configs, run_ctx.logger_name)
    components = build_training_stack(
        args_environment=args_environment,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        run_dir=path,
        sidecar_config=configs,
    )
    data_factory = components.data_factory
    task = components.task
    trainer = components.trainer
    trainer.fit(task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))
    task = load_best_model_checkpoint(task, trainer)
    result = trainer.test(task, data_factory.get_dataloader('test'))
    write_test_result_and_manifest(
        run_dir=path,
        metrics=result[0],
        iteration=iteration,
        args_trainer=args_trainer,
        seed=current_seed,
        trainer=trainer,
        stage="test",
        manifest_required=True,
    )
    if hasattr(data_factory, "data") and hasattr(data_factory.data, "close"):
        data_factory.data.close()
    close_lab()
    return task, trainer


def run_pretraining_stage(config_path, local_config: Optional[str] = None):
    """Run the pretraining stage and return the checkpoint path."""
    # 加载配置以获取iterations设置
    configs = load_config(config_path, local_config)
    iterations = configs.get('environment', {}).get('iterations', 1)

    ckpt_dict = {}
    for it in range(iterations):
        task, trainer = run_stage(config_path, iteration=it, local_config=local_config)
        print(f"Pretraining stage iteration {it} completed.")
        ckpt_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_path = cb.best_model_path
                break
        ckpt_dict[it] = ckpt_path
    return ckpt_dict


def run_fewshot_stage(fs_config_path, ckpt_dict=None, local_config: Optional[str] = None):
    """Run the few-shot stage. Optionally load a pretrained checkpoint."""
    # 加载配置以获取iterations设置
    configs = load_config(fs_config_path, local_config)
    iterations = configs.get('environment', {}).get('iterations', 1)

    for it1, ckpt_path in ckpt_dict.items():
        for it2 in range(iterations):
            print(f"Running few-shot stage iteration {it1}-{it2} with checkpoint {ckpt_path}")
            if ckpt_path:
                run_stage(fs_config_path, ckpt_path, iteration=it1 * len(ckpt_dict) + it2, local_config=local_config)
            else:
                print(f"No checkpoint found for iteration {it1}, skipping few-shot stage.")
                run_stage(fs_config_path, iteration=it1 * len(ckpt_dict) + it2, local_config=local_config)
    return True

def _load_pipeline_yaml(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"P02 config must be a YAML mapping: {config_path}")
    return cfg_dict


def _get_pipeline_mode(cfg_dict: dict, config_path: str) -> str:
    mode = cfg_dict.get("pipeline_mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(
            "Pipeline_02_pretrain_fewshot requires explicit pipeline_mode: "
            "single | staged | legacy"
        )
    mode = mode.strip().lower()
    if mode not in VALID_PIPELINE_MODES:
        raise ValueError(f"Unsupported pipeline_mode in {config_path}: {mode}")
    return mode


def run_single_stage(args, cfg_dict: Optional[dict] = None):
    """Run P02 as a single stage. No legacy fallback is allowed."""
    if getattr(args, "fs_config_path", None):
        raise ValueError("pipeline_mode=single conflicts with fs_config_path")
    if cfg_dict is not None and "stages" in cfg_dict:
        raise ValueError("pipeline_mode=single conflicts with stages")

    overrides = None
    if hasattr(args, "override") and args.override:
        overrides = parse_overrides(args.override)
    cfg = load_config(args.config_path, overrides=overrides)
    _run_single_stage_from_cfg(cfg)
    print("[INFO] Single-stage pipeline via P02 completed.")
    return True


def run_staged(args, cfg_dict: dict):
    """Run P02 with a unified staged YAML."""
    stages = cfg_dict.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("pipeline_mode=staged requires a non-empty stages list")
    if getattr(args, "fs_config_path", None):
        raise ValueError("pipeline_mode=staged conflicts with fs_config_path")

    print(f"[INFO] 使用 unified 多阶段配置运行训练: {args.config_path}")
    cli_overrides = getattr(args, "override", None) or []
    orchestrator = TwoStageOrchestrator(cfg_dict, cli_overrides=cli_overrides)
    summary = orchestrator.run_complete()
    print("[INFO] Unified multi-stage pipeline completed.")
    return summary


def run_legacy_dual_yaml(args):
    """Run P02 legacy dual-YAML mode. Requires explicit fs_config_path."""
    if not getattr(args, "fs_config_path", None):
        raise ValueError("pipeline_mode=legacy requires fs_config_path")

    unified = adapt_p02(args.config_path, args.fs_config_path, getattr(args, "local_config", None))
    if hasattr(args, "override") and args.override:
        print(f"[INFO] 应用CLI override参数到 legacy 两阶段流程: {args.override}")
        overrides = parse_overrides(args.override)
        unified = apply_overrides_to_config(unified, overrides)
        print(f"[INFO] 已应用 {len(overrides)} 个override参数到 legacy 两阶段配置")

    orchestrator = TwoStageOrchestrator(unified)
    summary = orchestrator.run_complete()
    print("[INFO] Unified two-stage pipeline (legacy dual YAML) completed.")
    return summary


def pipeline(args):
    """Run P02 according to explicit pipeline_mode."""
    cfg_dict = _load_pipeline_yaml(args.config_path)
    mode = _get_pipeline_mode(cfg_dict, args.config_path)

    if mode == "single":
        return run_single_stage(args, cfg_dict=cfg_dict)
    if mode == "staged":
        return run_staged(args, cfg_dict=cfg_dict)
    if mode == "legacy":
        return run_legacy_dual_yaml(args)
    raise AssertionError(f"Unhandled pipeline_mode: {mode}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='unified or pretrain config path')
    parser.add_argument('--fs_config_path', type=str, default=None, help='[legacy] few-shot config path (dual YAML)')
    parser.add_argument('--local_config', type=str, default=None, help='machine-specific override YAML')
    parser.add_argument('--override', nargs='*', default=None, help='override key=value pairs')
    args = parser.parse_args()
    pipeline(args)

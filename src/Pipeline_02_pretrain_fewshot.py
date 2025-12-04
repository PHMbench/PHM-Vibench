import argparse
import os
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_utils import load_config, path_name, transfer_namespace, ConfigWrapper
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from typing import Optional
from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator
from src.utils.config.pipeline_adapters import adapt_p02
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer


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

    # 随机种子与日志初始化
    seed_everything(getattr(args_environment, 'seed', 42))
    # 单阶段默认使用 iteration=0
    path, name = path_name(cfg, 0)
    init_lab(args_environment, cfg, name)

    # 构建 data/model/task/trainer
    data_factory = build_data(args_data, args_task)
    model = build_model(args_model, metadata=data_factory.get_metadata())
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=data_factory.get_metadata(),
    )
    trainer = build_trainer(args_environment, args_trainer, args_data, path)

    # 运行训练与测试
    trainer.fit(task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))
    task = load_best_model_checkpoint(task, trainer)
    trainer.test(task, data_factory.get_dataloader('test'))
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

    path, name = path_name(configs, iteration)
    seed_everything(args_environment.seed)
    init_lab(args_environment, configs, name)
    data_factory = build_data(args_data, args_task)
    model = build_model(args_model, metadata=data_factory.get_metadata())
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=data_factory.get_metadata()
    )
    trainer = build_trainer(args_environment, args_trainer, args_data, path)
    trainer.fit(task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))
    task = load_best_model_checkpoint(task, trainer)
    result = trainer.test(task, data_factory.get_dataloader('test'))
    result_df = pd.DataFrame([result[0]])
    result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)
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

def pipeline(args):
    """Run multi-stage training for P02.

    优先支持“单 YAML + stages 列表”的统一配置范式：
      - 推荐入口：只提供 `--config_path experiment_X_unified.yaml`；
      - `fs_config_path` 仅用于兼容 legacy 双 YAML 配置（已移动到 configs/legacy_dual_yaml）。
    """
    try:
        # 新范式：只有 config_path（优先支持 unified YAML + stages 列表）
        if not getattr(args, 'fs_config_path', None):
            import yaml

            with open(args.config_path, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}

            # 情况A：包含 stages → 走多阶段 Orchestrator
            if 'stages' in cfg_dict:
                print(f"[INFO] 使用 unified 多阶段配置运行训练: {args.config_path}")
                cli_overrides = getattr(args, 'override', None) or []
                orchestrator = TwoStageOrchestrator(cfg_dict, cli_overrides=cli_overrides)
                summary = orchestrator.run_complete()
                print("[INFO] Unified multi-stage pipeline (single YAML) completed.")
                return summary

            # 情况B：不含 stages → 视为单阶段配置，直接运行一次训练/测试
            print(f"[INFO] 检测到单阶段配置（无 stages），按单阶段模式运行: {args.config_path}")
            # 使用通用 load_config + overrides 构造最终 ConfigWrapper
            overrides = None
            if hasattr(args, 'override') and args.override:
                overrides = parse_overrides(args.override)
            cfg = load_config(args.config_path, overrides=overrides)
            _run_single_stage_from_cfg(cfg)
            print("[INFO] Single-stage pipeline via P02 completed.")
            return True

        # 情况C 兼容路径：config_path + fs_config_path 双 YAML（legacy）
        unified = adapt_p02(args.config_path, args.fs_config_path, getattr(args, 'local_config', None))

        # 应用CLI override参数（最高优先级）——旧路径仍使用全局 override 机制
        if hasattr(args, 'override') and args.override:
            print(f"[INFO] 应用CLI override参数到两阶段流程: {args.override}")
            overrides = parse_overrides(args.override)
            unified = apply_overrides_to_config(unified, overrides)
            print(f"[INFO] 已应用 {len(overrides)} 个override参数到两阶段配置")

        orchestrator = TwoStageOrchestrator(unified)
        summary = orchestrator.run_complete()
        print("[INFO] Unified two-stage pipeline (legacy dual YAML) completed.")
        return summary
    except Exception as e:
        print(f"[WARN] Unified orchestrator fallback due to: {e}")
        # fallback 仅支持 legacy 双 YAML，unified 情况请使用 debug 脚本或修复配置
        if getattr(args, 'fs_config_path', None):
            ckpt_dict = run_pretraining_stage(args.config_path, local_config=getattr(args, 'local_config', None))
            run_fewshot_stage(args.fs_config_path, ckpt_dict, local_config=getattr(args, 'local_config', None))
            return True
        raise





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='unified or pretrain config path')
    parser.add_argument('--fs_config_path', type=str, default=None, help='[legacy] few-shot config path (dual YAML)')
    parser.add_argument('--local_config', type=str, default=None, help='machine-specific override YAML')
    parser.add_argument('--override', nargs='*', default=None, help='override key=value pairs')
    args = parser.parse_args()
    pipeline(args)

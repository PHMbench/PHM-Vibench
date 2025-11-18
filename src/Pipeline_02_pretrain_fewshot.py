import argparse
import os
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_utils import load_config, path_name, transfer_namespace
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from typing import Optional
from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator
from src.utils.config.pipeline_adapters import adapt_p02
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer


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
    ckpt_dict = {}
    for it in range(os.environ.get('iterations', 1)):
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
    for it1, ckpt_path in ckpt_dict.items():
        for it2 in range(os.environ.get('iterations', 1)):
            print(f"Running few-shot stage iteration {it1}-{it2} with checkpoint {ckpt_path}")
            if ckpt_path:
                run_stage(fs_config_path, ckpt_path, iteration=it1 * len(ckpt_dict) + it2, local_config=local_config)
            else:
                print(f"No checkpoint found for iteration {it1}, skipping few-shot stage.")
                run_stage(fs_config_path, iteration=it1 * len(ckpt_dict) + it2, local_config=local_config)
    return True

def pipeline(args):
    """Run pretraining followed by a few-shot stage.

    Unified orchestrator path: adapt P02 configs and run two-stage flow.
    Fallback to legacy functions if orchestrator fails for any reason.
    """
    try:
        unified = adapt_p02(args.config_path, args.fs_config_path, getattr(args, 'local_config', None))

        # 应用CLI override参数（最高优先级）
        if hasattr(args, 'override') and args.override:
            print(f"[INFO] 应用CLI override参数到两阶段流程: {args.override}")
            overrides = parse_overrides(args.override)
            unified = apply_overrides_to_config(unified, overrides)
            print(f"[INFO] 已应用 {len(overrides)} 个override参数到两阶段配置")

        orchestrator = TwoStageOrchestrator(unified)
        summary = orchestrator.run_complete()
        print("[INFO] Unified two-stage pipeline completed.")
        return summary
    except Exception as e:
        print(f"[WARN] Unified orchestrator fallback due to: {e}")
        ckpt_dict = run_pretraining_stage(args.config_path, local_config=getattr(args, 'local_config', None))
        run_fewshot_stage(args.fs_config_path, ckpt_dict, local_config=getattr(args, 'local_config', None))
        return True





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='pretrain config path')
    parser.add_argument('--fs_config_path', type=str, required=True, help='few-shot config path')
    parser.add_argument('--local_config', type=str, default=None, help='machine-specific override YAML')
    args = parser.parse_args()
    pipeline(args)

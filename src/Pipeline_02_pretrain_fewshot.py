import argparse
import importlib
import os
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config_utils import load_config, path_name, transfer_namespace
from src.utils.utils import load_best_model_checkpoint
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer


def run_experiment(config_path, notes=""):
    configs = load_config(config_path)
    args_environment = transfer_namespace(configs.get('environment', {}))
    args_data = transfer_namespace(configs.get('data', {}))
    args_data.name = configs['data'].get('name', 'default')
    args_model = transfer_namespace(configs.get('model', {}).get('args', {}))
    args_model.name = configs['model'].get('name', 'default')
    args_task = transfer_namespace(configs.get('task', {}).get('args', {}))
    args_task.name = configs['task'].get('name', 'default')
    args_trainer = transfer_namespace(configs.get('trainer', {}).get('args', {}))
    args_trainer.name = configs['trainer'].get('name', 'default')

    for key, value in configs['environment'].items():
        if key.isupper():
            os.environ[key] = str(value)

    path, name = path_name(configs, 0)
    seed_everything(args_environment.seed)

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
    return task, trainer


def pipeline(args):
    # 1. pretraining stage
    pretask, trainer = run_experiment(args.config_path)
    ckpt_path = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_path = cb.best_model_path
            break

    # 2. few-shot stage
    fs_configs = load_config(args.fs_config_path)
    fs_args_environment = transfer_namespace(fs_configs.get('environment', {}))
    fs_args_data = transfer_namespace(fs_configs.get('data', {}))
    fs_args_data.name = fs_configs['data'].get('name', 'default')
    fs_args_model = transfer_namespace(fs_configs.get('model', {}).get('args', {}))
    fs_args_model.name = fs_configs['model'].get('name', 'default')
    if ckpt_path:
        fs_args_model.weights_path = ckpt_path
    fs_args_task = transfer_namespace(fs_configs.get('task', {}).get('args', {}))
    fs_args_task.name = fs_configs['task'].get('name', 'default')
    fs_args_trainer = transfer_namespace(fs_configs.get('trainer', {}).get('args', {}))
    fs_args_trainer.name = fs_configs['trainer'].get('name', 'default')

    for key, value in fs_configs['environment'].items():
        if key.isupper():
            os.environ[key] = str(value)

    path, name = path_name(fs_configs, 0)
    seed_everything(fs_args_environment.seed)

    data_factory = build_data(fs_args_data, fs_args_task)
    model = build_model(fs_args_model, metadata=data_factory.get_metadata())
    task = build_task(
        args_task=fs_args_task,
        network=model,
        args_data=fs_args_data,
        args_model=fs_args_model,
        args_trainer=fs_args_trainer,
        args_environment=fs_args_environment,
        metadata=data_factory.get_metadata()
    )
    trainer = build_trainer(fs_args_environment, fs_args_trainer, fs_args_data, path)
    trainer.fit(task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))
    trainer.test(task, data_factory.get_dataloader('test'))

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='pretrain config path')
    parser.add_argument('--fs_config_path', type=str, required=True, help='few-shot config path')
    args = parser.parse_args()
    pipeline(args)

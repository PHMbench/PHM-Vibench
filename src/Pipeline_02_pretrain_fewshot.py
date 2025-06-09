import argparse
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


def run_stage(config_path, ckpt_path=None):
    """Run a single training/testing stage given a config path."""
    configs = load_config(config_path)
    args_environment = transfer_namespace(configs.get('environment', {}))
    args_data = transfer_namespace(configs.get('data', {}))
    args_model = transfer_namespace(configs.get('model', {}))
    args_task = transfer_namespace(configs.get('task', {}))
    args_trainer = transfer_namespace(configs.get('trainer', {}))

    if ckpt_path:
        args_model.weights_path = ckpt_path

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


def run_pretraining_stage(config_path):
    """Run the pretraining stage and return the checkpoint path."""
    task, trainer = run_stage(config_path)
    ckpt_path = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_path = cb.best_model_path
            break
    return ckpt_path


def run_fewshot_stage(fs_config_path, ckpt_path=None):
    """Run the few-shot stage. Optionally load a pretrained checkpoint."""
    run_stage(fs_config_path, ckpt_path)
    return True

def pipeline(args):
    """Run pretraining followed by a few-shot stage."""
    ckpt_path = run_pretraining_stage(args.config_path)
    run_fewshot_stage(args.fs_config_path, ckpt_path)
    return True





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='pretrain config path')
    parser.add_argument('--fs_config_path', type=str, required=True, help='few-shot config path')
    args = parser.parse_args()
    pipeline(args)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
try:
    import wandb
except ImportError:
    print("[WARNING] wandb 未安装")
    wandb = None
try:
    import swanlab
    from swanlab.plugin.notification import LarkCallback
    from swanlab.plugin.notification import SlackCallback
except ImportError:
    print("[WARNING] swanlab 未安装")
    swanlab = None
import numpy as np


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False) -> bool:
    """Load pretrained weights via the shared loader (backward compatibility helper)."""
    from src.utils.pipeline_config.base_utils import load_pretrained_weights as _load_pretrained_weights

    return _load_pretrained_weights(model, checkpoint_path, strict)


def load_best_model_checkpoint(model: LightningModule, trainer: Trainer) -> LightningModule:
    """Load the best checkpoint produced by ``trainer`` into ``model``.

    Testing a trained run without its selected checkpoint changes the experiment
    semantics. Missing, unreadable, or incompatible checkpoints therefore fail
    immediately with the checkpoint path and original cause.
    """
    model_checkpoint = next(
        (
            callback
            for callback in trainer.callbacks
            if isinstance(callback, ModelCheckpoint)
        ),
        None,
    )
    if model_checkpoint is None:
        raise ValueError(
            "ModelCheckpoint callback not found in trainer.callbacks. "
            "Enable checkpointing before requesting best-model evaluation."
        )

    best_model_path = os.fspath(model_checkpoint.best_model_path or "")
    if not best_model_path:
        raise RuntimeError(
            "Training did not produce a best checkpoint. Check the monitored "
            "metric, validation loop, and ModelCheckpoint configuration."
        )
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(
            f"Best checkpoint does not exist or is not a file: {best_model_path}"
        )

    try:
        checkpoint = torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("state_dict"), dict
        ):
            raise TypeError(
                "Expected a Lightning checkpoint containing a mapping at "
                "'state_dict'."
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except Exception as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise RuntimeError(
            f"Failed to restore best checkpoint '{best_model_path}': {exc}"
        ) from exc

    return model


def init_lab(args_environment, cli_args, experiment_name):
    """
    Initializes wandb and swanlab loggers based on environment configuration.

    Args:
        args_environment: Namespace containing environment configurations (e.g., wandb, swanlab flags, project name, notes).
        cli_args: Namespace containing command-line arguments (e.g., notes).
        experiment_name: The name for the current experiment run.
    """
    use_wandb = getattr(args_environment, 'wandb', False)
    use_swanlab = getattr(args_environment, 'swanlab', False)

    # Initialize WandB
    if wandb and wandb.run is None: # Check if wandb module is available and not already initialized
        if use_wandb:
            project_name = getattr(args_environment, 'project', 'vbench')
            notes = f'Task Notes:{getattr(cli_args, "notes", "")}\nConfig Notes:{getattr(args_environment, "notes", "")}'
            wandb.init(project=project_name,
                        name=experiment_name,
                        notes=notes.strip())
            print(f"[INFO] WandB initialized for project '{project_name}', experiment '{experiment_name}'.")
        else:
            wandb.init(mode='disabled')
            print("[INFO] WandB disabled by configuration.")
    elif use_wandb and wandb is None:
        print("[WARNING] WandB is configured to be used, but the 'wandb' library is not installed.")


    # Initialize SwanLab
    if swanlab and swanlab.run is None: # Check if swanlab module is available and not already initialized
        if use_swanlab:
            project_name = getattr(args_environment, 'project', 'vbench')
            notes = f'N1:{getattr(cli_args, "notes", "")}\n_N2:{getattr(args_environment, "notes", "")}'
            swanlab.init(
                workspace = getattr(args_environment, 'workspace', 'PHMbench'), # SwanLab uses 'workspace'
                project=project_name, # Assuming swanlab uses 'project' similar to wandb
                experiment_name= notes, # experiment_name,
                description=notes.strip() # Swanlab uses 'description' for notes
                # logdir= # Optional: specify log directory if needed
            )
            print(f"[INFO] SwanLab initialized for project '{project_name}', experiment '{experiment_name}'.")
        else:
            swanlab.init(mode='disabled')
            print("[INFO] SwanLab disabled by configuration.")
    elif use_swanlab and swanlab is None:
        print("[WARNING] SwanLab is configured to be used, but the 'swanlab' library is not installed.")

def close_lab():
    """
    Closes the WandB and SwanLab loggers if they are initialized.
    """
    if wandb and wandb.run:
        wandb.finish()
        print("[INFO] WandB logger closed.")
    if swanlab and swanlab.run:
        try:
            swanlab.finish()
        except Exception as e:
            print(f"[INFO] SwanLab is not used: {e}")
        print("[INFO] SwanLab logger closed.")

def get_num_classes(metadata, dataset_id=None):
    """
    获取数据集类别数。

    Args:
        metadata: 元数据对象
        dataset_id: 可选，指定数据集ID时返回该数据集的类别数(int)，否则返回所有数据集的映射(dict)

    Returns:
        int: 当指定dataset_id时，返回该数据集的类别数
        dict: 当未指定dataset_id时，返回{dataset_id: num_classes}映射

    Raises:
        ValueError: 当指定的dataset_id不存在时
    """
    df = metadata.df if hasattr(metadata, 'df') else metadata

    if dataset_id is not None:
        # 返回特定数据集的类别数(int)
        dataset_data = df[df['Dataset_id'] == dataset_id]
        if len(dataset_data) == 0:
            raise ValueError(f"Dataset_id {dataset_id} not found in metadata")
        return int(max(dataset_data['Label']) + 1)
    else:
        # 返回所有数据集的类别数映射(dict) - 保持原有格式
        num_classes = {}
        for key in np.unique(df['Dataset_id']):
            num = max(df[df['Dataset_id'] == key]['Label']) + 1
            num_classes[str(key)] = int(num)  # 保持原有的整型key
        return num_classes


def get_num_channels(metadata, dataset_id=None):
    """
    获取数据集通道数。

    Args:
        metadata: 元数据对象
        dataset_id: 可选，指定数据集ID时返回该数据集的通道数(int)，否则返回所有数据集的映射(dict)

    Returns:
        int: 当指定dataset_id时，返回该数据集的通道数
        dict: 当未指定dataset_id时，返回{dataset_id: num_channels}映射

    Raises:
        ValueError: 当指定的dataset_id不存在时
    """
    df = metadata.df if hasattr(metadata, 'df') else metadata

    if dataset_id is not None:
        # 返回特定数据集的通道数(int)
        dataset_data = df[df['Dataset_id'] == dataset_id]
        if len(dataset_data) == 0:
            raise ValueError(f"Dataset_id {dataset_id} not found in metadata")
        return int(max(dataset_data['Channel']))
    else:
        # 返回所有数据集的通道数映射(dict)
        num_channels = {}
        for key in np.unique(df['Dataset_id']):
            num_channels[key] = int(max(df[df['Dataset_id'] == key]['Channel']))
        return num_channels

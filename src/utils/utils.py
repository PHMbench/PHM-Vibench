from __future__ import annotations

from src.utils.checkpoint_io import (resolve_trusted_checkpoint_path,
                                     safe_torch_load)

try:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    LightningModule = Trainer = object
    ModelCheckpoint = None

try: 
    import wandb
except ImportError:
    print("[WARNING] wandb 未安装")
    wandb = None
try:
    import swanlab
    from swanlab.plugin.notification import LarkCallback, SlackCallback
except ImportError:
    print("[WARNING] swanlab 未安装")
    swanlab = None
import numpy as np


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False) -> bool:
    """Load pretrained weights via the shared loader (backward compatibility helper)."""
    from src.utils.pipeline_config.base_utils import \
        load_pretrained_weights as _load_pretrained_weights

    return _load_pretrained_weights(model, checkpoint_path, strict)


def load_best_model_checkpoint(model: LightningModule, trainer: Trainer) -> LightningModule:
    """
    加载训练过程中保存的最佳模型检查点。

    参数:
    - model: 要加载检查点权重的模型实例。
    - trainer: 用于训练模型的训练器实例。

    返回:
    - 加载了最佳检查点权重的模型实例。
    """
    if ModelCheckpoint is None:
        raise ImportError("pytorch_lightning is required to load best model checkpoints")

    # 从trainer的callbacks中找到ModelCheckpoint实例，并获取best_model_path
    model_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            model_checkpoint = callback
            break

    if model_checkpoint is None:
        raise ValueError("ModelCheckpoint callback not found in trainer's callbacks.")

    best_model_path = model_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")

    # 确保最佳模型路径不是空的
    if not best_model_path:
        print("No best model path found. Please check if the training process saved checkpoints.")
    else:
    # 加载最佳检查点
    # pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m. 
    # 	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
    # 	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
    # 	WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default. Please use `torch.serialization.add_safe_globals([scalar])` or the `torch.serialization.safe_globals([scalar])` context manager to allowlist this global if you trust this class/function.
        state_dict = safe_torch_load(best_model_path, map_location="cpu")
        model.load_state_dict(state_dict['state_dict'])
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


from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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


def load_best_model_checkpoint(model: LightningModule, trainer: Trainer) -> LightningModule:
    """
    加载训练过程中保存的最佳模型检查点。

    参数:
    - model: 要加载检查点权重的模型实例。
    - trainer: 用于训练模型的训练器实例。

    返回:
    - 加载了最佳检查点权重的模型实例。
    """
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
        state_dict = torch.load(best_model_path)
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
    if wandb: # Check if wandb module is available
        if use_wandb:
            project_name = getattr(args_environment, 'project', 'vbench')
            notes = f'CLI Notes:{cli_args.notes}\nConfig Notes:{getattr(args_environment, "notes", "")}'
            wandb.init(project=project_name,
                        name=experiment_name,
                        notes=notes.strip())
            print(f"[INFO] WandB initialized for project '{project_name}', experiment '{experiment_name}'.")
        else:
            wandb.init(mode='disabled')
            print("[INFO] WandB disabled by configuration.")
    elif use_wandb:
        print("[WARNING] WandB is configured to be used, but the 'wandb' library is not installed.")


    # Initialize SwanLab
    if swanlab: # Check if swanlab module is available
        if use_swanlab:
            project_name = getattr(args_environment, 'project', 'vbench')
            notes = f'CLI Notes:{cli_args.notes}\nConfig Notes:{getattr(args_environment, "notes", "")}'
            swanlab.init(
                project=project_name, # Assuming swanlab uses 'project' similar to wandb
                experiment_name=experiment_name,
                description=notes.strip() # Swanlab uses 'description' for notes
                # logdir= # Optional: specify log directory if needed
            )
            print(f"[INFO] SwanLab initialized for project '{project_name}', experiment '{experiment_name}'.")
        else:
            swanlab.init(mode='disabled')
            print("[INFO] SwanLab disabled by configuration.")
    elif use_swanlab:
        print("[WARNING] SwanLab is configured to be used, but the 'swanlab' library is not installed.")

def close_lab():
    """
    Closes the WandB and SwanLab loggers if they are initialized.
    """
    if wandb and wandb.run:
        wandb.finish()
        print("[INFO] WandB logger closed.")
    if swanlab and swanlab.run:
        swanlab.finish()
        print("[INFO] SwanLab logger closed.")
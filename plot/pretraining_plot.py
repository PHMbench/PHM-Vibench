import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils.config_utils import load_config, transfer_namespace
from src.data_factory import build_data
from src.model_factory import build_model
from plot.A1_plot_config import configure_matplotlib

def add_mask(signal, forecast_part, mask_ratio):
    """
    Applies forecasting and random masking to a signal tensor.

    Args:
        signal (torch.Tensor): The input signal tensor of shape (B, L, C).
        forecast_part (float): The fraction of the sequence to be masked for forecasting.
        mask_ratio (float): The ratio of random masking for imputation.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x_in (torch.Tensor): The masked signal tensor.
            - total_mask (torch.Tensor): The boolean mask tensor.
    """
    B, L, C = signal.shape
    device = signal.device

    L_f = int(L * forecast_part)
    L_o = L - L_f

    # Prediction mask for forecasting
    mask_pred = torch.zeros(L, dtype=torch.bool, device=device)
    mask_pred[L_o:] = True

    # Random mask for imputation
    mask_rand = (torch.rand((B, L_o, 1), device=device) < mask_ratio)
    mask_rand = torch.cat([mask_rand, torch.zeros(B, L_f, 1, device=device)], 1)
    mask_rand = mask_rand.bool().expand(-1, -1, C)

    # Combine masks
    mask_pred_expanded = mask_pred.unsqueeze(0).unsqueeze(2).expand(B, L, C)
    total_mask = mask_pred_expanded | mask_rand

    # Apply mask to signal
    x_in = signal.clone()
    x_in[total_mask] = 0.0
    
    return x_in, total_mask

def run_plot_stage(config_path, ckpt_path=None):
    """
    Loads a model and data based on a config file, performs a prediction task,
    and visualizes the results.
    """
    print(f"加载配置文件: {config_path}")
    if ckpt_path:
        print(f"加载预训练模型: {ckpt_path}")
    
    # 1. 配置 matplotlib
    configure_matplotlib(style='ieee', font_lang='en')

    # 2. 加载配置并设置
    configs = load_config(config_path)
    args_environment = transfer_namespace(configs.get('environment', {}))
    args_data = transfer_namespace(configs.get('data', {}))
    args_model = transfer_namespace(configs.get('model', {}))
    args_task = transfer_namespace(configs.get('task', {}))
    
    if args_task.name == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list

    if ckpt_path:
        args_model.weights_path = ckpt_path

    # 3. 获取数据和模型
    data_factory = build_data(args_data, args_task)
    metadata = data_factory.get_metadata()
    model = build_model(args_model, metadata)
    model.eval()

    dataloader = data_factory.get_dataloader('train')
    batch = next(iter(dataloader))
    signal = batch['x']
    file_id = batch.get('file_id', None)

    # 4. 执行预测任务并复现掩码逻辑
    x_in, total_mask = add_mask(signal, args_task.forecast_part, args_task.mask_ratio)

    with torch.no_grad():
        x_hat = model(x_in, file_id, task_id='prediction')

    # 5. 绘图
    sample_idx = 0
    channel_to_plot = 0
    
    signal_to_plot = signal[sample_idx, :, channel_to_plot].cpu().numpy()
    masked_to_plot = x_in[sample_idx, :, channel_to_plot].cpu().numpy()
    predicted_to_plot = x_hat[sample_idx, :, channel_to_plot].cpu().numpy()
    mask_to_plot = total_mask[sample_idx, :, channel_to_plot].cpu().numpy()
    timesteps = np.arange(signal.shape[1])

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Prediction Visualization (Sample {sample_idx}, Channel {channel_to_plot})')

    # 子图1: Ground Truth
    axes[0].plot(timesteps, signal_to_plot, label='Ground Truth')
    axes[0].fill_between(timesteps, signal_to_plot.min(), signal_to_plot.max(), where=mask_to_plot, color='gray', alpha=0.2, label='Masked Region')
    axes[0].set_title('Subfig 1: Ground Truth Signal')
    axes[0].legend()
    axes[0].grid(True)

    # 子图2: Masked Input
    axes[1].plot(timesteps, masked_to_plot, label='Masked Input')
    axes[1].fill_between(timesteps, signal_to_plot.min(), signal_to_plot.max(), where=mask_to_plot, color='gray', alpha=0.5, label='Masked Region')
    axes[1].set_title('Subfig 2: Masked Input Signal')
    axes[1].legend()
    axes[1].grid(True)

    # 子图3: Predicted Signal
    axes[2].plot(timesteps, predicted_to_plot, label='Predicted Signal', color='orange')
    axes[2].fill_between(timesteps, signal_to_plot.min(), signal_to_plot.max(), where=mask_to_plot, color='gray', alpha=0.5, label='Masked Region')
    axes[2].set_title('Subfig 3: Predicted Signal')
    axes[2].legend()
    axes[2].grid(True)

    # 子图4: Absolute Error
    abs_error = np.abs(signal_to_plot - predicted_to_plot)
    axes[3].plot(timesteps, abs_error, label='Absolute Error', color='red')
    axes[3].fill_between(timesteps, 0, abs_error.max(), where=mask_to_plot, color='gray', alpha=0.5, label='Masked Region')
    axes[3].set_title('Subfig 4: Absolute Error (Ground Truth vs. Prediction)')
    axes[3].set_xlabel('Time Step')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plot_dir = os.path.join(project_root, 'plot', 'output')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, 'pretraining_prediction_visualization.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_pipeline(args):
    """Run the plot pipeline."""
    run_plot_stage(args.config_path, args.ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize pretraining prediction results.")
    parser.add_argument('--config_path', type=str, default='script/LQ1/Pretraining/Pretraining_C+P.yaml', help='Path to the pretraining configuration file.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the pretrained model checkpoint.')
    args = parser.parse_args()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Resolve paths
    if not os.path.isabs(args.config_path):
        args.config_path = os.path.join(project_root, args.config_path)
    if args.ckpt_path and not os.path.isabs(args.ckpt_path):
        args.ckpt_path = os.path.join(project_root, args.ckpt_path)

    plot_pipeline(args)

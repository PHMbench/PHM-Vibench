"""Pretraining-prediction visualization pipeline.

Migrated from ``plot/pretraining_plot.py``. The only import changes vs. the
original: ``plot.A1_plot_config`` -> the local ``plot_factory.plot_config``,
and the hard-coded ``plot/output`` save dir -> ``results/plot_factory/output``
(falls back to env override ``PHM_VIBENCH_PLOT_DIR``).

Registered as ``P_01_pretraining_prediction`` (resolved lazily via
:func:`plot_factory.get_plotter`).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make the project root importable when this file is run directly as a script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from src.data_factory import build_data
from src.model_factory import build_model
from src.configs.config_utils import load_config, transfer_namespace
from src.utils.training.masking import add_mask

from .plot_config import configure_matplotlib


def _default_plot_dir() -> str:
    """Resolve the base directory for saved plots (env-overridable)."""
    return os.environ.get(
        "PHM_VIBENCH_PLOT_DIR",
        os.path.join(_PROJECT_ROOT, "results", "plot_factory", "output"),
    )


def setup_pipeline(config_path, ckpt_path=None):
    """Load configurations and set up the environment."""
    print(f"加载配置文件: {config_path}")
    if ckpt_path:
        print(f"加载预训练模型: {ckpt_path}")

    configure_matplotlib(style="no-latex", font_lang="en")  # ieee no-latex

    configs = load_config(config_path)
    args_data = transfer_namespace(configs.get("data", {}))
    args_model = transfer_namespace(configs.get("model", {}))
    args_task = transfer_namespace(configs.get("task", {}))

    if args_task.name == "Multitask":
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list

    if ckpt_path:
        args_model.weights_path = ckpt_path

    return args_data, args_model, args_task


def prepare_data_and_model(args_data, args_model, args_task):
    """Prepare dataset and model."""
    data_factory = build_data(args_data, args_task)
    metadata = data_factory.get_metadata()
    model = build_model(args_model, metadata)
    model.eval()
    dataset = data_factory.get_dataset("train")
    return dataset, model


def run_prediction(model, signal, file_id, args_task):
    """Run prediction on a batch of data (reproduces the masking logic)."""
    x_in, total_mask = add_mask(signal, args_task.forecast_part, args_task.mask_ratio)

    with torch.no_grad():
        x_hat = model(x_in, file_id, task_id="prediction")

    return signal, x_in, x_hat, total_mask


def plot_results(
    signal,
    masked_signal,
    predicted_signal,
    mask,
    save_path,
    sample_idx=0,
    channel_to_plot=0,
):
    """Visualize and save the prediction results."""
    L = signal.shape[1]
    signal_to_plot = signal[sample_idx, :, channel_to_plot].cpu().numpy()
    masked_to_plot = masked_signal[sample_idx, :, channel_to_plot].cpu().numpy()
    predicted_to_plot = predicted_signal[sample_idx, :, channel_to_plot].cpu().numpy()
    mask_to_plot = mask[sample_idx, :, channel_to_plot].cpu().numpy()
    timesteps = np.arange(L)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Prediction Visualization (Sample {sample_idx}, Channel {channel_to_plot})")

    axes[0].plot(timesteps, signal_to_plot, label="Ground Truth")
    axes[0].fill_between(
        timesteps, signal_to_plot.min(), signal_to_plot.max(),
        where=mask_to_plot, color="gray", alpha=0.2, label="Masked Region",
    )
    axes[0].set_title("Subfig 1: Ground Truth Signal")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(timesteps, masked_to_plot, label="Masked Input")
    axes[1].fill_between(
        timesteps, signal_to_plot.min(), signal_to_plot.max(),
        where=mask_to_plot, color="gray", alpha=0.5, label="Masked Region",
    )
    axes[1].set_title("Subfig 2: Masked Input Signal")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(timesteps, predicted_to_plot, label="Predicted Signal", color="orange")
    axes[2].fill_between(
        timesteps, signal_to_plot.min(), signal_to_plot.max(),
        where=mask_to_plot, color="gray", alpha=0.5, label="Masked Region",
    )
    axes[2].set_title("Subfig 3: Predicted Signal")
    axes[2].legend()
    axes[2].grid(True)

    abs_error = np.abs(signal_to_plot - predicted_to_plot)
    axes[3].plot(timesteps, abs_error, label="Absolute Error", color="red")
    axes[3].fill_between(
        timesteps, 0, abs_error.max(),
        where=mask_to_plot, color="gray", alpha=0.5, label="Masked Region",
    )
    axes[3].set_title("Subfig 4: Absolute Error (Ground Truth vs. Prediction)")
    axes[3].set_xlabel("Time Step")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf", dpi=300)
    print(f"Plot saved to {save_path}")


def plot_pipeline(args):
    """Run the plot pipeline."""
    args_data, args_model, args_task = setup_pipeline(args.config_path, args.ckpt_path)
    dataset, model = prepare_data_and_model(args_data, args_model, args_task)

    if args.file_ids:
        if isinstance(args.file_ids, str):
            args.file_ids = [args.file_ids]
        file_ids_to_plot = args.file_ids
    else:
        all_file_ids = list(dataset.dataset_dict.keys())
        file_ids_to_plot = all_file_ids[: min(5, len(all_file_ids))]

    print(f"Will generate plots for file_ids: {file_ids_to_plot}")

    plot_dir = _default_plot_dir()

    for file_id in file_ids_to_plot:
        try:
            key = int(file_id)
        except ValueError:
            key = file_id

        original_dataset = dataset.dataset_dict.get(key)
        if original_dataset is None or len(original_dataset) == 0:
            print(f"Warning: No data for file_id {file_id}. Skipping.")
            continue

        sample_dict = original_dataset[0]
        signal = sample_dict["x"]
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        original_signal, x_in, x_hat, total_mask = run_prediction(model, signal, key, args_task)

        save_path = os.path.join(plot_dir, f"pretraining_prediction_fid_{file_id}")
        print(f"Plotting for file_id: {file_id}")
        plot_results(original_signal, x_in, x_hat, total_mask, save_path)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize pretraining prediction results.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the pretraining configuration file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--file_ids",
        nargs="+",
        default="10",
        help="List of file_ids to plot.",
    )
    return parser


if __name__ == "__main__":
    _args = _build_argparser().parse_args()

    if not os.path.isabs(_args.config_path):
        _args.config_path = os.path.join(_PROJECT_ROOT, _args.config_path)
    if _args.ckpt_path and not os.path.isabs(_args.ckpt_path):
        _args.ckpt_path = os.path.join(_PROJECT_ROOT, _args.ckpt_path)

    plot_pipeline(_args)

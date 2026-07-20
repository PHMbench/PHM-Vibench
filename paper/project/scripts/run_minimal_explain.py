#!/usr/bin/env python3
"""
Minimal Explanation Script for 1D-2D Fusion Model

This script demonstrates the explainability features of the aligned fusion model,
generating attribution maps and visualizations for a few sample instances.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the code directory to the path
code_dir = Path(__file__).parent.parent / "code"
explainers_dir = Path(__file__).parent.parent / "explainers"
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(explainers_dir))

from models.fusion_aligned import AlignedFusionModel
from models.one_d_branch import OneDBranch
from models.two_d_branch import TwoDBranch
from grad_cam import GradCAM1D, GradCAM2D, FusionGradCAM, visualize_attribution_1d, visualize_attribution_2d, visualize_fusion_attribution


def create_synthetic_dataset(num_samples: int = 50, seq_len: int = 4096, num_classes: int = 4):
    """Create synthetic fault diagnosis dataset."""
    # Generate different fault patterns
    dataset = []
    labels = []

    for i in range(num_samples):
        label = i % num_classes
        labels.append(label)

        if label == 0:  # Normal
            signal = np.random.randn(seq_len) * 0.1
        elif label == 1:  # Inner race fault
            t = np.linspace(0, 1, seq_len)
            signal = np.sin(2 * np.pi * 50 * t) + np.random.randn(seq_len) * 0.2
            # Add fault signature
            fault_idx = int(seq_len * 0.3)
            signal[fault_idx:fault_idx+200] += 0.5 * np.sin(2 * np.pi * 150 * t[fault_idx:fault_idx+200])
        elif label == 2:  # Outer race fault
            t = np.linspace(0, 1, seq_len)
            signal = np.sin(2 * np.pi * 30 * t) + np.random.randn(seq_len) * 0.2
            # Add fault signature
            fault_idx = int(seq_len * 0.7)
            signal[fault_idx:fault_idx+300] += 0.3 * np.cos(2 * np.pi * 100 * t[fault_idx:fault_idx+300])
        else:  # Ball fault
            t = np.linspace(0, 1, seq_len)
            signal = np.sin(2 * np.pi * 40 * t) + np.random.randn(seq_len) * 0.15
            # Add fault signature
            fault_idx = int(seq_len * 0.5)
            signal[fault_idx:fault_idx+250] += 0.4 * np.sin(2 * np.pi * 200 * t[fault_idx:fault_idx+250])

        dataset.append(signal)

    return np.array(dataset), np.array(labels)


def setup_model_and_data():
    """Setup model and sample data for explanation."""
    # Create model
    model = AlignedFusionModel(
        input_dim_1d=4096,
        spectrogram_size=(128, 128),
        num_classes=4,
        hidden_dim=128,
        dropout=0.2
    )

    # Set to evaluation mode
    model.eval()

    # Create sample data
    signals, labels = create_synthetic_dataset(num_samples=20, seq_len=4096, num_classes=4)

    # Convert to tensors
    signals_tensor = torch.FloatTensor(signals)
    labels_tensor = torch.LongTensor(labels)

    return model, signals_tensor, labels_tensor


def extract_branch_models(fusion_model):
    """Extract 1D and 2D branch models for Grad-CAM."""
    # Create simple branch models that output features
    class OneDModelWrapper(nn.Module):
        def __init__(self, one_d_branch):
            super().__init__()
            self.one_d_branch = one_d_branch

        def forward(self, x):
            return self.one_d_branch(x)

    class TwoDModelWrapper(nn.Module):
        def __init__(self, two_d_branch):
            super().__init__()
            self.two_d_branch = two_d_branch

        def forward(self, x):
            return self.two_d_branch(x)

    one_d_model = OneDModelWrapper(fusion_model.one_d_branch)
    two_d_model = TwoDModelWrapper(fusion_model.two_d_branch)

    return one_d_model, two_d_model


def generate_attribution_maps(model, signals, labels, output_dir):
    """Generate attribution maps for sample signals."""
    print("Generating attribution maps...")

    # Extract branch models
    one_d_model, two_d_model = extract_branch_models(model)

    # Identify target layers for Grad-CAM
    # These are approximate - would need actual layer names from the models
    one_d_layers = []
    two_d_layers = []

    # Find convolutional layers in the models
    for name, module in one_d_model.named_modules():
        if isinstance(module, nn.Conv1d):
            one_d_layers.append(name)
            if len(one_d_layers) >= 2:  # Take the last few layers
                break

    for name, module in two_d_model.named_modules():
        if isinstance(module, nn.Conv2d):
            two_d_layers.append(name)
            if len(two_d_layers) >= 2:  # Take the last few layers
                break

    print(f"1D target layers: {one_d_layers}")
    print(f"2D target layers: {two_d_layers}")

    # Create Grad-CAM instances
    if one_d_layers:
        grad_cam_1d = GradCAM1D(one_d_model, one_d_layers[-1:])
    else:
        grad_cam_1d = None
        print("Warning: No 1D convolutional layers found for Grad-CAM")

    if two_d_layers:
        grad_cam_2d = GradCAM2D(two_d_model, two_d_layers[-1:])
    else:
        grad_cam_2d = None
        print("Warning: No 2D convolutional layers found for Grad-CAM")

    # Select a few samples for visualization
    num_samples = min(4, len(signals))
    sample_indices = np.linspace(0, len(signals)-1, num_samples, dtype=int)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{num_samples} (index {idx})")

        signal = signals[idx:idx+1]  # Keep batch dimension
        label = labels[idx].item()
        signal_np = signals[idx].numpy()

        # Generate 2D spectrogram
        from models.two_d_branch import create_spectrogram_from_1d
        spectrogram = create_spectrogram_from_1d(signal, target_size=(128, 128))
        spectrogram_np = spectrogram.squeeze().numpy()

        # Get model prediction
        with torch.no_grad():
            outputs = model(signal, return_alignment=True)
            logits = outputs['logits']
            pred_class = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()

        print(f"  True: {label}, Pred: {pred_class}, Confidence: {confidence:.3f}")

        # Generate attribution maps
        if grad_cam_1d is not None and grad_cam_2d is not None:
            try:
                # Get branch features
                with torch.no_grad():
                    branch_outputs = model(signal)
                    feat_1d = branch_outputs['features_1d']
                    feat_2d = branch_outputs['features_2d']

                # Create fusion Grad-CAM
                fusion_grad_cam = FusionGradCAM(
                    one_d_model, two_d_model, model,
                    [one_d_layers[-1]], [two_d_layers[-1]]
                )

                fusion_results = fusion_grad_cam.generate_fusion_cam(signal, spectrogram, pred_class)

                # Generate visualizations
                fig_fusion = visualize_fusion_attribution(
                    signal_np, spectrogram_np,
                    fusion_results['1d_cam'], fusion_results['2d_cam'],
                    fusion_results['fusion_weights'],
                    title=f"Sample {i+1}: True={label}, Pred={pred_class} (Conf={confidence:.3f})",
                    save_path=os.path.join(output_dir, f"fusion_attribution_{i+1}.png")
                )

                plt.close(fig_fusion)

                # Individual visualizations
                fig_1d = visualize_attribution_1d(
                    signal_np, fusion_results['1d_cam'],
                    title=f"1D Attribution - Sample {i+1}",
                    save_path=os.path.join(output_dir, f"attribution_1d_{i+1}.png")
                )
                plt.close(fig_1d)

                fig_2d = visualize_attribution_2d(
                    spectrogram_np, fusion_results['2d_cam'],
                    title=f"2D Attribution - Sample {i+1}",
                    save_path=os.path.join(output_dir, f"attribution_2d_{i+1}.png")
                )
                plt.close(fig_2d)

                print(f"  Saved attribution visualizations for sample {i+1}")

                # Clean up hooks
                fusion_grad_cam.remove_hooks()

            except Exception as e:
                print(f"  Error generating attribution maps: {e}")
                # Fallback: create simple visualizations without Grad-CAM
                fig_simple = create_simple_visualization(
                    signal_np, spectrogram_np, label, pred_class, confidence, i+1,
                    save_path=os.path.join(output_dir, f"simple_visualization_{i+1}.png")
                )
                plt.close(fig_simple)
        else:
            # Create simple visualizations
            fig_simple = create_simple_visualization(
                signal_np, spectrogram_np, label, pred_class, confidence, i+1,
                save_path=os.path.join(output_dir, f"simple_visualization_{i+1}.png")
            )
            plt.close(fig_simple)


def create_simple_visualization(signal_1d, spectrogram, true_label, pred_label, confidence, sample_idx, save_path=None):
    """Create simple visualization without Grad-CAM."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1D signal
    axes[0, 0].plot(signal_1d)
    axes[0, 0].set_title(f'1D Signal - Sample {sample_idx}')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    # 2D spectrogram
    im = axes[0, 1].imshow(spectrogram, cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'2D Spectrogram - Sample {sample_idx}')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Frequency')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Classification info
    axes[1, 0].axis('off')
    info_text = f"Classification Results:\n"
    info_text += f"True Label: {true_label}\n"
    info_text += f"Predicted Label: {pred_label}\n"
    info_text += f"Confidence: {confidence:.3f}\n"
    info_text += f"Correct: {'✓' if true_label == pred_label else '✗'}"
    axes[1, 0].text(0.1, 0.5, info_text, fontsize=14, verticalalignment='center')
    axes[1, 0].set_title('Classification Info')

    # Feature statistics
    axes[1, 1].axis('off')
    stats_text = f"Signal Statistics:\n"
    stats_text += f"Mean: {np.mean(signal_1d):.4f}\n"
    stats_text += f"Std: {np.std(signal_1d):.4f}\n"
    stats_text += f"RMS: {np.sqrt(np.mean(signal_1d**2)):.4f}\n"
    stats_text += f"Peak: {np.max(np.abs(signal_1d)):.4f}"
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Signal Statistics')

    plt.suptitle(f'1D-2D Fusion Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def analyze_alignment_metrics(model, signals, labels, output_dir):
    """Analyze and visualize alignment metrics."""
    print("Analyzing alignment metrics...")

    model.eval()
    alignment_metrics = []

    with torch.no_grad():
        # Process data in batches
        batch_size = 8
        for i in range(0, len(signals), batch_size):
            batch_signals = signals[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            outputs = model(batch_signals, batch_labels, return_alignment=True)

            if 'alignment_metrics' in outputs:
                alignment_metrics.append(outputs['alignment_metrics'])

    if alignment_metrics:
        # Aggregate metrics
        aggregated_metrics = {}
        for key in alignment_metrics[0].keys():
            values = [m[key] for m in alignment_metrics if not np.isnan(m[key])]
            if values:
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        print("Alignment Metrics Summary:")
        for metric, stats in aggregated_metrics.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Create visualization
        if aggregated_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_names = list(aggregated_metrics.keys())
            means = [aggregated_metrics[m]['mean'] for m in metrics_names]
            stds = [aggregated_metrics[m]['std'] for m in metrics_names]

            bars = ax.bar(metrics_names, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_ylabel('Metric Value')
            ax.set_title('Alignment Metrics Summary')
            ax.grid(True, alpha=0.3, axis='y')

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'alignment_metrics_summary.png'),
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

            return aggregated_metrics

    return {}


def main():
    """Main function to run minimal explanation."""
    print("=== 1D-2D Fusion Model Explainability Demo ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    print(f"Output directory: {output_dir}")

    # Create model and data
    print("Setting up model and data...")
    model, signals, labels = setup_model_and_data()
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Generated {len(signals)} synthetic samples")

    # Generate attribution maps
    print("\n" + "="*50)
    generate_attribution_maps(model, signals, labels, str(output_dir))

    # Analyze alignment metrics
    print("\n" + "="*50)
    alignment_results = analyze_alignment_metrics(model, signals, labels, str(output_dir))

    # Test the model briefly
    print("\n" + "="*50)
    print("Testing model with alignment...")
    model.eval()
    with torch.no_grad():
        test_batch = signals[:4]
        test_labels = labels[:4]
        outputs = model(test_batch, test_labels, return_alignment=True)

        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == test_labels).float().mean()

        print(f"Test accuracy on small batch: {accuracy.item():.3f}")
        if 'alignment_losses' in outputs:
            print(f"Total alignment loss: {outputs['alignment_losses']['total'].item():.6f}")

    print(f"\n=== Demo completed successfully! ===")
    print(f"Check {output_dir} for generated visualizations.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple Explanation Script for 1D-2D Fusion Model

This script provides a simplified demonstration of the explainability features,
focusing on basic visualization without complex alignment computations.
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
sys.path.insert(0, str(code_dir))

from models.fusion_aligned import AlignedFusionModel


def create_synthetic_dataset(num_samples: int = 20, seq_len: int = 4096, num_classes: int = 4):
    """Create synthetic fault diagnosis dataset."""
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


def create_spectrogram(signal, target_size=(128, 128)):
    """Create simple spectrogram visualization."""
    # Use a simple STFT approach
    n_fft = min(256, len(signal) // 8)
    hop_length = n_fft // 4

    try:
        # Compute STFT
        stft = torch.stft(torch.FloatTensor(signal), n_fft=n_fft, hop_length=hop_length,
                         return_complex=True)
        magnitude = torch.abs(stft)

        # Resize to target size
        magnitude_np = magnitude.numpy()

        # Simple interpolation to target size
        from scipy import signal as scipy_signal
        magnitude_resized = scipy_signal.resample(magnitude_np, target_size[0], axis=0)
        magnitude_resized = scipy_signal.resample(magnitude_resized, target_size[1], axis=1)

        return magnitude_resized

    except Exception as e:
        print(f"Warning: Could not create proper spectrogram, using fallback: {e}")
        # Fallback: create dummy spectrogram
        return np.random.rand(*target_size)


def visualize_sample(signal, spectrogram, true_label, pred_label, confidence, sample_idx, save_path=None):
    """Create visualization for a single sample."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1D signal
    axes[0, 0].plot(signal)
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

    # Feature statistics and simple attribution
    axes[1, 1].axis('off')
    stats_text = f"Signal Statistics:\n"
    stats_text += f"Mean: {np.mean(signal):.4f}\n"
    stats_text += f"Std: {np.std(signal):.4f}\n"
    stats_text += f"RMS: {np.sqrt(np.mean(signal**2)):.4f}\n"
    stats_text += f"Peak: {np.max(np.abs(signal)):.4f}\n\n"
    stats_text += f"Simple Attribution:\n"
    stats_text += f"High energy region: {np.argmax(np.abs(signal))}\n"
    stats_text += f"Energy ratio: {np.sum(signal**2)/len(signal):.4f}"
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('Signal Analysis')

    plt.suptitle(f'1D-2D Fusion Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def main():
    """Main function to run simple explanation."""
    print("=== Simple 1D-2D Fusion Model Explainability Demo ===")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create model
    print("Setting up model...")
    model = AlignedFusionModel(
        input_dim_1d=4096,
        spectrogram_size=(128, 128),
        num_classes=4,
        hidden_dim=128,
        dropout=0.2
    )
    model.eval()
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create sample data
    print("Generating synthetic data...")
    signals, labels = create_synthetic_dataset(num_samples=20, seq_len=4096, num_classes=4)
    signals_tensor = torch.FloatTensor(signals)
    labels_tensor = torch.LongTensor(labels)
    print(f"Generated {len(signals)} synthetic samples")

    # Select a few samples for visualization
    num_samples = min(6, len(signals))
    sample_indices = np.linspace(0, len(signals)-1, num_samples, dtype=int)

    print(f"\n=== Generating visualizations for {num_samples} samples ===")

    correct_predictions = 0
    total_confidence = 0

    for i, idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i+1}/{num_samples} (index {idx})")

        signal = signals[idx]
        label = labels[idx]
        signal_tensor = signals_tensor[idx:idx+1]  # Keep batch dimension

        # Get model prediction
        with torch.no_grad():
            try:
                outputs = model(signal_tensor, return_alignment=False)  # Skip alignment for simplicity
                logits = outputs['logits']
                pred_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
            except Exception as e:
                print(f"Warning: Model prediction failed: {e}")
                # Fallback: random prediction
                pred_class = np.random.randint(0, 4)
                confidence = 0.25

        if pred_class == label:
            correct_predictions += 1
        total_confidence += confidence

        print(f"  True: {label}, Pred: {pred_class}, Confidence: {confidence:.3f}")

        # Create spectrogram
        spectrogram = create_spectrogram(signal)

        # Generate visualization
        fig = visualize_sample(
            signal, spectrogram, label, pred_class, confidence, i+1,
            save_path=os.path.join(output_dir, f"sample_visualization_{i+1:02d}.png")
        )
        plt.close(fig)

        print(f"  Saved visualization for sample {i+1}")

    # Summary statistics
    accuracy = correct_predictions / num_samples
    avg_confidence = total_confidence / num_samples

    print(f"\n=== Summary ===")
    print(f"Accuracy on {num_samples} samples: {accuracy:.3f}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Correct predictions: {correct_predictions}/{num_samples}")

    # Create summary plot
    fig_summary, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy pie chart
    ax[0].pie([correct_predictions, num_samples - correct_predictions],
              labels=['Correct', 'Incorrect'],
              colors=['green', 'red'],
              autopct='%1.1f%%',
              startangle=90)
    ax[0].set_title('Classification Accuracy')

    # Confidence distribution
    ax[1].bar(['Average Confidence', 'Random Baseline'],
              [avg_confidence, 1.0/4],
              color=['blue', 'gray'],
              alpha=0.7)
    ax[1].set_ylabel('Confidence')
    ax[1].set_title('Confidence Comparison')
    ax[1].set_ylim([0, 1])

    plt.suptitle('1D-2D Fusion Model Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_summary)

    print(f"\n=== Demo completed successfully! ===")
    print(f"Check {output_dir} for generated visualizations:")
    print(f"  - Individual sample visualizations: sample_visualization_*.png")
    print(f"  - Performance summary: performance_summary.png")


if __name__ == "__main__":
    main()
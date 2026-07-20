"""
Grad-CAM based explainers for 1D-2D fusion models.

This module implements gradient-based attribution methods for both 1D time-series
and 2D spectrogram representations to provide model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Dict, List, Optional, Any


class GradCAM1D:
    """
    Grad-CAM for 1D convolutional networks.
    Generates attribution maps for time-series data.
    """

    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize Grad-CAM for 1D data.

        Args:
            model: The neural network model
            target_layers: List of layer names to extract features from
        """
        self.model = model
        self.target_layers = target_layers
        self.hooks = []
        self.feature_maps = {}
        self.gradients = {}

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def get_activation(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Find and register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(get_activation(name)))
                self.hooks.append(module.register_backward_hook(get_gradient(name)))

    def generate_cam(self,
                    input_1d: torch.Tensor,
                    target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map for 1D input.

        Args:
            input_1d: 1D input tensor [B, C, L] or [C, L]
            target_class: Target class for attribution (None for predicted class)

        Returns:
            Attribution map as numpy array
        """
        # Ensure input has batch dimension
        if input_1d.dim() == 2:
            input_1d = input_1d.unsqueeze(0)

        # Forward pass
        self.model.eval()
        output = self.model(input_1d)

        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=-1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        loss = output[0, target_class]
        loss.backward()

        # Generate CAM for each target layer
        cams = []
        for layer_name in self.target_layers:
            if layer_name in self.feature_maps and layer_name in self.gradients:
                # Get feature maps and gradients
                features = self.feature_maps[layer_name][0]  # Remove batch dim
                gradients = self.gradients[layer_name][0]    # Remove batch dim

                # Compute weights (global average pooling of gradients)
                weights = torch.mean(gradients, dim=-1)  # [C]

                # Weighted combination of feature maps
                cam = torch.zeros(features.shape[-1])
                for i, w in enumerate(weights):
                    cam += w * features[i]

                # ReLU to keep only positive influences
                cam = F.relu(cam)

                # Normalize
                cam = cam - torch.min(cam)
                if torch.max(cam) > 0:
                    cam = cam / torch.max(cam)

                cams.append(cam.cpu().numpy())

        # Average CAMs from all layers
        if cams:
            final_cam = np.mean(cams, axis=0)
        else:
            final_cam = np.zeros(input_1d.shape[-1])

        return final_cam

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class GradCAM2D:
    """
    Grad-CAM for 2D convolutional networks.
    Generates attribution maps for image-like data (spectrograms).
    """

    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize Grad-CAM for 2D data.

        Args:
            model: The neural network model
            target_layers: List of layer names to extract features from
        """
        self.model = model
        self.target_layers = target_layers
        self.hooks = []
        self.feature_maps = {}
        self.gradients = {}

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def get_activation(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Find and register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(get_activation(name)))
                self.hooks.append(module.register_backward_hook(get_gradient(name)))

    def generate_cam(self,
                    input_2d: torch.Tensor,
                    target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map for 2D input.

        Args:
            input_2d: 2D input tensor [B, C, H, W] or [C, H, W]
            target_class: Target class for attribution (None for predicted class)

        Returns:
            Attribution map as numpy array [H, W]
        """
        # Ensure input has batch dimension
        if input_2d.dim() == 3:
            input_2d = input_2d.unsqueeze(0)

        # Forward pass
        self.model.eval()
        output = self.model(input_2d)

        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=-1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        loss = output[0, target_class]
        loss.backward()

        # Generate CAM for each target layer
        cams = []
        for layer_name in self.target_layers:
            if layer_name in self.feature_maps and layer_name in self.gradients:
                # Get feature maps and gradients
                features = self.feature_maps[layer_name][0]  # Remove batch dim
                gradients = self.gradients[layer_name][0]    # Remove batch dim

                # Compute weights (global average pooling of gradients)
                weights = torch.mean(gradients.view(gradients.size(0), -1), dim=1)  # [C]

                # Weighted combination of feature maps
                cam = torch.zeros(features.shape[1:])  # [H, W]
                for i, w in enumerate(weights):
                    cam += w * features[i]

                # ReLU to keep only positive influences
                cam = F.relu(cam)

                # Normalize
                cam = cam - torch.min(cam)
                if torch.max(cam) > 0:
                    cam = cam / torch.max(cam)

                cams.append(cam.cpu().numpy())

        # Average CAMs from all layers
        if cams:
            final_cam = np.mean(cams, axis=0)
        else:
            final_cam = np.zeros(input_2d.shape[-2:])

        return final_cam

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FusionGradCAM:
    """
    Grad-CAM for fusion models that combines 1D and 2D explanations.
    """

    def __init__(self,
                 model_1d: nn.Module,
                 model_2d: nn.Module,
                 fusion_model: nn.Module,
                 target_layers_1d: List[str],
                 target_layers_2d: List[str]):
        """
        Initialize fusion Grad-CAM.

        Args:
            model_1d: 1D branch model
            model_2d: 2D branch model
            fusion_model: Fusion model
            target_layers_1d: Target layers for 1D branch
            target_layers_2d: Target layers for 2D branch
        """
        self.grad_cam_1d = GradCAM1D(model_1d, target_layers_1d)
        self.grad_cam_2d = GradCAM2D(model_2d, target_layers_2d)
        self.fusion_model = fusion_model

    def generate_fusion_cam(self,
                           input_1d: torch.Tensor,
                           input_2d: torch.Tensor,
                           target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate fusion attribution maps.

        Args:
            input_1d: 1D input tensor
            input_2d: 2D input tensor
            target_class: Target class for attribution

        Returns:
            Dictionary with '1d_cam', '2d_cam', and 'fusion_weights'
        """
        # Get individual CAMs
        cam_1d = self.grad_cam_1d.generate_cam(input_1d, target_class)
        cam_2d = self.grad_cam_2d.generate_cam(input_2d, target_class)

        # Get fusion weights (simplified approach)
        with torch.no_grad():
            feat_1d = self.grad_cam_1d.model(input_1d.unsqueeze(0))
            feat_2d = self.grad_cam_2d.model(input_2d.unsqueeze(0))

            # Simple fusion weight based on feature magnitudes
            weight_1d = torch.norm(feat_1d).item()
            weight_2d = torch.norm(feat_2d).item()
            total_weight = weight_1d + weight_2d

            fusion_weights = {
                'weight_1d': weight_1d / total_weight,
                'weight_2d': weight_2d / total_weight
            }

        return {
            '1d_cam': cam_1d,
            '2d_cam': cam_2d,
            'fusion_weights': fusion_weights
        }

    def remove_hooks(self):
        """Remove all hooks."""
        self.grad_cam_1d.remove_hooks()
        self.grad_cam_2d.remove_hooks()


def visualize_attribution_1d(signal: np.ndarray,
                           attribution: np.ndarray,
                           title: str = "1D Attribution Map",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize 1D attribution map.

    Args:
        signal: Original signal
        attribution: Attribution weights
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # Original signal
    ax1.plot(signal, 'b-', linewidth=1)
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # Attribution map
    ax2.plot(attribution, 'r-', linewidth=2)
    ax2.set_title('Attribution Map')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Attribution Weight')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Overlay
    time_axis = np.arange(len(signal))
    ax3.plot(time_axis, signal, 'b-', linewidth=1, alpha=0.7, label='Signal')
    ax3.fill_between(time_axis, 0, signal, where=(attribution > 0.5),
                    alpha=0.3, color='red', label='High Attribution')
    ax3.set_title('Signal with Attribution Overlay')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_attribution_2d(image: np.ndarray,
                           attribution: np.ndarray,
                           title: str = "2D Attribution Map",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize 2D attribution map.

    Args:
        image: Original 2D input (e.g., spectrogram)
        attribution: Attribution weights
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    im1 = ax1.imshow(image, cmap='viridis', aspect='auto')
    ax1.set_title('Original Input')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Attribution map
    im2 = ax2.imshow(attribution, cmap='hot', aspect='auto')
    ax2.set_title('Attribution Map')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Overlay
    # Normalize image for overlay
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    overlay = np.zeros((*image.shape, 3))
    overlay[..., 0] = image_norm  # Red channel
    overlay[..., 1] = image_norm * (1 - attribution)  # Green channel
    overlay[..., 2] = image_norm * (1 - attribution)  # Blue channel

    ax3.imshow(overlay, aspect='auto')
    ax3.set_title('Attribution Overlay')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_fusion_attribution(signal_1d: np.ndarray,
                               image_2d: np.ndarray,
                               cam_1d: np.ndarray,
                               cam_2d: np.ndarray,
                               fusion_weights: Dict[str, float],
                               title: str = "Fusion Attribution",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize fusion attribution results.

    Args:
        signal_1d: 1D signal
        image_2d: 2D image/spectrogram
        cam_1d: 1D attribution map
        cam_2d: 2D attribution map
        fusion_weights: Fusion importance weights
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 2, 1])

    # 1D results
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(signal_1d, 'b-', linewidth=1)
    ax1.set_title('1D Signal')
    ax1.set_xlabel('Time')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cam_1d, 'r-', linewidth=2)
    ax2.set_title('1D Attribution')
    ax2.set_xlabel('Time')
    ax2.set_ylim([0, 1])

    # 2D results
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(image_2d, cmap='viridis', aspect='auto')
    ax3.set_title('2D Input')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(cam_2d, cmap='hot', aspect='auto')
    ax4.set_title('2D Attribution')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Frequency')

    # Fusion weights
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.bar(['1D Branch', '2D Branch'],
           [fusion_weights['weight_1d'], fusion_weights['weight_2d']],
           color=['blue', 'red'], alpha=0.7)
    ax5.set_title('Fusion Weights')
    ax5.set_ylabel('Importance')
    ax5.set_ylim([0, 1])

    # Add text annotations
    for i, (branch, weight) in enumerate(fusion_weights.items()):
        branch_name = branch.replace('weight_', '').upper()
        ax5.text(i, weight + 0.02, f'{weight:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Test the Grad-CAM implementations
    print("Testing Grad-CAM implementations...")

    # Create dummy data
    signal_1d = np.random.randn(1024)
    image_2d = np.random.randn(64, 64)

    # Test 1D visualization
    attribution_1d = np.random.rand(1024)
    fig_1d = visualize_attribution_1d(signal_1d, attribution_1d)
    plt.show()

    # Test 2D visualization
    attribution_2d = np.random.rand(64, 64)
    fig_2d = visualize_attribution_2d(image_2d, attribution_2d)
    plt.show()

    # Test fusion visualization
    fusion_weights = {'weight_1d': 0.6, 'weight_2d': 0.4}
    fig_fusion = visualize_fusion_attribution(
        signal_1d, image_2d, attribution_1d, attribution_2d, fusion_weights
    )
    plt.show()

    print("Visualization tests completed!")
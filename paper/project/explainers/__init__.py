"""
Explainers Module for 1D-2D Fusion Fault Diagnosis

This module provides explainability tools for understanding model decisions:
- Grad-CAM for both 1D and 2D modalities
- Fusion attribution methods
- Visualization utilities
"""

from .grad_cam import (
    GradCAM1D,
    GradCAM2D,
    FusionGradCAM,
    visualize_attribution_1d,
    visualize_attribution_2d,
    visualize_fusion_attribution
)

__all__ = [
    'GradCAM1D',
    'GradCAM2D',
    'FusionGradCAM',
    'visualize_attribution_1d',
    'visualize_attribution_2d',
    'visualize_fusion_attribution'
]
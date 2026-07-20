"""
Alignment Module for 1D-2D Fusion Fault Diagnosis

This module provides three-layer alignment mechanisms:
- Physical Alignment: Energy distribution and spectral consistency
- Semantic Alignment: Cross-modal contrastive learning and feature alignment
- Geometric Alignment: Neighborhood preservation and manifold alignment
"""

from .physical_alignment import (
    PhysicalAlignmentLoss,
    PhysicalConstraintLayer,
    compute_physical_alignment_metrics
)

from .semantic_alignment import (
    SemanticAlignmentLoss,
    SemanticProjectionHead,
    CrossModalMemoryBank,
    compute_semantic_alignment_metrics
)

from .geometric_alignment import (
    GeometricAlignmentLoss,
    GeometricProjection,
    compute_geometric_alignment_metrics
)

__all__ = [
    # Physical alignment
    'PhysicalAlignmentLoss',
    'PhysicalConstraintLayer',
    'compute_physical_alignment_metrics',

    # Semantic alignment
    'SemanticAlignmentLoss',
    'SemanticProjectionHead',
    'CrossModalMemoryBank',
    'compute_semantic_alignment_metrics',

    # Geometric alignment
    'GeometricAlignmentLoss',
    'GeometricProjection',
    'compute_geometric_alignment_metrics'
]
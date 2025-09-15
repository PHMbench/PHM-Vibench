"""
ISFM_Prompt: Industrial Signal Foundation Model with Prompt-guided Learning

This module implements the innovative Prompt-guided ISFM architecture that combines
system metadata as learnable prompt vectors with contrastive learning for enhanced
cross-system fault diagnosis generalization.

Key Innovation:
- Two-level prompt encoding: System-level (Dataset_id + Domain_id) + Sample-level (Sample_rate)
- Prompt-guided contrastive learning for cross-domain knowledge transfer  
- Complete independence from existing ISFM models to avoid conflicts

Architecture Components:
- components: SystemPromptEncoder, PromptFusion utilities
- embedding: E_01_HSE_v2 (Prompt-guided Hierarchical Signal Embedding)
- backbone: Reuse existing ISFM backbone networks
- task_head: Reuse existing ISFM task heads

Author: PHM-Vibench Team
Date: 2025-01-06
License: MIT
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

# Import prompt-specific submodules
from . import components
from . import embedding

# Component dictionaries for factory pattern integration
PROMPT_EMBEDDING_REGISTRY = {
    'E_01_HSE_v2': 'embedding.E_01_HSE_v2'
}

PROMPT_COMPONENT_REGISTRY = {
    'SystemPromptEncoder': 'components.SystemPromptEncoder',
    'PromptFusion': 'components.PromptFusion'
}

# Model registry (lazy import to avoid circular dependencies)
def _get_prompt_models():
    """Lazy import for prompt models to avoid circular dependencies."""
    models = {}
    try:
        from .M_02_ISFM_Prompt import Model as M_02_ISFM_Prompt
        models['M_02_ISFM_Prompt'] = M_02_ISFM_Prompt
    except ImportError:
        pass  # Model not yet implemented
    return models

PROMPT_MODEL_REGISTRY = _get_prompt_models()

__all__ = [
    'components',
    'embedding', 
    'PROMPT_EMBEDDING_REGISTRY',
    'PROMPT_COMPONENT_REGISTRY',
    'PROMPT_MODEL_REGISTRY'
]
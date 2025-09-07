"""
ISFM_Prompt Module - Prompt-guided Industrial Signal Foundation Model

This module implements the Prompt-guided ISFM architecture for industrial fault diagnosis.
It combines system metadata as learnable prompts with SOTA contrastive learning methods.

Components:
- M_02_ISFM_Prompt: Main prompt-guided foundation model (lazy import)
- SystemPromptEncoder: Two-level system metadata encoder 
- PromptFusion: Multi-strategy fusion for signal and prompt features

Author: PHM-Vibench Team
Date: 2025-01-06
"""

# Lazy imports to avoid dependency issues during testing
def _get_M_02_ISFM_Prompt():
    """Lazy import for M_02_ISFM_Prompt to avoid dependency loading during testing."""
    from .M_02_ISFM_Prompt import Model
    return Model

# Direct imports for components (no ISFM dependencies)
from .components.SystemPromptEncoder import SystemPromptEncoder
from .components.PromptFusion import PromptFusion

__all__ = [
    'SystemPromptEncoder', 
    'PromptFusion'
]

# Add M_02_ISFM_Prompt to __all__ but don't import it directly
__all__.append('M_02_ISFM_Prompt')

# Make M_02_ISFM_Prompt available through attribute access
def __getattr__(name):
    if name == 'M_02_ISFM_Prompt':
        return _get_M_02_ISFM_Prompt()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
"""
ISFM_Prompt Embedding Module

This module contains embedding layers specifically designed for Prompt-guided
Industrial Signal Foundation Model (ISFM) architecture.

The embedding layers integrate system metadata as learnable prompts with
signal processing for enhanced cross-system fault diagnosis generalization.

Key Features:
- Complete independence from existing ISFM embeddings (zero dependencies)
- Two-level prompt encoding: System + Sample metadata integration
- Seamless fallback to signal-only processing when metadata unavailable

Components:
- E_01_HSE_v2: Completely new Prompt-guided Hierarchical Signal Embedding
  
CRITICAL: E_01_HSE_v2 has ZERO dependencies on existing E_01_HSE.py to ensure
complete model isolation and avoid any code mixing conflicts.

Author: PHM-Vibench Team  
Date: 2025-01-06
License: MIT
"""

# Import prompt-guided embedding components
# Note: Using lazy import to handle cases where components are not yet implemented
def _get_prompt_embeddings():
    """Lazy import for prompt embeddings to avoid dependency issues."""
    embeddings = {}
    try:
        from .E_01_HSE_v2 import E_01_HSE_v2
        embeddings['E_01_HSE_v2'] = E_01_HSE_v2
    except ImportError:
        pass  # Component not yet implemented
    return embeddings

# Component registry for factory pattern
PROMPT_EMBEDDINGS = _get_prompt_embeddings()

__all__ = list(PROMPT_EMBEDDINGS.keys()) if PROMPT_EMBEDDINGS else []
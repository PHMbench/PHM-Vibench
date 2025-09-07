"""
ISFM_Prompt Embedding Module

This module contains embedding layers specifically designed for Prompt-guided
Industrial Signal Foundation Model (ISFM) architecture.

The embedding layers integrate system metadata as learnable prompts with
signal processing for enhanced cross-system fault diagnosis generalization.

Components:
- E_01_HSE_Prompt: Prompt-guided Hierarchical Signal Embedding (in ISFM/embedding/E_01_HSE.py)

Note: The actual E_01_HSE_Prompt implementation is located in the main ISFM embedding
directory to maintain backward compatibility, but it's logically part of this module.

Author: PHM-Vibench Team
Date: 2025-01-06
"""

# Import from the main ISFM embedding directory for compatibility
try:
    from ...ISFM.embedding.E_01_HSE import E_01_HSE_Prompt
    __all__ = ['E_01_HSE_Prompt']
except ImportError:
    # Fallback if main ISFM module not available
    __all__ = []
    print("Warning: E_01_HSE_Prompt not available - check ISFM embedding module")
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

from . import components
from . import embedding

__all__ = [
    'components',
    'embedding',
]

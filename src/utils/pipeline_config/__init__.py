"""
Pipeline Configuration Module

This module provides utilities for multi-stage pipeline configuration and management,
including weight loading, result summarization, and configuration generation.

Author: PHM-Vibench Team
Date: 2025-01-06
"""

# Import existing utilities from the base module
from .base_utils import (
    load_pretrained_weights,
    generate_pipeline_summary
)

# Import HSE Prompt integration utilities
from .hse_prompt_integration import (
    HSEPromptPipelineIntegration,
    create_pretraining_config,
    create_finetuning_config,
    adapt_checkpoint_loading
)

__all__ = [
    'load_pretrained_weights',
    'generate_pipeline_summary', 
    'HSEPromptPipelineIntegration',
    'create_pretraining_config',
    'create_finetuning_config',
    'adapt_checkpoint_loading'
]
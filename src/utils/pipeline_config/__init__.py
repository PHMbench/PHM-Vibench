"""
Pipeline Configuration Module

This module provides utilities for multi-stage pipeline configuration and management,
including weight loading, result summarization, and configuration generation.

⚠️ HSE utilities have been moved to src.utils.hse for better organization.
Use the new imports for HSE functionality:
- from src.utils.hse.integration_utils import HSEIntegrationUtils, create_pretraining_config, create_finetuning_config

Author: PHM-Vibench Team
Date: 2025-01-06
Updated: 2025-11-20
"""

# Import existing utilities from the base module
from .base_utils import (
    load_pretrained_weights,
    generate_pipeline_summary
)

# Import HSE utilities from their new location (with deprecation warning)
try:
    from ..hse.integration_utils import (
        HSEIntegrationUtils as HSEPromptPipelineIntegration,
        create_pretraining_config,
        create_finetuning_config,
        adapt_checkpoint_loading
    )
    _hse_available = True
except ImportError:
    _hse_available = False
    HSEPromptPipelineIntegration = None
    create_pretraining_config = None
    create_finetuning_config = None
    adapt_checkpoint_loading = None

__all__ = [
    'load_pretrained_weights',
    'generate_pipeline_summary'
]

if _hse_available:
    __all__.extend([
        'HSEPromptPipelineIntegration',
        'create_pretraining_config',
        'create_finetuning_config',
        'adapt_checkpoint_loading'
    ])

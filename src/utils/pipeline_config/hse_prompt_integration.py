"""
⚠️ DEPRECATED: hse_prompt_integration.py

This module has been moved to the HSE utilities directory for better organization.

NEW: from src.utils.hse.integration_utils import HSEIntegrationUtils
OLD: from src.utils.pipeline_config.hse_prompt_integration import HSEPromptPipelineIntegration

Migration Guide:
- Replace 'from src.utils.pipeline_config.hse_prompt_integration import HSEPromptPipelineIntegration'
- With 'from src.utils.hse.integration_utils import HSEIntegrationUtils'
- Update class name from HSEPromptPipelineIntegration to HSEIntegrationUtils

Last updated: 2025-11-20
Removal timeline: v2.1.0
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "src.utils.pipeline_config.hse_prompt_integration is deprecated. "
    "Please use src.utils.hse.integration_utils instead. "
    "This module will be removed in v2.1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export functions and classes from the new location
from ..hse.integration_utils import (
    HSEIntegrationUtils as HSEPromptPipelineIntegration,
    create_pretraining_config,
    create_finetuning_config,
    adapt_checkpoint_loading,
)

# Maintain backward compatibility
__all__ = [
    'HSEPromptPipelineIntegration',
    'create_pretraining_config',
    'create_finetuning_config',
    'adapt_checkpoint_loading',
]

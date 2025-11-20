"""
⚠️ DEPRECATED: hse_prompt_validator.py

This module has been moved to the HSE utilities directory for better organization.

NEW: from src.utils.hse.prompt_validator import HSPPromptValidator
OLD: from src.utils.config.hse_prompt_validator import HSEPromptConfigValidator

Migration Guide:
- Replace 'from src.utils.config.hse_prompt_validator import HSEPromptConfigValidator'
- With 'from src.utils.hse.prompt_validator import HSPPromptValidator'
- Update class name from HSEPromptConfigValidator to HSPPromptValidator

Last updated: 2025-11-20
Removal timeline: v2.1.0
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "src.utils.config.hse_prompt_validator is deprecated. "
    "Please use src.utils.hse.prompt_validator instead. "
    "This module will be removed in v2.1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the new location
from ..hse.prompt_validator import HSPPromptValidator as HSEPromptConfigValidator

# Maintain backward compatibility
__all__ = ['HSEPromptConfigValidator']
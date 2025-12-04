"""
Simplified ISFM_Prompt Embedding Module

This module contains simplified embedding layers for Prompt-guided
Industrial Signal Foundation Model (ISFM) architecture.

The embedding layers combine heterogeneous signal processing with lightweight
system-specific learnable prompts for enhanced cross-system generalization.

Key Features:
- Heterogeneous Signal Embedding (HSE) with system prompts
- Simple Dataset_id → learnable prompt mapping
- Direct signal + prompt combination (add/concat)
- Lightweight and easy to understand

Components:
- HSE_prompt: Simplified HSE with system prompts

Author: PHM-Vibench Team
Date: 2025-01-23
License: MIT
"""

# Import simplified prompt embedding components
from .HSE_prompt import HSE_prompt

# Component registry for factory pattern
# 说明：
# - 当前主线实验（Experiment 3–7）只使用 `HSE_prompt` 作为标准 HSE-Prompt embedding；
# - `E_01_HSE_v2` 保留在单独文件中作为研究用实现，不再在此自动注册，避免在加载
#   ISFM_Prompt 时引入额外依赖和复杂度。
PROMPT_EMBEDDINGS = {
    'HSE_prompt': HSE_prompt,
}

__all__ = ['HSE_prompt']

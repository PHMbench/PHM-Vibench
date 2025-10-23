"""
ISFM_Prompt Components

Core components for Prompt-guided Industrial Signal Foundation Model.

Components:
- SystemPromptEncoder: Two-level hierarchical prompt encoding
- PromptFusion: Multi-strategy signal-prompt fusion
- PromptLibrary: Metadata-conditioned prompt candidate generation
- PromptInjector: Prompt-to-token injection utilities
- PromptSelector: Discrete or continuous prompt selection
"""

from .SystemPromptEncoder import SystemPromptEncoder
from .PromptFusion import PromptFusion
from .PromptLibrary import PromptLibrary, PromptLibraryOutput, build_prompt_library
from .PromptInjector import PromptInjector, InjectionMode
from .PromptSelector import PromptSelector, PromptSelectionOutput, SelectionMode

__all__ = [
    'SystemPromptEncoder',
    'PromptFusion',
    'PromptLibrary',
    'PromptLibraryOutput',
    'build_prompt_library',
    'PromptInjector',
    'InjectionMode',
    'PromptSelector',
    'PromptSelectionOutput',
    'SelectionMode',
]

"""
Pretraining tasks for PHM foundation models.

This module contains various pretraining tasks that can be used to train
PHM foundation models in an unsupervised manner.
"""

# Import all pretraining tasks to ensure they are registered
from .classification import *
from .classification_prediction import *
from .prediction import *
from .masked_reconstruction import *
from .flow_pretrain import *

# Import self-testing infrastructure
from .self_testing import (
    ValidationResult,
    TestConfiguration, 
    PerformanceMetrics,
    SelfTestOrchestrator,
    ResourceManager,
    TimeoutError,
)

__all__ = [
    'MaskedReconstructionTask',
    'FlowPretrainTask',
    # Self-testing infrastructure
    'ValidationResult',
    'TestConfiguration', 
    'PerformanceMetrics',
    'SelfTestOrchestrator',
    'ResourceManager',
    'TimeoutError',
]
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

__all__ = [
    'MaskedReconstructionTask',
]
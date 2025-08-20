"""MLP family models."""
from .Dlinear import Model as DLinear
from .ResNetMLP import Model as ResNetMLP
from .MLPMixer import Model as MLPMixer
from .gMLP import Model as gMLP
from .DenseNetMLP import Model as DenseNetMLP

__all__ = [
    "DLinear",
    "ResNetMLP",
    "MLPMixer",
    "gMLP",
    "DenseNetMLP"
]

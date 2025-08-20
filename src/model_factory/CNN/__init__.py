"""CNN family models."""
from .ResNet1D import Model as ResNet1D
from .TCN import Model as TCN
from .AttentionCNN import Model as AttentionCNN
from .MobileNet1D import Model as MobileNet1D
from .MultiScaleCNN import Model as MultiScaleCNN

__all__ = [
    "ResNet1D",
    "TCN",
    "AttentionCNN",
    "MobileNet1D",
    "MultiScaleCNN"
]

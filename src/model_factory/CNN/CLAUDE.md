# CNN Models - CLAUDE.md

This module provides architecture guidance for convolutional neural networks in PHM-Vibench. For available models and configuration, see [@README.md].

## Architecture Overview

CNN models in this module implement 1D convolutions specifically designed for vibration signal processing:

```
Input Signal (L, C)
     ↓
┌─────────────────────────────────────┐
│  Convolutional Blocks (N ×)           │
│  - 1D Convolution                    │
│  - Batch Normalization               │
│  - ReLU Activation                   │
│  - Pooling (optional)                │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Global Pooling / Flatten            │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Fully Connected Layers              │
│  - Classification / Prediction       │
└─────────────────────────────────────┘
     ↓
Output
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `ResNet1D` | Residual network with skip connections | Deep architectures, gradient flow |
| `AttentionCNN` | CNN with attention mechanisms | Salient feature detection |
| `MultiScaleCNN` | Multi-scale feature extraction | Varying frequency patterns |
| `MobileNet1D` | Efficient mobile-friendly CNN | Edge deployment |
| `TCN` | Temporal Convolutional Network | Sequential dependencies |

## Design Considerations

### 1D Convolutions
- Unlike 2D CNNs for images, 1D convolutions operate along the time axis
- Kernel size determines temporal receptive field
- Deeper networks capture longer-range patterns

### Residual Connections
- ResNet architecture enables training very deep networks
- Skip connections help gradient flow
- Batch normalization stabilizes training

## Configuration Pattern

```yaml
model:
  type: "CNN"
  name: "ResNet1D"

  # Architecture
  depth: 18              # Network depth (layers)
  in_channels: 1         # Input channels
  num_classes: 10        # Output classes

  # Convolution parameters
  kernel_size: 7         # Convolution kernel size
  stride: 2              # Stride for downsampling
  padding: 3             # Padding to maintain size

  # Regularization
  dropout: 0.2
  activation: "relu"
```

## Kernel Size Selection

| Kernel Size | Receptive Field | Best For |
|-------------|-----------------|----------|
| 3 | Small | High-frequency features |
| 7 | Medium | General purpose |
| 15+ | Large | Low-frequency patterns |

## Related Documentation

- [@README.md] - Configuration and Usage Guide
- [@../README.md] - Model Factory Overview

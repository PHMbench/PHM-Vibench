# Neural Operator Models - CLAUDE.md

This module provides architecture guidance for Neural Operator (NO) models in PHM-Vibench. For configuration details, see [@README.md].

## Architecture Overview

Neural Operators implement continuous kernel parameterizations for learning operators (mappings between function spaces):

```
Input Signal (L, C)
     ↓
┌─────────────────────────────────────┐
│  Fourier Layer (N ×)                 │
│  - FFT to spectral domain             │
│  - Complex-valued linear transform    │
│  - Inverse FFT                        │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Activation & Pooling                │
│  - GELU / ReLU                        │
│  - Adaptive pooling                   │
└─────────────────────────────────────┘
     ↓
Output
```

## Available Models

| Model | Description | Key Feature |
|-------|-------------|-------------|
| `FNO` | Fourier Neural Operator | Spectral domain learning |
| `DeepONet` | Deep Operator Network | Branch-trunk architecture |

## Design Principles

### Spectral Learning
- Operates in frequency domain via FFT
- Learns global patterns efficiently
- Resolution-independent inference

### Continuous Operators
Unlike traditional networks that discretize functions:
- NO models learn operators in continuous space
- Can generalize to different resolutions
- Better for physical system modeling

## Configuration Pattern

```yaml
model:
  type: "NO"
  name: "FNO"

  # Architecture
  modes: 32              # Fourier modes to keep
  width: 64              # Hidden layer width
  num_layers: 4          # Number of FNO layers

  # Parameters
  in_channels: 1         # Input channels
  out_channels: 10       # Output (classes/features)

  # Regularization
  dropout: 0.1
```

## When to Use Neural Operators

- **Multi-scale patterns**: When signals have varying frequencies
- **Physical modeling**: When system follows PDEs
- **Resolution variation**: When train/test resolutions differ

## Related Documentation

- [@README.md] - Configuration Guide
- [@../README.md] - Model Factory Overview

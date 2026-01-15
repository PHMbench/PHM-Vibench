# MLP Models - CLAUDE.md

This module provides architecture guidance for multi-layer perceptron models in PHM-Vibench. For configuration details, see [@README.md].

## Architecture Overview

MLP models implement simple feedforward networks for baseline comparisons and simple classification tasks:

```
Input Features (F)
     ↓
┌─────────────────────────────────────┐
│  Fully Connected Layers (N ×)        │
│  - Linear transformation               │
│  - Activation (ReLU/GELU/tanh)        │
│  - Dropout (optional)                 │
│  - Batch Normalization (optional)     │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Output Layer                         │
│  - Softmax / Linear                   │
└─────────────────────────────────────┘
     ↓
Output (Classes / Regression Value)
```

## Design Philosophy

### Simplicity First
- MLPs serve as strong baselines
- Fast to train and evaluate
- Interpretable architecture

### When to Use MLP
- **Baseline**: Compare against more complex models
- **Feature-based**: After feature extraction
- **Simple patterns**: When temporal structure is less important
- **Quick experiments**: Rapid prototyping

## Configuration Pattern

```yaml
model:
  type: "MLP"
  name: "MLP"

  # Architecture
  input_dim: 128          # Input feature dimension
  hidden_dims: [256, 128] # Hidden layer dimensions
  output_dim: 10          # Output (classes or regression)

  # Regularization
  dropout: 0.2
  activation: "relu"

  # Initialization
  init_method: "kaiming"
```

## Limitations

- **No temporal modeling**: Cannot capture sequential dependencies
- **Fixed input size**: Requires padded/truncated sequences
- **Limited representation**: May underfit complex patterns

## Related Documentation

- [@README.md] - Configuration Guide
- [@../README.md] - Model Factory Overview

# RNN Models - CLAUDE.md

This module provides architecture guidance for recurrent neural networks in PHM-Vibench. For available models and configuration, see [@README.md].

## Architecture Overview

RNN models implement recurrent architectures designed for capturing sequential dependencies in vibration signals:

```
Input Signal (L, C)
     ↓
┌─────────────────────────────────────┐
│  RNN Layers (N ×)                     │
│  - LSTM / GRU cells                   │
│  - Bidirectional (optional)           │
│  - Dropout for regularization         │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Attention Layer (optional)          │
│  - Focus on important time steps     │
└─────────────────────────────────────┘
     ↓
Output (Classification/Prediction)
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `AttentionLSTM` | LSTM with attention mechanism | Long sequences, salient features |
| `AttentionGRU` | GRU with attention | Faster training, good performance |
| `ConvLSTM` | Convolutional LSTM | Spatial-temporal patterns |
| `ResidualRNN` | RNN with residual connections | Deep recurrent architectures |

## Design Considerations

### LSTM vs GRU
- **LSTM**: More parameters, better for complex patterns
- **GRU**: Fewer parameters, faster training

### Bidirectional Processing
- Processes sequence in both forward and backward directions
- Captures past and future context
- Doubles parameter count

### Sequence Length
RNNs can struggle with very long sequences (>10000 steps):
- Consider downsampling or windowing
- Use attention mechanisms to focus on important parts

## Configuration Pattern

```yaml
model:
  type: "RNN"
  name: "AttentionLSTM"

  # Architecture
  hidden_size: 128        # Hidden state dimension
  num_layers: 2           # Number of RNN layers
  bidirectional: true     # Process both directions

  # Regularization
  dropout: 0.2

  # Output
  num_classes: 10
```

## Related Documentation

- [@README.md] - Configuration and Usage Guide
- [@../README.md] - Model Factory Overview

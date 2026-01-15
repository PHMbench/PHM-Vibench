# Transformer Models - CLAUDE.md

This module provides architecture guidance for transformer-based architectures in PHM-Vibench. For available models and configuration, see [@README.md].

## Architecture Overview

Transformer models in this module implement attention-based architectures for long-range temporal modeling of vibration signals:

```
Input Signal (L, C)
     ↓
┌─────────────────────────────────────┐
│  Patch Extraction (optional)         │
│  - PatchTST: Non-overlapping patches  │
│  - Autoformer: Decomposition          │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Embedding Layer                     │
│  - Positional encoding               │
│  - Token embedding                   │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Encoder Layers (N ×)                │
│  - Multi-head self-attention         │
│  - Feed-forward network              │
│  - Layer normalization               │
└─────────────────────────────────────┘
     ↓
Output (Classification/Prediction)
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `PatchTST` | Patch-based Time Series Transformer | Long sequences, classification |
| `Autoformer` | Decomposition-based transformer | Forecasting with trend/seasonality |
| `Informer` | Efficient long-sequence transformer | Very long sequences |
| `Linformer` | Linear-complexity transformer | Memory-constrained scenarios |
| `ConvTransformer` | Hybrid CNN-Transformer | Local + global patterns |
| `Transformer_Dummy` | Testing/debugging placeholder | Development |

## Design Considerations

### Patching Strategy
- **PatchTST**: Divides signal into non-overlapping patches
- Reduces sequence length from L to L/patch_size
- Enables efficient attention computation

### Attention Mechanism
- **Self-attention**: Captures long-range dependencies
- **Multi-head**: Learns different representation subspaces
- **Causal masking**: For autoregressive tasks (forecasting)

## Configuration Pattern

```yaml
model:
  type: "Transformer"
  name: "PatchTST"

  # Architecture
  d_model: 128        # Model dimension
  n_heads: 8         # Attention heads
  num_layers: 6       # Encoder depth
  d_ff: 256          # Feed-forward dimension
  dropout: 0.1

  # Patching (PatchTST-specific)
  patch_size: 16
  stride: 8

  # Task-specific
  num_classes: 10    # or output_dim for prediction
```

## Integration with ISFM

When used within ISFM framework:
- Transformer models serve as the **backbone** component
- Configure via `model.backbone: "B_08_PatchTST"`
- See [@../ISFM/README.md] for ISFM integration

## Related Documentation

- [@README.md] - Configuration and Usage Guide
- [@../README.md] - Model Factory Overview
- [@../ISFM/README.md] - ISFM Framework

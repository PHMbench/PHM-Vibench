# ISFM Layers

This directory contains shared neural network layers used across ISFM components.

## Overview

The layers module provides reusable building blocks for constructing ISFM architectures, including attention mechanisms, normalization, and utility functions.

## Available Layers

### Attention Mechanisms

| File | Description |
|------|-------------|
| `SelfAttention_Family.py` | Standard multi-head self-attention variants |
| `AutoCorrelation.py` | Auto-correlation for time-series |
| `FourierCorrelation.py` | Frequency-domain correlation |
| `MultiWaveletCorrelation.py` | Wavelet-based correlation |
| `VIB_attnetion.py` | VIB-specific attention mechanisms |

### Encoder-Decoder Architectures

| File | Description |
|------|-------------|
| `Transformer_EncDec.py` | Standard transformer encoder-decoder |
| `Autoformer_EncDec.py` | Autoformer encoder-decoder |
| `Crossformer_EncDec.py` | Crossformer encoder-decoder |
| `ETSformer_EncDec.py` | ETSformer encoder-decoder |
| `Pyraformer_EncDec.py` | Pyraformer encoder-decoder |

### Processing Layers

| File | Description |
|------|-------------|
| `Conv_Blocks.py` | Convolutional blocks |
| `DWT_Decomposition.py` | Discrete Wavelet Transform decomposition |
| `Embed.py` | Embedding layers |
| `StandardNorm.py` | Normalization utilities |

### Utilities

| File/Directory | Description |
|----------------|-------------|
| `utils/` | Helper functions for layer operations |

## Usage

These layers are imported by backbone and task head implementations:

```python
# Example usage in backbone
from src.model_factory.ISFM.layers.SelfAttention_Family import MultiHeadAttention
from src.model_factory.ISFM.layers.StandardNorm import LayerNormalization

attention = MultiHeadAttention(d_model, n_heads)
norm = LayerNormalization(d_model)
```

## Related Documentation

- [@../README.md] - ISFM Module Overview
- [@../backbone/README.md] - Backbone Networks

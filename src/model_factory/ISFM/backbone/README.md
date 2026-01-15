# ISFM Backbone Networks

This directory contains backbone network implementations for the Industrial Signal Foundation Model (ISFM).

## Overview

Backbone networks perform the core temporal modeling in ISFM, receiving embeddings from the embedding layer and passing features to the task head.

## Available Backbones

| File | Backbone ID | Description | Best For |
|------|-------------|-------------|----------|
| `B_01_basic_transformer.py` | `B_01_basic_transformer` | Standard transformer encoder | General purpose |
| `B_02_basic_other.py` | `B_02_basic_other` | Alternative transformer variants | Research |
| `B_03_FITS.py` | `B_03_FITS` | FFT-based transformer | Frequency-domain analysis |
| `B_04_Dlinear.py` | `B_04_Dlinear` | Direct linear layer | Efficient baseline |
| `B_05_Mamba.py` | `B_05_Mamba` | State space model | Long sequences |
| `B_06_TimesNet.py` | `B_06_TimesNet` | TimesNet with period detection | Multi-periodicity |
| `B_07_TSMixer.py` | `B_07_TSMixer` | Time-series Mixer | Efficient forecasting |
| `B_08_PatchTST.py` | `B_08_PatchTST` | Patch-based transformer | Long sequences |
| `B_09_FNO.py` | `B_09_FNO` | Fourier Neural Operator | Spectral modeling |
| `B_10_VIBT.py` | `B_10_VIBT` | VIB-specific transformer | Industrial signals |
| `B_11_MomentumEncoder.py` | `B_11_MomentumEncoder` | Momentum-based encoding | Trend analysis |

## Configuration Example

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"

  # Backbone selection
  backbone: "B_08_PatchTST"  # or any B_XX_* ID

  # Backbone-specific parameters
  patch_size: 16
  num_layers: 6
  d_model: 128
  n_heads: 8
```

## Integration

Backbones are automatically loaded by the ISFM model:
```python
# In ISFM model initialization
from src.model_factory.ISFM.backbone import B_08_PatchTST
backbone = B_08_PatchTST(configs)
```

## Related Documentation

- [@../README.md] - ISFM Module Overview
- [@../task_head/README.md] - Task Heads
- [@../embedding/README.md] - Embedding Layers

# Native point forecasting

Four forecast models use one `DG.point_forecasting` Task and the existing runtime,
optimizer, selected-checkpoint and declared-metric lifecycle. Nothing is converted into
a classifier to increase the model count. No pretrained weights are required.

| Selector | Complete example | Upstream implementation |
| --- | --- | --- |
| `MLP.NLinear` | [NLinear](../../configs/experiments/model_integration/nlinear_dummy.yaml) | [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py) |
| `MLP.SparseTSF` | [SparseTSF](../../configs/experiments/model_integration/sparsetsf_dummy.yaml) | [SparseTSF](https://github.com/lss-1138/SparseTSF/blob/main/models/SparseTSF.py) |
| `MLP.FITS` | [FITS](../../configs/experiments/model_integration/fits_dummy.yaml) | [FITS](https://github.com/VEWOXIC/FITS/blob/main/models/FITS.py) |
| `RNN.SegRNN` | [SegRNN](../../configs/experiments/model_integration/segrnn_dummy.yaml) | [SegRNN](https://github.com/lss-1138/SegRNN/blob/main/models/SegRNN.py) |

Sources and their Apache-2.0 licenses were inspected on 2026-09-18. The NLinear port
retains the DLinear Authors' 2022 copyright. Each port describes its modifications;
third-party training scripts, downloads and runners are not imported. The repository
[Apache-2.0 license](../../LICENSE) accompanies these ports. Upstream parameter names
are not automatically remapped when loading weights.

## Exact task

A raw `[B,L+H,C]` window is split into prefix history and suffix target. Only history is
passed to the model. The model returns `[B,H,C]`. Existing class labels in metadata are
not regression targets. The Task optimizes horizon MSE and reports scalar MSE/MAE over
all horizon/channel elements with the existing stateful metric lifecycle.

The data configuration must explicitly use `normalization: none`, because per-window
statistics over history plus future would leak targets. Full-window SNR augmentation is
also rejected in this initial task; it cannot be silently disabled. Architecture-specific
normalization uses history only. This input-isolation property does not prove that all
train/validation windows are disjoint. The bundled Dummy split remains software smoke.

```bash
phmfactory --config configs/experiments/model_integration/nlinear_dummy.yaml
python -m pytest test/test_native_forecasting.py -q
python -m pytest test/test_native_model_public_path.py -q
```

## Architecture semantics

NLinear retains detached last-value centering and shared/individual linear projections.
SparseTSF retains residual temporal convolution and sparse phase-wise forecasting;
periods must divide both input and output lengths. SegRNN preserves explicit recurrent
cell, parallel/recurrent decoder, channel embedding and non-affine RevIN choices; segment
lengths must divide both lengths. RevIN statistics are local variables, not reusable
state from a previous batch.

FITS uses complex frequency interpolation, sample variance, and length-ratio compensation.
This port returns only the forecast tail; horizon-only MSE is not the full reconstruction
objective available in upstream training. Explicit inverse-FFT length preserves odd total
lengths. Complex-valued parameters require the tested PyTorch/optimizer combination;
unsupported precision or devices must not silently select another model.

Validation includes actual gradients and optimizer updates, strict parameter save/load,
independent projection/FFT/segment equations, and a counterfactual future-suffix change
that must leave model output unchanged. A locally discovered non-contiguous TorchMetrics
input failure was corrected by flattening scalar regression metric inputs at the Task
boundary; no values, labels or forecast lengths are repaired.

The public tests execute the exact complete configurations and selected-checkpoint test
on CI. Formula and smoke tests do not claim reproduction of upstream published metrics.

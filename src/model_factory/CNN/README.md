# CNN implementations

This family contains 1D convolutional implementations and the periodic-grid TimesNet
classifier. Set `model.type: CNN` and choose an exact module name: `ResNet1D`,
`AttentionCNN`, `MobileNet1D`, `MultiScaleCNN`, `TCN`, or `TimesNet`.
See [the catalogue](../model_registry.csv) for locations and [Model Factory](../README.md)
for the constructor and checkpoint contract. Listed code is not proof of a maintained
experiment or reproduction of every paper associated with an architecture name.

A model fragment for inspection:

```yaml
model:
  type: CNN
  name: ResNet1D
  input_dim: 3
  block_type: basic
  layers: [2, 2, 2, 2]
  initial_channels: 64
  num_classes: 4
```

The full experiment supplies data, Task and Trainer choices. Check the selected file for
input layout, pooling, output meaning and accepted parameters. Typical fields include
`input_dim`, depth/layers, channels, kernel size and output dimensions; they are not a
shared schema for every CNN. Do not add unused ISFM embedding/backbone/head settings.

When changing a CNN, test its actual forward contract and one compatible configuration.
Do not infer diagnostic accuracy, latency or forecasting support from the family name.

## TimesNet classification

[Complete CPU example](../../../configs/experiments/model_integration/timesnet_dummy.yaml):

```bash
phmfactory preflight --config configs/experiments/model_integration/timesnet_dummy.yaml
phmfactory --config configs/experiments/model_integration/timesnet_dummy.yaml
```

The commands above run from the repository root. Outside a checkout, resolve the packaged
config/data with `importlib.resources` and supply the data and writable cache paths explicitly,
as exercised by [the installed-wheel test](../../../test/test_timesnet_public.py).

The native model consumes fully observed float32 `[B,L,C]` and returns `[B,K]` logits, or
`(logits, features)` with `return_feature=True`. Configure `seq_len`, `input_dim`,
`num_classes`, `d_model`, `d_ff`, `e_layers`, `top_k`, `num_kernels`, and `dropout` explicitly.
Length is 2–5000, width is even, and top_k cannot exceed the positive FFT bins. Classification
uses no calendar covariates, forecasting normalization, pretrained weights or variable-length
mask. Multiple independent label ontologies require a different head, not silent pooling.

The port retains frequency selection, period-aligned 2D convolution, adaptive aggregation,
residuals and the original classification head. Period selection is **batch-dependent**,
as upstream: evaluation batch membership can affect predictions. Keep batching fixed when
comparing results. Internal period padding is an algorithm step, not resampling user data.

Source: [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/TimesNet.py),
MIT. The full notice is included in [TimesNet.py](TimesNet.py). One explicit correction masks
DC to negative infinity; upstream merely zeroes its amplitude and can select DC in a
zero-spectrum tie. This exception is tested separately rather than claimed as exact parity.

The [fidelity tests](../../../test/test_timesnet.py) execute pinned upstream definitions,
compare logits/loss/input and parameter gradients with the same weights, and restore a
PHMFactory state_dict strictly. Test-only upstream sources are not installed in the wheel.
Unused calendar parameters are not ported, and the positional buffer retains only the
configured prefix; arbitrary upstream checkpoints are not automatically remapped.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest test/test_timesnet.py -q
```

Dummy fit/test is software evidence only. It does not establish held-out PHM independence,
accuracy improvements or original-paper results. The existing Task, split, metric definition
and checkpoint selection remain unchanged.

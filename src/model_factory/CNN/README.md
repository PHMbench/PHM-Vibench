# CNN implementations

This family contains 1D convolutional implementations. Set `model.type: CNN` and choose
an exact module name: `ResNet1D`, `AttentionCNN`, `MobileNet1D`, `MultiScaleCNN`, or `TCN`.
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

# Transformer implementations

Set `model.type: Transformer` with an exact module name: `PatchTST`, `Autoformer`,
`Informer`, `Linformer`, `ConvTransformer`, `Transformer_Dummy`, or `TSLTransformer`.
The [catalogue](../model_registry.csv) locates implementations; [Model Factory](../README.md)
defines construction. These names do not assert original-paper reproduction or support
for every data/task combination.

## Example selection

```yaml
model:
  type: Transformer
  name: PatchTST
  input_dim: 3
  patch_size: 16
  stride: 8
  d_model: 256
  n_heads: 8
  num_layers: 6
  d_ff: 512
  dropout: 0.1
  num_classes: 4
```

This is a model fragment, not a complete experiment. Here `stride < patch_size`, so
successive patches overlap; do not describe this configuration as non-overlapping.

## Check the actual interface

Inspect the selected source for sequence length, channels, embedding size, attention
heads, encoder/decoder inputs, prediction horizon and output semantics. Typical names
such as `num_layers`, `e_layers` and `d_layers` are not interchangeable. Forecasting and
classification interfaces need their own compatible Task, not just a changed output
field. `TSLTransformer` is a clean-room classification implementation, not a general
forecasting contract.

Do not crop or pad an incompatible input silently. Test patch boundaries, input layout
and actual outputs for the implementation changed. Complexity or benchmark claims need
measured evidence for that configuration, not an architecture label.

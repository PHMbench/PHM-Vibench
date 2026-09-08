# Recurrent implementations

Use `model.type: RNN` with an exact module name: `AttentionLSTM`, `AttentionGRU`,
`ConvLSTM`, `ResidualRNN`, or `TransformerRNN`. Locations and typical arguments are in the
[model catalogue](../model_registry.csv); construction is described in [Model Factory](../README.md).

A selection fragment is:

```yaml
model:
  type: RNN
  name: AttentionLSTM
```

Read the selected implementation before adding its `input_dim`, `hidden_dim`, layer count,
bidirectionality and output settings. Field names, layouts, recurrent state handling and
supported outputs differ; a generic `hidden_size` recipe is not a universal interface.
ISFM embedding/backbone/head fields are not part of these standalone implementations.

Preserve the time and channel axes expected by the implementation. Verify the output
against the chosen Task with focused tests; do not turn an incompatible sequence or
state shape into another input by silent truncation. Catalogue presence is not a
performance, generalization or exact-experiment support claim.

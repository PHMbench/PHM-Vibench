# MLP and linear implementations

Use `model.type: MLP` and an exact implementation name from
[model_registry.csv](../model_registry.csv):

| Module | Architectural idea |
| --- | --- |
| `ResNetMLP` | Residual connections between MLP layers |
| `MLPMixer` | Token and channel mixing |
| `gMLP` | Gated mixing with sequence-length-dependent parameters |
| `DenseNetMLP` | Dense feature reuse |
| `Dlinear` | Linear projections with series decomposition |

A selection fragment is:

```yaml
model:
  type: MLP
  name: ResNetMLP
```

`name: MLP` and a standalone `model_name` field are not substitutes for Factory type/name
resolution. The [Model Factory](../README.md) constructs `Model(args_model, metadata)`;
a bare `build_model(args)` snippet is not the documented full call.

## Implementation-specific inputs

Inspect the selected module before setting `input_dim`, `hidden_dim`, depth, activation,
dropout or output dimensions. Mixing/gating models may depend on fixed token counts or
sequence lengths. `Dlinear` has forecasting-specific sequence settings; do not assume
all family members expose the same classification or regression outputs.

Check that predictions have the meaning and shape required by the Task. An output tensor
is not automatically a probability distribution, and an MLP is not automatically an
interpretable model.

## Evidence boundary

The former generic parameter/speed/accuracy table did not identify a reproducible exact
configuration or result source, so it is not retained as performance evidence. Numerical
comparisons require the actual data split, model settings, training budget and measured
results. This documentation change neither verifies nor changes those implementations.

For a model edit, add a focused forward/input-failure test and exercise a compatible
complete configuration. Do not add another model registry or inferred fallback head.

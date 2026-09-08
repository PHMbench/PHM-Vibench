# Neural-operator implementations

Use `model.type: NO` with the exact module `FNO`, `DeepONet`, `NeuralODE`, `GraphNO`, or
`WaveletNO`. See [model_registry.csv](../model_registry.csv) for locations and typical
arguments, and [Model Factory](../README.md) for construction and checkpoint behavior.

| Module | Structure to inspect |
| --- | --- |
| `FNO` | Spectral modes, channels and Fourier layers |
| `DeepONet` | Branch/trunk inputs and coordinate interface |
| `NeuralODE` | State, integration times, solver and tolerances |
| `GraphNO` | Graph representation and spectral/geometry assumptions |
| `WaveletNO` | Wavelet, levels and boundary handling |

A selection fragment is:

```yaml
model:
  type: NO
  name: FNO
```

Read the selected file for its actual constructor, inputs, outputs and dependencies;
these implementations do not share a universal tensor or argument schema. In particular,
a tuple branch/trunk interface or an irregular timestamp input must be implemented and
tested before a caller relies on it.

Resolution invariance, PDE accuracy, adjoint memory savings and irregular-sampling
support are method-specific properties, not guarantees for every module in this folder.
The former generic performance table and synthetic PDE recipes did not establish those
properties for an exact PHMFactory run and are not retained as operational guidance.

For a change, validate the selected implementation on its explicit grid/time/graph and
Task contract. Do not resample data, substitute solvers or repair incompatible shapes
without an explicit experiment change. A software smoke is not a physical-law or
cross-resolution generalization result.

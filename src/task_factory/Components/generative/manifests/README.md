# Generative Manifests

Synthetic data manifests provide the evidence chain for PHM generative outputs.
Generated samples are `exploratory` unless manifest, protocol, config,
normalization, leakage, and metric evidence are complete.

## Required Evidence

Benchmark-valid synthetic data requires:

- source split is `train`
- config and protocol hashes
- seed and environment metadata
- condition counts for `fault_label` and `domain_id`
- domain map path and hash when a domain map is used
- normalization method, scope, params artifact, and params hash
- leakage check results
- metric status and missing reasons

Forbidden synthetic source splits:

- `val`
- `valid`
- `validation`
- `test`
- `target_test`

## Normalization Contract

Allowed V0 normalization methods:

- `standardization`
- `robust_scaler`
- `none`

Recommended methods are:

- `robust_scaler`: median/IQR statistics.
- `standardization`: mean/std statistics, optionally paired with explicit
  clipping in a later goal.

MinMaxScaler is not allowed as the V0 default for PHM vibration generation
because impulses and high dynamic range can compress normal signal variation.

Manifests must record whether inverse transform is required for physical-scale
evaluation.

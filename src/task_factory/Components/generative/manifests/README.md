# Generative Manifests

Synthetic data manifests provide the evidence chain for PHM generative outputs.
Generated samples are `exploratory` unless manifest, protocol, config,
normalization, leakage, and metric evidence are complete.

## Sample Payload

Sample mode writes `synthetic/samples.pt` as a dictionary:

| Field | Meaning |
|---|---|
| `samples` | Generated `[N, C, L]` tensor. |
| `fault_label` | Per-sample fault labels used for conditioning. |
| `domain_id` | Per-sample domain IDs used for conditioning. |
| `condition_policy` | Policy used to produce label/domain pairs. |
| `condition_counts` | Counts keyed as `fault=<label>,domain=<domain>`. |
| `num_steps` | Sampler steps / NFE. |
| `sampler_id` | Sampler implementation ID, for example `euler_ode`. |
| `sampler_metadata` | Sampler-specific metadata emitted by the task. |

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

Benchmark-valid requests are downgraded to `exploratory` instead of being
accepted when any required evidence is missing. Missing fields are recorded in
`validity.missing_evidence` and summarized in `validity.reason`.

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

## Validity Checklist

Before treating generated samples as benchmark-valid, verify:

- `source_data.source_split` is `train`
- `protocol.protocol_hash` is present and not `unspecified`
- `config.config_hash` is present and not `unspecified`
- `environment.dependency_lock_hash` is present and not `missing`
- normalization params artifact and hash are recorded
- leakage split guard passed
- nearest-neighbor leakage check passed
- condition sampling policy is recorded
- condition counts are non-empty
- train-distribution sampling has explicit metadata split evidence when used
- metric status reasons are recorded for not-computable metrics

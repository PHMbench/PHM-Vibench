# Selective diagnosis on explicit features

The operators implement the existing normalized Gaussian-rule certificate and
finite-family calibration protocol. `Fuzzy` remains in Model Factory; this
package owns certificates, unit-balanced estimators, source-only model search,
selective objectives and feature-track execution. It does not infer physical
unit identity or upstream feature provenance.

Use the public research command with
`configs/experiments/p05/selective_diagnosis.yaml` and explicit `--data` and
`--output`. Supported phases are `export`, `sanity`, `capacity`, `tune`, `fit`,
`overfit`, `predict`, `summarize` and `plot`. The `predict` checkpoint must be the
exact saved joint-policy NPZ (`model_<seed>.npz`), supplied with `--checkpoint`.
It contains normalization, rule parameters, rule costs and the issued threshold;
raw Torch training checkpoints alone do not define a selective policy.

`fit` accepts an explicit source-only `--selection` JSON from `tune`. The search
space resides in `task.selective_diagnosis.search` and uses train/tune values
only. A `completed` software search is not a baseline-valid result. The generic
waveform pipeline does not execute this feature-archive protocol.

For `summarize` and `plot`, `--data` denotes the saved run directory. Synthetic,
tune-only and ineligible runs require
`--override task.selective_diagnosis.allow_diagnostic=true`. Failed or partial
runs remain rejected. Risk differences are **control minus joint**.

The feature NPZ contains `x`, `y`, `unit`, `split`, `domain`, `feature_names` and
`kind`. Splits are train/tune/cal/test and no declared unit can cross them.
Normalization uses training units. These checks cannot establish genuine
independence or physical meaning of the declared IDs. `capacity` returns exit
code 2 when the existing bound is infeasible; this is not an observed diagnostic
failure. All-abstain conditional risk and unattainable matched risk remain
undefined. Matched coverage is descriptive fractional-boundary ranking, not
the deployed calibrated rule.

The certificate concerns persistence of the original nominal class and
acceptance within a declared standardized-feature set. It does not certify
correctness or physical robustness. Real-data qualification and comparative
scientific conclusions are separate from software execution.

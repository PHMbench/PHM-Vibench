# P01: TSPN-anchored operator residual, competence and measured gain

The new opt-in `TSPN_fusion` head `operator_residual` retains the complete
calibrated TSPN logits and adds separate bounded nonlinear functions of the
reference-feature and physical-operator branches. Final branch matrices start
at zero. It therefore starts exactly at the reference, not at a random new
classifier. Each branch has zero output at zero features and class-centered
outputs. Their exported `logit_contribution__<branch>` arrays sum to the direct
candidate's log-odds correction. This is branch-level computational attribution,
not a unique physical cause or a univariate additive-model interpretation.
An intermediate probability mixture is NOT additive in these logit terms.

The existing envelope branch optionally computes regularized kurtosis,
envelope coefficient-of-variation squared, signed Teager energy ratio, and
non-wrapping lag correlations from its existing filtered waveforms. Frequencies,
lags, boundary conventions and epsilon must be declared from actual source
observations. These classical operators are not new physical discoveries. The
old branch with no `diagnostics` and old linear/MLP head keys remain unchanged.

The existing source trainer, feature trace and source prediction export support
the mode. Its exact zero-residual initialization is retained as checkpoint -1,
so source selection cannot be worse than that incumbent under the declared
composite score. This does not guarantee source accuracy or test performance.
`--reference-min-accuracy 0.8` adds an explicit source qualification gate before
candidate optimization. Failed qualification is saved, not retried on another
test. It is development evidence, not an independent confidence statement.

`python -m experiments.p01.diagnostic_gain --help` describes a saved-prediction
report. It separates development qualification from independent confirmation.
Confirmation allocates error to reference accuracy, paired accuracy gain and
paired Brier excess for every reported candidate and condition, using the
existing radius. A gain statement requires all declared bounds to pass; weak
sample sizes may leave it unresolved. No rule, predictor or test set is chosen
by the report. The assumptions are explicit declarations, not inferred from an
array. Datasets require separately budgeted errors when combined.

Implementation and configuration examples are not industrial results. PU D1/G07,
its original losses, checkpoints, decisions and sealed test specimens are not
changed. The older 16-fit follow-up attachment was not merged and is not quietly
activated here. New data/physical groups and a prospective protocol are required.

# Shared TSPN paper configurations and source-only DG diagnostics

The existing `X_model/TSPN` supports explicit connection parameterization,
per-feature mixing, feature definitions and classifier topology. Defaults retain
the existing TSPN parameter names and behavior; paper names do not create duplicate
model implementations. New absolute-value negative entropy and population-moment
kurtosis are explicit options, not silent changes to existing checkpoints.
Odd-length wave filters preserve input length, and running normalization statistics
no longer retain training autograd graphs.

The DG runner accepts an independently trained TSPN–TON configuration with native
CE and the common downstream source selector. Its declared adaptation is recorded
with checkpoints and qualification; undocumented mappings remain ineligible for
formal comparison. All families share the frozen twelve-trial source-only grid.

Freeze measures complete prediction latency on the same cached source batch,
including configured operators and readout. Full fusion includes its reference;
standalone baseline latency excludes the reference used only for paired analysis.
Analysis exports deterministic correctness-stratified cases, same-forward feature
descriptors and signed class contrasts. Descriptor arrays follow the same identity
ordering as predictions. These outputs neither admit a physical dataset nor prove
baseline competitiveness; real-data qualification is still required.

Focused regression coverage: `test_tspn_paper_configs.py`,
`test_p01_source_classifier.py`, `test_p01_multiview_dg.py`,
`test_p01_dg_stages.py`, and `test_p01_diagnostic_analysis.py`.

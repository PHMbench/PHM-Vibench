# M0 native integration: PASS (constructed fixtures)

The authoritative implementation is in PHMFactory. Component checks pass for
matched S/U branch shapes and execution, common-only null updates, incremental
input organization, unavailable-input isolation and zero input gradients.
`M_01_ISFM.encode` bypasses source heads and metadata; native HSE accepts explicit
rates and patch starts. Target readout and episode code reject invalid identities
before grouping and retain numeric zero. Failed fixed draws are not redrawn.

Native Data → Model → Task → Trainer public CLI runs completed 20 joint rounds
per arm, seed 0, CPU, two analytic sine sources. Both sources update the same
encoder; selected-head tests preserve unselected AdamW parameters and state.
Source RMS and validation use equal dataset/group/window mass and disjoint
source-train/validation groups. Full saved-state recovery, fixed sampler rounds,
tokens, features, logits, RMS and optimizer states passed. Raw source-validation
fixture predictions and both actual checkpoints are preserved.

The saved runs preceded a discovered float32 validation-monitor precision defect.
The final code computes validation NLL in float64 and logs a float64 monitor;
seven focused checkpoint tests pass, including the actual Task-to-callback path
at the fixed 1e-8 threshold. Existing fixture runs were not relabeled as evidence
of that precision fix or silently replaced. Fractional labels/channels now fail
before H5 projection; two runtime iterations retain the source_train RMS request.

Seven TII test files have 52 passing tests in LQ_signal (executed individually or
in focused groups; see commands.md), plus the repeated-iteration regression.
Query pollution tests reject all prescribed cases before NLL. Physical projection
checks establish fixed-duration Fourier projection and unit conversion on fixtures.

J1-C real two-qualified-source execution: NOT RUN. Qualification found zero
qualified corpora and zero eligible folds. Thus M0 software evidence does not
establish J1 Go or support any industrial performance claim. J2/N/H/J3/J3-R/J4,
query bootstrap, episode sensitivity and result figures were NOT RUN.

Final repository-wide pytest attempt: FAIL during collection because LQ_signal
has no streamlit. No dependencies were installed. Focused relevant checks and
remote CI are reported separately; no full-suite pass is claimed.

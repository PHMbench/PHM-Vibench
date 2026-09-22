# TII: distinguish source fitting from transfer feasibility

Move the existing paper-side PU observation-provenance execution instructions into
`configs/experiments/tii/DATA_FEASIBILITY.md`, alongside the maintained TII runtime
instructions. The paper retains a link, not another execution body. This is the
support-conditioned-tokenization study, not the separate TSPN G08 in experiments/p01.

Keep the same PU records, unchanged episode rule and zero-training inspection budget.
Separate two-qualified-source fitting from three-dataset held-out transfer and full
four-cell pooling. Require a nonempty, statically feasible inner-alpha task set before
any encoder fit. No acquisition rule, runtime input, old report or model is changed.
The existing fit95 command remains the only bounded source-fitting follow-up.

Clarify planned group statistics and retire the unsupported 0.005-nat equivalence
threshold. These are future-analysis specifications, not newly executed evaluator
features or empirical results. Missing raw evidence still blocks the relevant stage.

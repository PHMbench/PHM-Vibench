# D1 diagnostic previews

These are exploratory visual summaries of the frozen analysis, not publication-validated figures. No target-journal specification has been applied. No significance stars or ranking selection is used.

Source analysis directory: `/tmp/p01-d1-20260919-1irHPs/artifacts/d1_pu_20260919/analysis_test`. Partition: `test`. Population: source: conditions 0, 2, 3; unseen: conditions 1.

Losses average windows within acquisitions, acquisitions equally within physical groups, and physical groups equally within conditions; named populations average their conditions equally. Plotting reads saved tables only and does not recompute metrics or bootstrap intervals.

## main_contrasts

Sources: `contrast_seed_summary.csv` and `paired_contrasts.csv`. Each marker is the saved finite three-seed mean of the labeled treatment-minus-control contrast; negative Brier or CE favors the treatment. Thin horizontal intervals are descriptive 95% paired global-physical-group bootstrap intervals (2,000 draws, analysis seed 20260919, stratified by condition-incidence mask, shared resampling across arms and training seeds). Small gray points are individual training-seed estimates, not independent physical specimens. The zero line denotes no contrast. Intervals are conditional on frozen predictors and can be unstable with few groups. Delta_pipe is a processing-pipeline contrast, not an isolated representation-causality claim.

## seed_dispersion

Source: `seed_summary.csv`, direct candidates only. Markers and bars are the saved mean ± sample SD across fixed training seeds 42, 123, and 456. SD describes training-seed dispersion; it is not a confidence interval and the seeds are not independent specimens.

## adoption_mechanism

Source: `mechanism.csv`, actual adopted role and deployed predictor only. Panels show saved A, sqrt(A), b/sqrt(A), alpha, and alpha sqrt(A) for the selected candidate relative to the frozen reference. They are point estimates without inferred error bars. When A = 0, b/sqrt(A) is undefined and omitted explicitly; alpha = 0 remains zero. Selection and coefficient are already frozen; this figure selects neither.

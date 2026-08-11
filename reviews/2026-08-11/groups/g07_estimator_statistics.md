# g07_estimator_statistics — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Metric estimators, label ontology, population weighting, statistical claim boundaries
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific claim
- P0: target-domain requests can be rewritten.
- P1: every maintained demo remains `smoke_only`.
- No accuracy value is benchmark evidence until population, unit, aggregation, checkpoint and seed are frozen.

## R02 — Configuration
- P0: missing pipeline becomes Pipeline 01.
- Statistical estimator settings must not be supplied by hidden defaults.
- Correction: canonical config declares evaluation unit, aggregation and checkpoint selection.

## R03 — Data population
- P0: NaN/-1 labels are silently filtered.
- Impact: test population and class prevalence change without a recorded protocol decision.
- Correction: invalid supervised labels fail with IDs; exclusions must be explicit.

## R04 — Split estimator
- P0: impossible target-domain count is clamped.
- Target test data must be excluded from training, validation and checkpoint selection.
- Correction: observed split facts must satisfy explicit protocol requirements.

## R05 — Stochastic model
- P0: HSE eval uses random patches.
- A single prediction is not a stable estimator under current behavior.
- Correction: deterministic eval for the first baseline.

## R06 — Metric implementation
- P0: regularization omits first parameter.
- P0: unknown metric is warned and skipped.
- P1: current logging does not explicitly prove sample-level global micro aggregation.

## R07 — Evaluation lifecycle
- P0: Pipeline 02 can return success with empty metrics.
- An estimator is undefined when evaluation failed.
- Correction: success requires finite non-empty metric mapping.

## R08 — Decoupling
- P0: metric dataset identity uses only first file ID in a batch.
- Impact: mixed-dataset observations can be assigned to the wrong population.
- Correction: dataset-homogeneous batches or explicit mixed-dataset task semantics.

## R09 — User result
- P1: several result files/manifests compete as authority.
- User result should expose: metric, unit, aggregation, `N_test`, class counts, checkpoint rule, seed.
- No artifact audit is needed for estimator validity.

## R10 — Data Factory
- P0: `drop_last=True` can remove all samples from a small system.
- Window-level micro weights files by their window counts.
- Correction: report per-file/system/window counts and ensure selected systems are represented.

## R11 — Model Factory
- P0: class count is inferred as `max(Label)+1` without verifying labels are exactly `0..K-1`.
- Impact: head and metric dimensions may be wrong.
- Correction: validate ontology or require explicit mapping; never silently recode.

## R12 — Task Factory
- P0: metric data name comes from first file ID.
- P0: unknown metric and regularization names may disappear.
- Correction: exact metric/objective identity and explicit aggregation tests.

## R13 — Trainer Factory
- P0: Pipeline 02 wrong-success invalidates any stage-level statistic.
- Trainer must restore the declared best checkpoint before test and return sample counts with metrics.
- Empty, NaN or Inf metrics fail.

## R14 — Group meta-review
- First estimator recommendation:
  `window-level accuracy, global micro, deterministic evaluation, best validation-loss checkpoint, one declared seed`.
- Claim must be exactly “window-level held-out-domain accuracy,” not file-, bearing-, machine- or repeated-seed performance.
- File-macro and multi-seed estimates are later benchmark-candidate work, not prerequisites for semantic baseline validity.

# Estimator contract

\[
\widehat{Acc}_{micro}=\frac{\sum_i \mathbf 1[\hat y_i=y_i]}{N_{test}}
\]

Required accompanying facts:

```text
test file count
test window count
per-class counts
domain IDs
checkpoint selection rule
seed
deterministic prediction check
```

# Stop condition

A result may be promoted to `baseline_valid` only when its reported number equals the declared estimator over the declared test population, and the claim does not exceed its unit, aggregation or repetition design.

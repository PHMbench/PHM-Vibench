/goal

## Goal ID
GOAL-GEN-M2-002-MULTIDATASET-AGGREGATION

## Objective

Extend benchmark-effect aggregation to report quality, utility, efficiency,
leakage, missing reasons, and source paths across multiple datasets.

## Scope

Allowed:

- `scripts/generative_benchmark_effect.py`
- `scripts/paperpack_generative.py` only for paperpack table/figure-source
  aggregation gaps.
- Focused tests under `test/generative/`.

Out of scope:

- Do not change metric formulas unless a metric-specific goal requests it.
- Do not hide missing utility metrics.

## Required Behavior

- Before implementation, confirm active feature directory
  `specs/002-phm-genbench-frontier/` exists.
- This goal's real-evidence completion is task `T048` and depends on `T047`
  completing `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE`; fixture aggregation is not a
  substitute for real six-dataset run directories.
- Read and update `specs/002-phm-genbench-frontier/spec.md`, `plan.md`, and
  `tasks.md` if this goal changes requirements, design, or task sequencing.
- Record aggregation design notes and validation results under
  `specs/002-phm-genbench-frontier/reviews/codex/`.
- Summary rows are grouped by `dataset / method / metric`.
- Baseline deltas are computed per dataset and metric.
- Utility metrics may be missing for invalid label/domain protocols, but every
  missing value must carry a reason.
- Rows are exploratory unless all contributing manifests are benchmark-valid.
- Benchmark-effect manifest must record configured dataset count, observed
  datasets/count, observed configured datasets/count, missing configured
  datasets, unexpected observed datasets, `min_datasets`, `min_datasets_met`,
  and `input_gaps`.
- `min_datasets_met` must be computed from observed configured datasets, not
  total observed datasets, so matrix-external datasets cannot satisfy the
  paper minimum.
- Missing configured datasets and unexpected observed datasets must create
  machine-readable `input_gaps`.
- Keep generated benchmark summaries in `results/` or paperpack outputs; keep
  process decisions and review notes under `specs/002-phm-genbench-frontier/`.
- If aggregation contracts need durable public documentation, update
  `scripts/README.md`, `configs/paper/phm_generative/README.md`, or the
  relevant metric/component README. Do not create `docs/phm_generative/` or
  `docs/generative/`.

## Acceptance Criteria

- Six-dataset fixture aggregation produces a summary CSV, report, manifest, and
  missing-metrics appendix.
- Real-evidence completion requires aggregating the actual
  `results/paper/phm_generative/six_dataset_submission_v1/runs` directory after
  M2-003 succeeds.
- Benchmark-valid aggregate rows retain nonempty metric and manifest source
  paths.
- Fixture tests cover complete six-dataset evidence, missing configured
  dataset evidence, and unexpected observed dataset evidence.
- The active feature analysis records whether this goal changed aggregation
  contracts used by later figure and paper-draft goals.

## Validation Commands

```bash
python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q
```

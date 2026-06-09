# M2 Cross-Artifact Analysis

## Finding

The M2 goal contracts correctly describe six-dataset benchmark delivery, but
they must also encode where development-process artifacts live. Without that
rule, Claude task specs, handoffs, validation notes, blocked reviews, and paper
readiness logs drift into `.codex/`, `.claude/`, or durable docs without a
single Speckit feature index.

## Decision

`specs/002-phm-genbench-frontier/` is the canonical process-artifact home for
M2. Tool-private directories may exist only as scratch or mirrors.

## Required Alignment

- `.specify/goals/v2/` remains the goal queue.
- `specs/002-phm-genbench-frontier/m2/` indexes the M2 queue.
- `specs/002-phm-genbench-frontier/reviews/` records Claude and Codex review
  artifacts.
- `specs/002-phm-genbench-frontier/handoffs/` records session continuity notes.
- `specs/002-phm-genbench-frontier/paper/` stores working paper draft,
  readiness, and evidence-gap notes.
- Module READMEs store durable public guidance. `docs/` remains a project-level
  index only.

## Review Result

M2 goals must be revised so the Speckit artifact location and acceleration
workflow are explicit Required Behavior and Acceptance Criteria, not informal
operator notes.

## M2-004 Table And Figure Readiness

The paperpack contract is scaffold-covered but not real-evidence-ready.
`scripts/paperpack_generative.py` and `test/generative/test_paperpack_generative.py`
cover the required table and figure-source shapes:

- quality, utility, efficiency, leakage, and ablation tables
- spectra and temporal overlays
- metric barplot sources
- dataset-method heatmap sources
- missing-metric audit sources
- manifest index with synthetic manifest paths and metric source paths
- run index, manifest completeness, and missing-metric appendix

The focused test verifies source-path propagation into tables and figure
sources, verifies manifest/metric source indexing, and verifies missing metric
reasons. These artifacts are ready for the Markdown draft only after M2-003
creates real run directories and M2-002 aggregates them. Until then, M2-004
remains scaffold-covered and evidence blocked.

## M2-002 Aggregation Contract Impact

M2-002 changes the contract consumed by M2-004 figures/tables and M2-005 paper
draft readiness. Benchmark-effect aggregation must emit:

- `benchmark_effect_summary.csv`
- `benchmark_effect_report.md`
- `benchmark_effect_manifest.json`
- `missing_metrics.md`

The summary rows are grouped by `dataset / method / metric` and retain
`manifest_paths` plus `metric_source_paths` for benchmark-valid rows. Baseline
deltas are computed per dataset and metric against `baseline_method`.

The benchmark-effect manifest records:

- `configured_dataset_count`
- `observed_datasets`
- `observed_dataset_count`
- `observed_configured_datasets`
- `observed_configured_dataset_count`
- `missing_datasets`
- `unexpected_datasets`
- `min_datasets`
- `min_datasets_met`
- `input_gaps`

This means M2-004 may generate paperpack tables and figure sources only from
traceable summary rows, and M2-005 may mark a draft `SUBMISSION_READY` only
when the manifest has `min_datasets_met: true`, no missing or unexpected
datasets, no `input_gaps`, and at least six observed configured datasets.
Matrix-external datasets cannot satisfy the six-dataset paper claim.

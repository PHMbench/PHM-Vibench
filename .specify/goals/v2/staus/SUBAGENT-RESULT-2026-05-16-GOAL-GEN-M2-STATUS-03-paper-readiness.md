# Subagent Result 03: Paper Readiness

**Date**: 2026-05-16
**Mode**: read-only advisory analysis
**Scope**: M2-004, M2-005, paper draft sidecars
**Mutation**: none

## Status

`NOT_SUBMISSION_READY`

## Issues

- M2-003 real six-dataset evidence is still blocked.
- `results/paper/phm_generative/six_dataset_submission_v1/runs` is absent.
- `specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md` states that no
  numerical benchmark claim should be treated as a result.
- `specs/002-phm-genbench-frontier/paper/submission_readiness.md` remains
  `NOT_SUBMISSION_READY`.
- `specs/002-phm-genbench-frontier/paper/evidence_gaps.md` reports missing
  effect summary and manifest.
- Paperpack scaffolding exists, but real paperpack artifacts from six-dataset
  runs are not available.

## Missing Evidence

| Missing evidence | Expected path or condition |
| --- | --- |
| Real run directories | `results/paper/phm_generative/six_dataset_submission_v1/runs` |
| Effect summary | `results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv` |
| Effect manifest | `results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json` |
| Six-dataset coverage | `observed_configured_dataset_count >= min_datasets` |
| Manifest readiness | `min_datasets_met: true`, no missing/unexpected datasets, no `input_gaps` |
| Figure/table sources | real CSV/source artifacts traceable to metric source paths |

## Downstream Dependencies

- M2-004 depends on M2-003 real runs and M2-002 aggregation before figures and
  tables can be paper-grade.
- M2-005 depends on M2-003 and M2-004 evidence before the draft can become
  submission-ready.
- M2-006 advisory review remains downstream of real evidence and endpoint
  approval.

## No-Fabrication Constraints

- Do not fabricate numerical values, figures, tables, or benchmark claims.
- Do not mark demos, dry-runs, blocked ledgers, or GPU preflight failures as
  benchmark-valid evidence.
- Do not promote the draft to `SUBMISSION_READY` until real six-dataset
  evidence exists.
- Do not use CPU fallback for M2 paper benchmark evidence under the current
  goal contract.

## Decomposition After M2-003

1. Restore GPU 6/7 CUDA visibility and rerun M2-003 preflight.
2. Execute staged M2-003 runs.
3. Aggregate real runs through M2-002.
4. Regenerate M2-004 paperpack tables and figure sources.
5. Regenerate M2-005 draft with `--require-submission-ready`.
6. Run M2-006 advisory review only after real evidence and endpoint approval.

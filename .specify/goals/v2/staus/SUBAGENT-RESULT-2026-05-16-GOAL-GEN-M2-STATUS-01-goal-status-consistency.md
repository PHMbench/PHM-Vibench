# Subagent Result 01: Goal Status Consistency

**Date**: 2026-05-16
**Mode**: read-only advisory analysis
**Scope**: `.specify/goals/v2/`, `.specify/goals/v2/staus/`,
`specs/002-phm-genbench-frontier/`
**Mutation**: none

## Problems

| Severity | Problem | Evidence |
| --- | --- | --- |
| High | M2 goal IDs are not normalized across ledgers. Some status rows use short IDs such as `GOAL-GEN-M2-003`, while contracts and M2 execution status use canonical full IDs such as `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE`. | `.specify/goals/v2/GOAL-GEN-M2-003-real-runs-evidence.md`, `.specify/goals/v2/staus/STATUS-2026-05-16.md`, `specs/002-phm-genbench-frontier/m2/execution-status.md` |
| High | Completion labels mix artifact readiness with benchmark completion. `COVERED`, `SCAFFOLD_COVERED`, and `BLOCKED` are used to describe different dimensions. | `.specify/goals/v2/staus/STATUS-2026-05-16.md`, `specs/002-phm-genbench-frontier/m2/execution-status.md` |
| High | `tasks.md` has only T047 open, but downstream real-evidence work remains for M2-002, M2-004, M2-005, and M2-006 after M2-003 unblocks. | `specs/002-phm-genbench-frontier/tasks.md` |
| Medium | Executive counts are numerically correct but can hide downstream incompleteness. | `.specify/goals/v2/staus/STATUS-2026-05-16.md` |
| Medium | May 12 and May 16 status snapshots use different label taxonomies, which makes trend comparison harder. | `.specify/goals/v2/staus/STATUS-2026-05-12.md`, `.specify/goals/v2/staus/STATUS-2026-05-16.md` |
| Low | The status directory spelling `staus` is preserved intentionally, but remains typo-prone for future scripts. | `.specify/goals/v2/staus/` |

## Recommended Decomposition

Use separate fields in future status ledgers:

| Field | Suggested values |
| --- | --- |
| `artifact_status` | `complete`, `scaffold_complete`, `missing` |
| `evidence_status` | `benchmark_complete`, `blocked_gpu_preflight`, `pending_real_runs`, `pending_aggregation`, `pending_review` |

Suggested post-M2-003 task split:

| New task | Maps to | Done condition |
| --- | --- | --- |
| `T048` | M2-002 | Aggregate real six-dataset run directories and write effect summary/manifest. |
| `T049` | M2-004 | Generate final tables and figure sources from real evidence. |
| `T050` | M2-005 | Regenerate the paper draft only after readiness gates pass. |
| `T051` | M2-006 | Run final Codex verification and advisory review. |

## Status Rule

Use canonical full goal IDs in machine-oriented status ledgers. Short IDs can
remain display aliases only.

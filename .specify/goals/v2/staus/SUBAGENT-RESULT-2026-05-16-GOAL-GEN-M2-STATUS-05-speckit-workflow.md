# Subagent Result 05: SpecKit Workflow

**Date**: 2026-05-16
**Mode**: read-only advisory analysis
**Scope**: SpecKit artifacts, M2 goals, active constitution
**Mutation**: none

## Workflow Order Assessment

Expected governance order:

```text
constitution -> specify -> clarify -> plan -> checklist -> tasks -> analyze -> implement
```

| Stage | Reflected | Evidence | Status |
| --- | --- | --- | --- |
| Constitution | yes | `.specify/memory/constitution.md` | complete |
| Specify | yes | `specs/002-phm-genbench-frontier/spec.md` | complete |
| Clarify | partial | no separate clarification artifact found; checklists have no open clarification markers | implicitly resolved |
| Plan | yes | `specs/002-phm-genbench-frontier/plan.md` | complete |
| Checklist | yes | `checklists/requirements.md`, `checklists/benchmark-readiness.md` | complete |
| Tasks | yes | `specs/002-phm-genbench-frontier/tasks.md` | complete except T047-T051 |
| Analyze | yes | `analysis/m2-cross-artifact-analysis.md` | complete |
| Implement | mostly | T001-T046 checked, T047-T051 open | blocked on real GPU evidence chain |

## Workflow Mismatch

The full constitution order is represented in artifacts and governance, but the
workflow YAML only encodes the narrower implementation path:

```text
specify -> review-spec -> plan -> review-plan -> tasks -> implement
```

This is a process gap, not the active benchmark blocker.

## Current Goal Decomposition

| Level | Goals |
| --- | --- |
| `COMPLETE` | `GOAL-GEN-000`, `GOAL-GEN-001`, `GOAL-GEN-002`, `GOAL-GEN-003`, `GOAL-GEN-004`, `GOAL-GEN-M2-000-SPECKIT-FREEZE` |
| `COVERED` | `GOAL-GEN-M1-REPO-NATIVE`, `GOAL-GEN-M2-001-SIX-DATASET-MATRIX-GPU`, `GOAL-GEN-M2-002-MULTIDATASET-AGGREGATION`, partial `GOAL-GEN-M2-006-REVIEW-HANDOFF` |
| `SCAFFOLD_COVERED` | `GOAL-GEN-M2-004-FIGURES-TABLES`, `GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT` |
| `BLOCKED` | `GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE` |
| `DOWNSTREAM_BLOCKED` | `GOAL-GEN-M2-004-FIGURES-TABLES`, `GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT`, real-review part of `GOAL-GEN-M2-006-REVIEW-HANDOFF` |
| `NOT_READY` | submission paper package and benchmark-valid six-dataset claims |

## Next Breakpoints

1. Decide whether to encode `clarify`, `checklist`, and `analyze` as explicit
   workflow YAML steps or keep them as constitution-governed side artifacts.
2. Restore CUDA visibility for GPU 6 and GPU 7.
3. Complete T047 with staged `train -> sample -> eval -> paperpack`.
4. Run real aggregation for M2-002.
5. Promote paper outputs only after real evidence exists.
6. Run advisory Claude review only after endpoint approval.

## Status Summary

The SpecKit decomposition is coherent at artifact level. The remaining gap is
benchmark evidence execution, not planning structure.

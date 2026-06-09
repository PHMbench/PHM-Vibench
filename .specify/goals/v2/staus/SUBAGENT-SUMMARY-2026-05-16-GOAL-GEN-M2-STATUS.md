# Subagent Summary: GOAL-GEN M2 Status

**Date**: 2026-05-16
**Status directory**: `.specify/goals/v2/staus/`
**Inputs**: six read-only subagent analyses
**Codex role**: synthesize advisory findings into a status decomposition
**Mutation scope**: status files only

## Subagent Result Index

| Subagent | Scope | Result file | Status |
| --- | --- | --- | --- |
| 01 | Goal/status consistency | `SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-01-goal-status-consistency.md` | COMPLETE |
| 02 | GPU run evidence | `SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-02-gpu-run-evidence.md` | COMPLETE |
| 03 | Paper readiness | `SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-03-paper-readiness.md` | COMPLETE |
| 04 | Validation guardrails | `SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-04-validation-guardrails.md` | COMPLETE |
| 05 | SpecKit workflow | `SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-05-speckit-workflow.md` | COMPLETE |
| 06 | Handoff/team review | `SUBAGENT-RESULT-2026-05-16-GOAL-GEN-M2-STATUS-06-handoff-team-review.md` | COMPLETE |

These results are advisory. Codex remains responsible for final status wording,
validation, and deciding whether a goal can be marked complete.

Supersession note: the subagent analyses were produced before the elevated
`nvidia-modprobe -u -c=0` preflight and partial train execution. The current
canonical status is in `STATUS-2026-05-16.md`: GPU 6/7 preflight passes in the
elevated context, `runs/` now exists with partial train evidence, and the root
blocker is missing complete train/sample/eval/paperpack evidence rather than
missing run directories.

## Consolidated Problem Tree

```text
Level 0: Submission package is not ready.
  Level 1: GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE is incomplete.
    Level 2: GPU 6/7 preflight passes only in the elevated execution context.
      Level 3: Real six-dataset run directories contain partial train evidence only.
        Level 4: Aggregation, figures/tables, paper draft, and review remain downstream blocked.
```

## Current Root Blocker

`GOAL-GEN-M2-003-REAL-RUNS-EVIDENCE` is incomplete. The current evidence
records:

- Default sandboxed GPU probes cannot communicate with the NVIDIA driver.
- Elevated `nvidia-modprobe -u -c=0` GPU 6/7 preflight passes.
- `results/paper/phm_generative/six_dataset_submission_v1/runs` exists with
  8 train completion sidecars, 9 checkpoints, and 6 manifest files.
- `samples.pt`, `generative_eval_metrics.csv`, and paperpack
  `manifest_index.json` artifacts are absent.
- The paper draft remains `NOT_SUBMISSION_READY`.

## Downstream Impact

| Goal | Artifact status | Evidence status | Current action |
| --- | --- | --- | --- |
| `GOAL-GEN-M2-001-SIX-DATASET-MATRIX-GPU` | complete | blocked by GPU preflight | keep dry-run evidence, rerun after GPU fix |
| `GOAL-GEN-M2-002-MULTIDATASET-AGGREGATION` | scaffold complete | pending real runs | aggregate only after M2-003 creates real run dirs |
| `GOAL-GEN-M2-004-FIGURES-TABLES` | scaffold complete | downstream blocked | do not generate final claims from fixtures |
| `GOAL-GEN-M2-005-MARKDOWN-PAPER-DRAFT` | scaffold complete | not submission-ready | keep draft sidecars `NOT_SUBMISSION_READY` |
| `GOAL-GEN-M2-006-REVIEW-HANDOFF` | covered | advisory review blocked | run only after endpoint approval and real evidence |

## Governance Issues To Track

| Issue | Severity | Recommended handling |
| --- | --- | --- |
| Short goal IDs in status ledgers | High | Use canonical full IDs in future status rows. |
| Mixed status labels | High | Split `artifact_status` from `evidence_status`. |
| Downstream work hidden behind T047 | High | Split into open T048-T051 in `tasks.md` for real aggregation, figures, draft, and review. |
| Latest-known pytest counts | Medium | Mark as latest-known unless rerun in the current pass. |
| `staus` directory spelling | Low | Preserve path for user compatibility; document it explicitly. |
| Claude Teams endpoint approval | Medium | Keep Claude review `BLOCKED_NOT_RUN` until approval is explicit. |

## Recommended Status Levels

| Level | Meaning |
| --- | --- |
| `COMPLETE` | Contract satisfied with no known dependency blocker. |
| `COVERED` | Structure, docs, tests, or dry-run path exist, but final real evidence is not present. |
| `SCAFFOLD_COVERED` | Output shape or implementation shell exists; production evidence is absent. |
| `BLOCKED` | Cannot proceed in the current environment or without external approval. |
| `DOWNSTREAM_BLOCKED` | Technically ready but depends on a blocked upstream goal. |
| `NOT_READY` | Must not be used for submission or benchmark-valid claims. |

## Task Split After GPU Unblock

| Task | Goal | Done condition |
| --- | --- | --- |
| `T047` | M2-003 | Real staged train/sample/eval/paperpack completes for the six-dataset matrix. |
| `T048` | M2-002 | Real run directories are aggregated into effect summary and manifest. |
| `T049` | M2-004 | Figures and tables are generated from traceable real evidence. |
| `T050` | M2-005 | Paper draft passes readiness gates with real evidence. |
| `T051` | M2-006 | Codex verification and advisory review run against the completed evidence package. |

## Current Validation Interpretation

- `python -m scripts.validate_docs` and `python -m scripts.validate_configs`
  are the current low-cost structural gates to rerun after status edits.
- Pytest counts in the status package are latest-known unless explicitly rerun.
- `results/` freshness is not fully guaranteed by doc scanning.
- A missing `runs/` directory is the expected blocked state, not a passing
  benchmark state.

## Decision

Do not mark the active M2 evidence chain complete. Continue to report the
paper package as `NOT_SUBMISSION_READY` until GPU-backed real six-dataset
evidence exists and downstream aggregation, figures, draft generation, and
review consume that evidence.

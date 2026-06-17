# PHM-GenBench v0.3 Closure Backlog

Date: 2026-06-10

## Decision

The code gate is `READY_FOR_REAL_RUN`.

## Blocking Backlog

No GOAL-V3-001 through GOAL-V3-006 blocker remains.

## Next Required Goal

Use `GOAL-V3-008-REAL-SIX-DATASET-RUN` from
`.specify/goals/v3/phm_generative_paper_update_pack_v0_3/00goal.md`.

Do not start `GOAL-V3-009-PAPER-EVIDENCE-PACKAGE` until V3-008 produces real
run outputs.

## Guardrails

- Do not use dry-run output as paper evidence.
- Do not mark submission draft `SUBMISSION_READY` without real benchmark-valid
  quality and utility rows.
- Do not include exploratory one-step methods in benchmark-valid main tables.

# PHM-GenBench v0.3 Blocking Backlog

Date: 2026-06-10

## Current Reviewer-Gate Blockers

No remaining GOAL-V3-001 through GOAL-V3-006 code-side blockers are open.

## Execution Blocker For Next Goal

### GOAL-V3-008-REAL-SIX-DATASET-RUN

Issue: real six-dataset benchmark execution now requires a long-running job
context.

Evidence:
- The matrix dry-run succeeds and produces 144 planned stage commands.
- CUDA/data preflight passed in the long-running 2026-06-10 execution context.
- CWRU has complete train/sample/eval/paperpack chains for CFM, Rectified Flow,
  and DDPM across two seeds.
- XJTU CFM seed 0 has completed train and has a partial stage ledger.
- XJTU CFM seed 1 is still in train as of the latest recorded monitor snapshot.
- Current progress and row status are recorded at
  `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-progress.md`
  and
  `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv`.

Risk:
Dry-run, CWRU-only, or partial XJTU results could be mistaken for six-dataset
paper evidence if V3-008 is skipped or stopped early.

Required fix:
Let the remaining train/sample/eval/paperpack stages finish from the
long-running job context. Preserve every failed stage in the run status ledger.

Proposed `/goal`:

```md
/goal

## Goal ID
GOAL-V3-008-REAL-SIX-DATASET-RUN

## Objective
Execute the six-dataset CFM / Rectified Flow / DDPM train/sample/eval/paperpack
queue only after CUDA and dataset preflight pass.

## Acceptance criteria
- Every dataset/method/seed has a status row.
- Successful rows have stage_ledger.json, synthetic_data_manifest.json,
  eval_evidence_manifest.json, generative_eval_metrics.csv, and paperpack.
- Failed rows keep exact failure reasons.
- No `<experiment_name>` placeholder remains in executed artifact paths.
```

## Non-Blocking Follow-Up

### GOAL-V3-EVIDENCE-001-PRIMARY-METRIC-POLICY

Wire `primary_metric_policy.yaml` into benchmark-effect/submission gating so
primary metric omissions are explicit policy failures rather than only
appendix-level missing metric rows.

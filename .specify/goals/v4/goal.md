# PHM-GenBench V4 Goal Index

Date: 2026-06-18

## Operating Decision

V4 is a paper-first implementation pack.

The goal is to produce a high-quality, claim-safe PHM generative paper by
implementing every method in the locked V4 SOTA roster as repo-native code,
running focused evidence, and writing the paper around what is actually
verified.

Current evidence remains:

```text
benchmark_valid_row_count=0
current_result_status=exploratory
current_readiness=NOT_SUBMISSION_READY
```

This does not block a strong paper. It only blocks unsupported claims such as
SOTA superiority, benchmark winner, cross-domain generalization, or downstream
diagnosis improvement.

## Authority Files

The V4 authority files are:

```text
.specify/goals/v4/goal.md
.specify/goals/v4/goal_sota.md
.specify/goals/v4/paper_ready.md
.specify/goals/v4/handoff.md
.specify/goals/v4/reviewer.md
```

Rules:

- `goal.md` is the index and execution order.
- `goal_sota.md` owns method implementation goals.
- `paper_ready.md` owns the final paper package and claim boundary.
- `handoff.md` owns the six-subagent collaboration protocol.
- `reviewer.md` owns paper-quality review.
- A `GOAL-V4-*` body should appear in only one file.

## Non-Negotiable Rules

1. Keep the maintained entry path:

   ```bash
   python main.py --config <yaml> [--override key=value ...]
   ```

2. Preserve the five config blocks:

   ```text
   environment / data / model / task / trainer
   ```

3. Do not add new dependencies unless the active goal explicitly justifies
   them.
4. Do not copy external code unless license and provenance are recorded.
5. Every method in the V4 SOTA roster must receive a repo-native
   implementation goal. Literature-only coverage is not enough.
6. Do not hide a method placeholder behind a real method name.
7. Do not add heavy promotion gates before the first high-quality paper draft.
8. Do not claim method superiority unless later evidence supports it.
9. Every implementation handoff must follow `.specify/goals/v4/handoff.md`.
10. Every paper-quality review must follow `.specify/goals/v4/reviewer.md`.

## V4 SOTA Roster

The initial V4 roster is intentionally finite. New papers may be added only by
adding implementation goals to `goal_sota.md`.

| Method | V4 requirement | Primary source |
|---|---|---|
| Conditional Flow Matching | maintain repo-native baseline | Flow Matching family |
| Rectified Flow | maintain repo-native baseline | Rectified Flow family |
| DDPM epsilon | maintain repo-native baseline | DDPM family |
| Score SDE | implement faithful score-based SDE path | https://arxiv.org/abs/2011.13456 |
| MeanFlow | implement method-specific one-step objective | https://arxiv.org/abs/2505.13447 |
| Drifting | implement method-specific drifting objective | https://arxiv.org/abs/2602.04770 |
| Transition Flow Matching | implement transition-flow objective | https://arxiv.org/abs/2603.15689 |
| OT-NFM | implement torch-native OT neural flow map | https://arxiv.org/abs/2604.06413 |

## Six Subagent Lanes

V4 uses six xhigh subagent lanes. The lanes are goal owners, not extra approval
layers.

| Lane | Owner | Owns |
|---|---|---|
| S1 | Baseline Contract | CFM, RF, DDPM, shared schema and task/model contracts |
| S2 | Score SDE | Score SDE loss, sampler, config, tests, paper status |
| S3 | MeanFlow | MeanFlow loss, one-step sampler, config, tests, paper status |
| S4 | Drifting and TFM | Drifting and Transition Flow Matching implementations |
| S5 | OT-NFM | torch-native OT coupling, sampler, config, tests, paper status |
| S6 | Paper Integration | method matrix, draft, claim cleanup, reviewer pass |

## Execution Order

```text
Wave 0:
  GOAL-V4-SOTA-000-ROSTER-LOCK
  GOAL-V4-PAPER-001-PAPER-SCOPE

Wave 1, parallel after Wave 0:
  GOAL-V4-SOTA-101-BASELINE-CONTRACT
  GOAL-V4-SOTA-102-SCORE-SDE
  GOAL-V4-SOTA-103-MEANFLOW
  GOAL-V4-SOTA-104-DRIFTING
  GOAL-V4-SOTA-105-TRANSITION-FLOW-MATCHING
  GOAL-V4-SOTA-106-OT-NFM

Wave 2:
  GOAL-V4-SOTA-107-SOTA-SMOKE-MATRIX
  GOAL-V4-PAPER-002-EVIDENCE-PACK

Wave 3:
  GOAL-V4-PAPER-003-FINAL-DRAFT
  GOAL-V4-PAPER-004-REVIEWER-PASS
```

## Minimal Paper Checks

These checks exist to prevent false claims, not to delay the paper:

```text
method implemented: task/loss/sampler/config/test exist
method smoke evidence: focused smoke or preflight recorded
paper integration: method appears in method matrix and limitations
claim safety: no unsupported SOTA/performance/utility claim
```

Rows may remain exploratory. The paper can still be finalized if the title,
abstract, results, and limitations state the exploratory boundary clearly.

## Validation

After editing V4 goal files:

```bash
python -m scripts.validate_docs
rg -n "GOAL-V4-SOTA|GOAL-V4-PAPER|S1|S2|S3|S4|S5|S6" .specify/goals/v4/*.md
rg -n "literature escape|unimplemented roster method" .specify/goals/v4/*.md
```

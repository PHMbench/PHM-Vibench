# GOAL B — v0.2.0 Review Advisor

## Mission
Provide continuous review, advice, and scoring while Goal A executes repository convergence work. B owns quality control, risk review, decision critique, and release-standard alignment.

## Objective
For every A handoff, B must:

1. check whether A's facts are supported by evidence;
2. score the current state;
3. identify risks and over-broad changes;
4. propose a smaller or safer next move when needed;
5. protect the final release surface from workflow clutter;
6. keep the work aligned with a serious PHM research benchmark standard.

## Review workflows

### R1 Evidence review
Each claim needs command output, SHA/ref evidence, file path evidence, validation log, or an assumption label.

### R2 Safety review
Check archive-tag planning, separate handling of local and remote tips, file-level triage for active lines, and rollback points.

### R3 Product review
Check whether a new PHM researcher can run a smoke demo, understand configs, add components, and reproduce validation.

### R4 Scientific review
Check config-first stability, explicit validation, no silent fallback, and separation of paper-only material from core runtime.

### R5 Strategic review
Check whether the current direction lowers first-run friction, clarifies architecture, reduces noise, strengthens tests, and improves contribution flow.

## Decision labels
- `APPROVE`
- `APPROVE_WITH_CONDITION`
- `REQUEST_CHANGES`
- `BLOCKER`

## Scoring
- evidence quality: 20
- safety and reversibility: 20
- release focus: 15
- PHM user value: 15
- scientific reproducibility: 15
- maintainability: 10
- next-action clarity: 5

Below 75 means A must revise the plan. Below 60 means B should issue `BLOCKER`.

## Deliverables
Update `HANDOFF_TO_A.md` after every A handoff with decision, scores, requested changes, and next-step advice.

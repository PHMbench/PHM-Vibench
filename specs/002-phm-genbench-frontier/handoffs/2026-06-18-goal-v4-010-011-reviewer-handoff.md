# V4 Handoff: GOAL-V4-010/011 Reviewer Handoff

## Goal ID

- `GOAL-V4-010-REVIEWER-GATE`
- `GOAL-V4-011-TWO-AGENT-HANDOFF-PROTOCOL`

## Objective

Install the V4 reviewer gate and two-agent handoff protocol so every later V4
goal is reviewed against paper-claim safety, evidence traceability, method
fidelity, and benchmark-valid promotion risk before it can support paper work.

## Current State

GOAL-V4-010 and GOAL-V4-011 are structurally complete for the current process
scope. The repository still has no benchmark-valid paper evidence; the
six-dataset package remains exploratory and `NOT_SUBMISSION_READY`.

## Implementation Summary

The V4 handoff protocol now binds Agent A handoffs to
`.specify/goals/v4/reviewer.md`, requires canonical evidence-root and claim
boundary fields, requires reviewer axis scoring, records per-goal review report
paths, and prevents later-stage work after a `BLOCKING` review.

A V4 review report template was added under the feature-scoped review directory
so future goals can produce consistent machine-parseable review reports.

## Runtime Behavior Changed

No.

## Files Changed

- `.specify/goals/v4/handoff.md` - strengthened Agent A/Agent B duties,
  Builder Handoff fields, review report requirements, blocked-closure handling,
  required handoff file contents, and minimal review checklist.
- `specs/002-phm-genbench-frontier/reviews/v4/README.md` - added the V4
  per-goal reviewer report template and rules.
- `specs/002-phm-genbench-frontier/reviews/v4/2026-06-18-goal-v4-010-011-reviewer-handoff-review.md`
  - recorded the local V4 reviewer gate audit for this process goal.
- `specs/002-phm-genbench-frontier/handoffs/2026-06-18-goal-v4-010-011-reviewer-handoff.md`
  - this closure handoff.

## Contracts Touched

- V4 two-agent handoff protocol.
- V4 reviewer report path and report template.
- V4 blocked-review handling: blocked or unavailable review is not approval.
- V4 claim guard: smoke tests, path existence, and package presence are not
  paper readiness.

## Evidence Produced

- `specs/002-phm-genbench-frontier/reviews/v4/README.md`
- `specs/002-phm-genbench-frontier/reviews/v4/2026-06-18-goal-v4-010-011-reviewer-handoff-review.md`
- `specs/002-phm-genbench-frontier/handoffs/2026-06-18-goal-v4-010-011-reviewer-handoff.md`

## Canonical Evidence Root

- root: `results/paper/phm_generative/six_dataset_submission_v1/`
- package manifest:
  `results/paper/phm_generative/six_dataset_submission_v1/paper_evidence_package/package_manifest.json`
- readiness:
  `specs/002-phm-genbench-frontier/paper/submission_readiness.md`

The current paper evidence remains `NOT_SUBMISSION_READY` with
`benchmark_valid_row_count=0`. This goal did not promote any result rows.

## Validation Commands Run

```bash
python -m scripts.validate_docs
rg -n "REVIEW_DECISION|READINESS_SCORE|NEXT_GOAL|FIX_INSTRUCTION" .specify/goals/v4/reviewer.md .specify/goals/v4/handoff.md specs/002-phm-genbench-frontier/reviews/v4/README.md
rg -n "Agent A|Agent B|Builder Start|Builder Handoff|Required Handoff File" .specify/goals/v4/handoff.md
rg -n "reviewer.md|Reviewer Gate|Agent B" .specify/goals/v4/goal.md
```

## Validation Results

- `python -m scripts.validate_docs`: passed, 122 files scanned.
- reviewer XML tag search: passed.
- handoff required-section search: passed.
- `goal.md` reviewer gate reference search: passed.

## Reviewer Decision

`PASS_WITH_WARNINGS`

Review report:
`specs/002-phm-genbench-frontier/reviews/v4/2026-06-18-goal-v4-010-011-reviewer-handoff-review.md`

## Unresolved Blockers

None for GOAL-V4-010/011.

## Remaining Warnings

- The current review is a local Codex audit. Future runtime/evidence goals
  should use an independent Agent B review before supporting paper claims.
- `.specify/goals/v4/` and `specs/002-phm-genbench-frontier/reviews/v4/` are
  currently untracked and should be added intentionally when committing.

## Known Risks

- A future agent could skip the independent Agent B review unless the V4
  protocol is enforced per goal.
- The V4 process artifacts are currently untracked in this worktree.
- The protocol improves review discipline but does not itself fix evidence,
  evaluation, statistics, utility, method fidelity, or promotion gates.

## Required Reviewers

- Agent B: V4 paper-readiness auditor using `.specify/goals/v4/reviewer.md`.
- For runtime/frontier goals, add method-fidelity and leakage-focused review as
  needed before paper claims.

## Required Context Files

- `.specify/goals/v4/goal.md`
- `.specify/goals/v4/handoff.md`
- `.specify/goals/v4/reviewer.md`
- `specs/002-phm-genbench-frontier/paper/submission_readiness.md`
- `specs/002-phm-genbench-frontier/reviews/v3/paper-readiness-decision.md`
- `results/paper/phm_generative/six_dataset_submission_v1/paper_evidence_package/package_manifest.json`

## Review Output Format

```xml
<REVIEW_DECISION>APPROVE | PASS_WITH_WARNINGS | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<READINESS_SCORE>0-100</READINESS_SCORE>
<BLOCKING_ISSUES>
- issue / evidence path / required fix
</BLOCKING_ISSUES>
<NON_BLOCKING_ISSUES>
- issue / evidence path / suggested fix
</NON_BLOCKING_ISSUES>
<NEXT_GOAL>
GOAL-V4-xxx-title
</NEXT_GOAL>
<FIX_INSTRUCTION>
Codex-ready instruction with exact files, behavior, and tests.
</FIX_INSTRUCTION>
```

## Next Goal

`GOAL-V4-000-CLAIM-FREEZE`

## Next Steps

1. Start `GOAL-V4-000-CLAIM-FREEZE` with the Builder Start template from
   `.specify/goals/v4/handoff.md`.
2. Keep all current evidence wording exploratory while
   `benchmark_valid_row_count=0`.
3. Run the goal validation commands, then create a feature-scoped handoff and
   V4 review report before proceeding to later V4 goals.

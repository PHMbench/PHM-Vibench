# V4 Reviewer Reports

This directory stores per-goal V4 paper-readiness review reports.

Every report must use `.specify/goals/v4/reviewer.md` as the review gate and
must be named:

```text
<YYYY-MM-DD>-<goal-id>-review.md
```

## Report Template

```md
# V4 Review: GOAL-V4-xxx-title

## Scope

- Goal ID:
- Builder handoff:
- Review date:
- Reviewer:

## Repo Truth Inspected

- `git status --short`:
- Changed files inspected:
- Required context inspected:
  - `.specify/goals/v4/goal.md`
  - `.specify/goals/v4/reviewer.md`
  - goal-specific evidence paths

## Validation Checked

| Command | Actually run | Result | Evidence |
|---|---:|---|---|
| `python -m scripts.validate_docs` | yes/no | pass/fail/not run | path or note |

## Evidence Paths Checked

- canonical root:
- summary:
- manifest:
- paperpack:
- draft/readiness:

## Reviewer Axis Scores

| Axis | Score 0-5 | Evidence | Notes |
|---|---:|---|---|
| Claim safety |  |  |  |
| Evidence root |  |  |  |
| Eval protocol |  |  |  |
| Condition budget |  |  |  |
| Statistical adequacy |  |  |  |
| Utility claim |  |  |  |
| Method fidelity |  |  |  |
| Promotion gate |  |  |  |
| Paperpack traceability |  |  |  |
| Reproducibility |  |  |  |

## Decision

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
```

## Rules

- A blocked or unavailable review is not approval.
- `PASS_WITH_WARNINGS` does not allow benchmark-valid paper claims.
- Smoke tests, path existence, and package presence are not submission readiness.
- Every `BLOCKING` issue must map to a small next V4 goal.

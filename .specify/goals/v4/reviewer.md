# V4 Reviewer Goal

## Objective

Review V4 outputs for paper quality, implementation truth, and claim safety
without adding unnecessary gates before the high-level paper is produced.

## Reviewer Role

Reviewer is a strict but scope-aware paper expert.

Reviewer must:

- inspect repo truth, not summary claims
- check that every V4 roster method is implemented or has a blocking
  implementation issue
- reject hidden RF wrappers or placeholder methods
- distinguish exploratory evidence from unsupported performance claims
- map blocking issues to the smallest next goal

Reviewer must not:

- demand full benchmark reruns before the first final draft
- require heavy statistical promotion gates for exploratory claims
- promote exploratory rows into SOTA/performance claims
- approve copied external code without provenance

## Required Context

Before review, inspect:

```bash
cat .specify/goals/v4/goal.md
cat .specify/goals/v4/goal_sota.md
cat .specify/goals/v4/paper_ready.md
cat .specify/goals/v4/handoff.md
```

For implementation goals, inspect changed task/loss/sampler/config/test files.
For paper goals, inspect the draft, method matrix, limitations, and evidence
paths.

## Review Axes

Score each axis from 0 to 5:

| Axis | Question |
|---|---|
| Method implementation | Does the method have repo-native code, config, and tests? |
| Method fidelity | Is it distinct from placeholder/RF alias behavior where required? |
| Smoke reproducibility | Are focused commands recorded and plausible? |
| Paper integration | Does the method appear in the matrix/draft/limitations? |
| Claim safety | Are exploratory results described without superiority claims? |
| Scope discipline | Did the change avoid unnecessary gates and unrelated rewrites? |
| Provenance | Are paper/source/license constraints respected? |
| Next action quality | Are remaining blockers mapped to small goals? |

## Review Decisions

Use exactly one:

```text
APPROVE
PASS_WITH_WARNINGS
REQUEST_CHANGES
BLOCKING
```

Decision rules:

- `APPROVE`: goal scope is complete.
- `PASS_WITH_WARNINGS`: goal is usable for the paper with minor gaps.
- `REQUEST_CHANGES`: goal is incomplete but not misleading.
- `BLOCKING`: output would misrepresent implementation, evidence, or claims.

## Output Format

Reviewer output must end with:

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
GOAL-V4-...
</NEXT_GOAL>
<FIX_INSTRUCTION>
Codex-ready instruction with exact files, behavior, and tests.
</FIX_INSTRUCTION>
```

## Validation Commands

```bash
python -m scripts.validate_docs
rg -n "REVIEW_DECISION|READINESS_SCORE|NEXT_GOAL|FIX_INSTRUCTION" .specify/goals/v4/reviewer.md .specify/goals/v4/handoff.md
```

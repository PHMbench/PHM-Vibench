# V4 Six-Subagent Handoff Protocol

## Purpose

Coordinate six xhigh subagent lanes for PHM-GenBench V4. The protocol is a work
split and handoff format, not an extra approval system.

## Six Subagents

| ID | Role | Owns | Must not own |
|---|---|---|---|
| S1 | Baseline Contract | CFM/RF/DDPM and shared generative contracts | paper claims |
| S2 | Score SDE | Score SDE implementation and tests | unrelated flow methods |
| S3 | MeanFlow | MeanFlow implementation and tests | RF fallback claims |
| S4 | Drifting and TFM | Drifting and Transition Flow Matching implementations | OT-NFM |
| S5 | OT-NFM | OT coupling, OT-NFM sampler, tests | paper wording |
| S6 | Paper Integration | method matrix, final draft, reviewer pass | runtime shortcuts |

## Global Rules

1. One active goal per subagent lane.
2. Every method in the V4 roster must be implemented, not only cited.
3. Subagents may work in parallel only when their owned files do not overlap.
4. Shared schema or pipeline changes belong to S1 unless the active goal says
   otherwise.
5. The paper can proceed with exploratory evidence if claims are explicit.
6. Do not add heavyweight promotion gates before the first final draft.
7. A failed smoke/preflight is a blocking implementation issue for that method.
8. A claim mismatch is a paper issue for S6.

## Required Inputs

Every subagent must inspect:

```text
.specify/goals/v4/goal.md
.specify/goals/v4/goal_sota.md
.specify/goals/v4/paper_ready.md
.specify/goals/v4/reviewer.md
```

Implementation subagents must also inspect the owned task/loss/sampler/config
files before editing.

## Builder Start

Use this template before implementation:

```md
## Builder Start

Subagent:
Sx - role

Active goal file:
.specify/goals/v4/<goal-file>.md

Goal ID:
GOAL-V4-...

Objective:
...

Owned files:
- path

Expected behavior:
...

Validation plan:
...

Known risks:
...
```

## Builder Handoff

Use this template after implementation:

````md
## Builder Handoff

Subagent:
Sx - role

Goal ID:
GOAL-V4-...

Implementation summary:
...

Files changed:
- path: reason

Behavior changed:
...

Method status:
- implemented:
- smoke/preflight:
- exploratory evidence:
- unsupported claims:

Paper integration:
- method matrix updated:
- draft/limitations updated:

Validation commands run:
```bash
...
```

Validation results:
...

Known gaps:
...

Requested reviewer focus:
...
````

Save the handoff to:

```text
specs/002-phm-genbench-frontier/handoffs/<YYYY-MM-DD>-<goal-id>.md
```

## Reviewer Response

The reviewer follows `.specify/goals/v4/reviewer.md` and returns:

```xml
<REVIEW_DECISION>APPROVE | PASS_WITH_WARNINGS | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<READINESS_SCORE>0-100</READINESS_SCORE>
<BLOCKING_ISSUES>
...
</BLOCKING_ISSUES>
<NON_BLOCKING_ISSUES>
...
</NON_BLOCKING_ISSUES>
<NEXT_GOAL>
...
</NEXT_GOAL>
<FIX_INSTRUCTION>
...
</FIX_INSTRUCTION>
```

## Parallel Work Rule

Recommended parallel split:

```text
S1: baseline/shared contracts
S2: Score SDE
S3: MeanFlow
S4: Drifting and TFM
S5: OT-NFM
S6: paper integration after method statuses exist
```

If two subagents need the same shared file, S1 makes the shared change first.

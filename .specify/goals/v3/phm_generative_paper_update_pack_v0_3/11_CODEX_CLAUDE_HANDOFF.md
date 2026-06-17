# 11. Codex / Claude Code Handoff

## Codex role

Codex owns implementation and validation.

Codex should:
- make small PRs
- run validation commands
- write tests
- keep runtime changes inside factories
- avoid adding new method families before current evidence gates pass

## Claude Code role

Claude Code should be used as advisory reviewer only.

Claude should:
- review architecture
- identify leakage and benchmark-valid risks
- check path coherency between train/sample/eval/paperpack
- check method naming and paper claims
- check if one-step methods are incorrectly promoted
- check if `.specify` artifacts are canonical

## Handoff rule

Every implementation goal must write:

```text
specs/002-phm-genbench-frontier/handoffs/<date>-<goal>.md
```

with:

```text
goal id
files changed
validation commands run
evidence produced
known gaps
next suggested goal
```

## Default Claude review prompt

See `prompts/CLAUDE_REVIEW_PROMPT_GEN_V3.md`.

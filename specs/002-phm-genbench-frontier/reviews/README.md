# Reviews

This directory stores feature-scoped review artifacts for PHM-GenBench goals.
`.codex/` and `.claude/` may mirror tool output, but this directory is the
canonical review location.

Claude Code Teams must start in read-only `plan` or `review` mode unless a
later implementation goal partitions write ownership explicitly. Do not launch
Claude if the configured endpoint would export private workspace content to an
unapproved external service; record `BLOCKED_NOT_RUN` instead.

Required reviewer roles for general PHM generative review:

- `phm-gen-architect`
- `phm-gen-loss-reviewer`
- `phm-gen-leakage-reviewer`

The general read-only review task spec template is:
`claude-team/phm-gen-general-review-template/TASK_SPEC.md`.

M2 paper-package reviews may add bounded teammates for dataset protocol,
metrics/figures, paper narrative, and governance/leakage.

Required files for each Claude team run:

- `TASK_SPEC.md`
- `report.md`
- `risks.md`
- `test-log.md`

Required Claude output ending:

```xml
<REVIEW_DECISION>APPROVE | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<BLOCKING_ISSUES>
...
</BLOCKING_ISSUES>
<NON_BLOCKING_ISSUES>
...
</NON_BLOCKING_ISSUES>
<FIX_INSTRUCTION>
Codex-ready patch instruction.
</FIX_INSTRUCTION>
```

Required local context:

- `src/task_factory/task/generative/README.md`
- `src/task_factory/Components/generative/README.md`
- `src/task_factory/Components/generative/losses/README.md`
- `src/task_factory/Components/generative/manifests/README.md`
- `src/model_factory/generative_model/README.md`

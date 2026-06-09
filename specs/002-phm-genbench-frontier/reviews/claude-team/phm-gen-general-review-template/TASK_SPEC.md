# Claude Team Task Spec Template: PHM-GenBench General Review

## Objective

Review a PR-sized PHM generative goal for factory placement, loss correctness,
synthetic-data governance, and benchmark-validity risk before Codex accepts the
goal as complete.

## Mode

Read-only `review` mode first. Edits are not allowed for this template.

## Teammates

- `phm-gen-architect`
- `phm-gen-loss-reviewer`
- `phm-gen-leakage-reviewer`

## Required Context

- `.specify/goals/v2/<goal-id>.md`
- `src/task_factory/task/generative/README.md`
- `src/task_factory/Components/generative/README.md`
- `src/task_factory/Components/generative/losses/README.md`
- `src/task_factory/Components/generative/manifests/README.md`
- `src/model_factory/generative_model/README.md`
- Active feature artifacts under `specs/002-phm-genbench-frontier/`

## Review Checklist

- Factory placement uses existing `data_factory`, `model_factory`,
  `task_factory`, and `trainer_factory` paths.
- CFM target remains `x1 - z` for V0 velocity matching.
- `fault_label` and `domain_id` are the only default model condition keys.
- `load` and `rpm` remain domain-map metadata.
- FFT, STFT, envelope, and spectral calculations remain eval-only in V0.
- Synthetic data cannot be `benchmark-valid` without manifest, protocol,
  normalization, condition, leakage, and metric evidence.
- MeanFlow and Drifting remain research-only unless a promotion goal exists.
- No output is trusted without Codex verification.

## Required Outputs

Write these files under the concrete Claude team run directory:

- `report.md`
- `risks.md`
- `test-log.md`

The final review report must end with:

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

If endpoint approval is missing, do not launch the team. Write
`BLOCKED_NOT_RUN` in each required output file for the concrete run and treat
the blocked review as evidence of non-execution, not approval.

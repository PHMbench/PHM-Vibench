# Claude Team Task Spec: PHM-GenBench M2 Six-Dataset Submission Queue

## Objective

Review the M2 six-dataset PHM generative benchmark package before any long
training run is accepted as paper evidence.

## Mode

Read-only `review` mode first. Edits are not allowed in this run.

## Target Paths

- `.specify/goals/v2/GOAL-GEN-M2-*`
- `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
- `scripts/generative_benchmark_effect.py`
- `scripts/generative_submission_draft.py`
- `test/generative/test_six_dataset_submission.py`
- `specs/002-phm-genbench-frontier/`

## Out Of Scope

- Do not push, publish, deploy, delete, or read secrets.
- Do not start full training.
- Do not modify runtime model, task, sampler, metric, or manifest semantics.

## Teammates

- Dataset protocol auditor.
- Metrics and figures auditor.
- Paper narrative reviewer.
- Governance and leakage reviewer.

## Required Outputs

Write these files under this run directory:

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

If the team is not launched because endpoint approval is missing, write
`BLOCKED_NOT_RUN` in each required output file. A blocked review is evidence of
non-execution only, not independent approval.

## Safety Gate

Before launch, Codex must verify that the configured Claude endpoint is approved
for the workspace content in scope. If it is not approved, do not launch the
team; write `BLOCKED_NOT_RUN` review files and continue with local Codex
verification.

## Acceptance Checks

```bash
python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q
python -m compileall scripts/generative_benchmark_effect.py scripts/generative_submission_draft.py
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run
python -m scripts.validate_docs
```

GPU preflight is required before real runs:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run
```

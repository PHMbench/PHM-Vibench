# UXFD Seven-Paper Goal Index

This directory is the parent-level goal package for preparing the seven UXFD
paper submodules as independent IEEE Transactions submissions.

## Workflow

Use the Spec Kit sequence as the controlling workflow:

```text
constitution -> specify -> clarify -> plan -> checklist -> tasks -> analyze -> implement
```

The current feature artifacts live in:

- `specs/006-uxfd-ieee-trans-submission-readiness/spec.md`
- `specs/006-uxfd-ieee-trans-submission-readiness/plan.md`
- `specs/006-uxfd-ieee-trans-submission-readiness/tasks.md`

## Goal Files

| File | Purpose |
|---|---|
| `00_overall_goal.md` | Shared seven-paper objective, evidence contract, and operating rules. |
| `01_explainable_fd_toolkit.md` | Paper 1 goal: explainability infrastructure and benchmark toolkit. |
| `02_1d2d_fusion.md` | Paper 2 goal: 1D-2D fusion explainable diagnosis. |
| `03_llm_explainable_fd_toolkit.md` | Paper 3 goal: LLM evidence-chain explanation layer. |
| `04_moe_explainable.md` | Paper 4 goal: physics-constrained MoE and route-level explanations. |
| `05_fuzzy_xfd.md` | Paper 5 goal: fuzzy rule-level explainable diagnosis. |
| `06_neuralsymbolic_theory.md` | Paper 6 goal: neural-symbolic theory and proposition validation. |
| `07_tii_operator_attention.md` | Paper 7 goal: operator-attention theory and signal-processing evidence. |
| `08_recent_work_citation_readme.md` | TOP recent-work citation map and runnable reproduction policy. |
| `99_submission_readiness_matrix.md` | Cross-paper status matrix and next milestones. |

## Status Legend

- `blocked`: the paper cannot be treated as submission-ready until a named blocker is resolved.
- `unverified`: a contract exists, but evidence has not been accepted for the claim.
- `evidence-ready`: all required artifacts exist and are mapped, but manuscript or compile work remains.
- `compile-ready`: canonical TeX compiles without fatal errors.
- `submission-ready`: evidence, manuscript, compile, and strict-reviewer gates pass.

## Commit Policy

- Paper-specific edits are committed inside the owning submodule first.
- Parent commits record only goal/spec updates and intentional submodule gitlink updates.
- Each important milestone should be one reviewable submodule commit, not a mixed cross-paper batch.
- Existing dirty submodule work is treated as user work until attributed.

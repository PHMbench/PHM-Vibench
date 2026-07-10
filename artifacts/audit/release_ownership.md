# PHM-Vibench v0.2.0 Ownership and PR Sequence

## Independence rule

The principal implementer must not issue the final release approval. `Reviewer R1` owns the independent release recommendation. `Reviewer C1` provides a counterexample-oriented second opinion and cannot merge.

## Role ownership

| Role | Primary scope | Repository write scope | Gate output |
|---|---|---|---|
| A0 Release Owner | decomposition, PR order, conflicts, risk escalation, handoff | audit/release docs and coordination-only changes | release decision package |
| A1 Pipeline Auditor | entrypoints, commands, smoke/mini-E2E evidence | pipeline-specific tests/fixes in scoped PR | `pipeline_ledger.csv` |
| A2 Task Auditor | task registry, losses, metrics, task/model/trainer constraints | task-specific scoped PR | task portion of component and compatibility inventories |
| A3 Model Auditor | model registry, shapes, dtype/device, checkpoint contract | model-specific scoped PR | model contract ledger |
| A4 Sampler/Config Auditor | sampler dispatch, schema, defaults, CLI override, parameter lifecycle | config/sampler scoped PR | `parameter_trace_matrix.csv` and compatibility rules |
| A5 Deep Code Archaeologist | seven-level static/dynamic paths, side effects, dead/duplicate code | no broad refactor without evidence PR | callgraph and duplication ledgers |
| A6 Test/Reproducibility Engineer | unit/integration/smoke/E2E/seed/device/CI/artifact evidence | tests and CI scoped PR | test evidence and reproducibility report |
| A7 Paper/Docs/Figure Auditor | claim-evidence, provenance, docs, release narrative | docs/paper/figure scoped PR | claim-evidence matrix and user-facing release docs |
| R1 Independent Reviewer | independent diff, evidence, and release review | review comments only; no main implementation | `ACCEPT`, `ACCEPT_WITH_RISK`, `REWORK`, or `BLOCK` |
| C1 Claude Reviewer | adversarial counterexample search | structured findings only; no merge | `state/claude_findings.md` |

## Cloud PR sequence

| PR stage | Primary objective | Entry condition | Exit gate |
|---|---|---|---|
| PR-0 | static baseline, inventories, SOP, ownership | current `main` | no runtime claims; baseline facts reviewable |
| PR-1 | pipeline blockers and honest status | local baseline logs available | supported candidates reach at least S3/S4 with evidence |
| PR-2 | task/model/sampler/config semantics and compatibility | component inventory complete | public parameters traced; incompatibilities rejected early |
| PR-3 | evidence-backed duplication reduction | callgraph and equivalence tests exist | no accidental behavior or compatibility regression |
| PR-4 | test matrix, active CI, reproducibility | stable component contracts | repeatable automated gates and attached evidence |
| PR-5 | paper/docs/figures/user entry | implementation status frozen | claims, commands, and provenance match code |
| PR-6 | v0.2.0 release preparation | R1/C1 review and no P0/P1 | release files complete; scorecard gate passes |

## Active PR interaction

| PR | Classification for v0.2.0 control | Required action |
|---|---|---|
| #39 | UXFD U1 runtime candidate | keep independently reviewable; do not label release-supported until pipeline evidence exists |
| #40 | repository slimming | review deletions and recovery evidence; do not suppress canonical `.github/` or generated config SSOT |
| #41 | generative migration design | documentation/design only; no generative support claim |
| #42 | stacked UXFD minimal demo | retain draft until #39 merges and local smoke passes |
| #43 | PR-0 baseline and optimization SOP | this branch owns initial audit facts only |

## Route handoff contract

Each route writes a structured handoff containing:

```yaml
task_id:
agent:
start_commit:
end_commit:
files_read:
files_changed:
commands_run:
tests_passed:
tests_failed:
pipelines_checked:
combinations_checked:
parameters_traced:
findings:
risks:
open_questions:
recommended_next_actions:
evidence_paths:
```

Unexecuted work must be written as `NOT_EXECUTED`, never inferred as passing.

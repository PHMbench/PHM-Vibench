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
| `09_gpu_execution_queue.yaml` | Machine-readable 2x4090 execution queue, resource preflight, metadata contract, and SOTA gate order. |
| `99_submission_readiness_matrix.md` | Cross-paper status matrix and next milestones. |

## Status Legend

- `blocked`: the paper cannot be treated as submission-ready until a named blocker is resolved.
- `unverified`: a contract exists, but evidence has not been accepted for the claim.
- `evidence-ready`: all required artifacts exist and are mapped, but manuscript or compile work remains.
- `compile-ready`: canonical TeX compiles without fatal errors.
- `submission-ready`: evidence, manuscript, compile, and strict-reviewer gates pass.

## Queue Dry Run

Use the parent expander to inspect the blocked GPU queue without launching any
experiment:

```bash
python -m scripts.uxfd_gpu_queue --format markdown
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
python -m scripts.uxfd_gpu_queue --format json --output paper/UXFD_paper/results/gpu_queue_dry_run.json
```

The command is intentionally dry-run only. Use `--require-preflight` to fail
with a non-zero exit code until local GPUs `0,1` are visible and accepted.
Use `--live-preflight` to check the current `nvidia-smi -L` and PyTorch CUDA
state in the manifest before any experiment launch.
The generated manifest includes validation status, summary counts by phase and
paper, and the expanded command sources.

## Objective Audit

Use the objective audit before claiming that the active goal is achieved. It
maps the user request to concrete filesystem and gate evidence:

```bash
python -m scripts.uxfd_objective_audit --format markdown
python -m scripts.uxfd_objective_audit --format json --allow-not-achieved --output paper/UXFD_paper/results/objective_audit_current.json
python -m scripts.uxfd_objective_audit --format markdown --allow-not-achieved --output paper/UXFD_paper/results/objective_audit_current.md
```

The command returns non-zero until every named goal file, Spec Kit artifact,
handoff artifact, team/subagent evidence, seven paper matrices, TOP gate, GPU
queue, accepted artifacts, and cross-paper submission gate are satisfied. Use
`--allow-not-achieved` only to export the current audit without treating it as
complete.

## Goal Clarity Audit

Use the clarity audit when checking whether the goal package itself has enough
specificity to proceed. This is a human-readable audit artifact, not a
submission-readiness gate and not accepted experiment evidence:

```bash
sed -n '1,220p' paper/UXFD_paper/results/goal_clarity_audit_current.md
python -m pytest -q test/test_uxfd_goal_clarity.py
```

The current audit records that the goal files are structurally clear enough for
staged preparation, but still blocked for full execution by GPU preflight,
accepted artifacts, TOP representative evidence, dirty submodules, and
submission gates.

## Owner Review Gate

Use the owner-review gate before staging or committing dirty paper-submodule
work. The template is decision support only; it is not paper-owner approval:

```bash
python -m scripts.uxfd_owner_review_gate --format markdown
python -m scripts.uxfd_owner_review_gate --format json --allow-not-ready --output paper/UXFD_paper/results/submodule_owner_review_gate_current.json
python -m scripts.uxfd_owner_review_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/submodule_owner_review_gate_current.md
```

The gate returns non-zero until
`paper/UXFD_paper/results/submodule_owner_review_decisions.json` exists, covers
all current owner-review packets, and has no `pending_owner_review` records.
Use `paper/UXFD_paper/results/submodule_owner_review_action_packet.md` as the
short owner-facing response form, but do not treat it as approval.
Use
`paper/UXFD_paper/results/submodule_owner_review_decisions.template.json` only
as the starting point for real paper-owner decisions.

## Submission Gate

Use the parent gate checker to prove the package is or is not ready for
submission without launching experiments:

```bash
python -m scripts.uxfd_submission_gate --format markdown
python -m scripts.uxfd_submission_gate --format json --allow-not-ready --output paper/UXFD_paper/results/submission_gate_current.json
python -m scripts.uxfd_submission_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/submission_gate_current.md
python -m scripts.uxfd_submission_gate --artifact-root paper/UXFD_paper/results/accepted_runs --format markdown --allow-not-ready
```

The command returns non-zero while any paper remains non-ready. Use
`--allow-not-ready` only for non-failing audit export.
The report includes blocking findings plus one queue-derived next action per
paper. It also includes an objective checklist mapping goal files, Claude Team
artifacts, paper matrices, baseline/ablation counts, GPU queue status, and final
submission readiness.
The submission gate also runs the artifact metadata gate against
`paper/UXFD_paper/results/accepted_runs` by default; use `--artifact-root` to
point at a specific accepted evidence bundle.
It also runs the SOTA aggregate gate against
`paper/UXFD_paper/results/sota_aggregates` by default; use `--sota-root` to
point at a specific aggregate evidence bundle.

## Recent Work Gate

Use the TOP recent-work gate to audit citation freshness, low-tier exclusion,
per-paper TOP quotas, and the seven queued TOP representative bindings:

```bash
python -m scripts.uxfd_recent_work_gate --format markdown
python -m scripts.uxfd_recent_work_gate --format json --allow-not-ready --output paper/UXFD_paper/results/recent_work_gate_current.json
python -m scripts.uxfd_recent_work_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/recent_work_gate_current.md
```

The command returns non-zero while TOP representative artifacts remain pending.
Use `--allow-not-ready` only for non-failing audit export. `policy_ready: true`
means the accepted TOP pool, per-paper quotas, and queue bindings are coherent;
`evidence_ready: false` means the queued representatives still lack accepted
same-protocol logs, metrics, and `run_meta.yaml` evidence.

## Artifact Gate

Use the artifact gate after real runs finish to check accepted-run metadata
without changing any result:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage --format markdown --allow-not-ready --output paper/UXFD_paper/results/artifact_gate_queue_coverage.md
```

The gate requires `run_meta.yaml` plus local 4090 GPU metadata, config/log/metrics
paths, seed, split, batch size, precision, runtime, and command provenance.
Its field map is tested against `09_gpu_execution_queue.yaml` so the scheduler
metadata contract and artifact validator stay aligned.

## SOTA Aggregate Gate

Use the SOTA aggregate gate after artifact coverage exists to check that SOTA
wording is based on matched-seed aggregate evidence, not single runs:

```bash
python -m scripts.uxfd_sota_scaffold --output-root paper/UXFD_paper/results/sota_aggregate_templates --format markdown --output paper/UXFD_paper/results/sota_aggregate_templates/scaffold_report.md
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_sota_gate --format json --allow-not-ready --output paper/UXFD_paper/results/sota_gate_current.json
python -m scripts.uxfd_sota_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/sota_gate_current.md
```

The scaffold command writes non-evidence templates under
`paper/UXFD_paper/results/sota_aggregate_templates/`. The gate expects one
`sota_aggregate.yaml` per paper under
`paper/UXFD_paper/results/sota_aggregates/<paper_id>/`. Each aggregate must
cover the proposed method, all declared baselines, and runnable TOP
representative bindings with matched seeds, mean/std/95% CI, and effect-size or
paired-test evidence. Every proposed, baseline, and TOP aggregate entry must
also list `accepted_run_refs`: relative paths to existing `run_meta.yaml` files
under `paper/UXFD_paper/results/accepted_runs`. Template, smoke, demo, dummy,
pending, absolute, missing, or out-of-root references are rejected.

## Commit Policy

- Paper-specific edits are committed inside the owning submodule first.
- Parent commits record only goal/spec updates and intentional submodule gitlink updates.
- Each important milestone should be one reviewable submodule commit, not a mixed cross-paper batch.
- Existing dirty submodule work is treated as user work until attributed.

## Commit Recovery Plan

If git-index writes are blocked, use the recovery plan to resume without
restaging unrelated work:

```bash
sed -n '1,260p' paper/UXFD_paper/results/commit_recovery_plan.md
```

The plan stages Paper02 planning files first, then the parent goal-control
checkpoint, and keeps generated figures, unreviewed manuscripts, model weights,
and unrelated parent edits out of the checkpoint.

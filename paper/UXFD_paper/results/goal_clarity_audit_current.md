# UXFD Goal Clarity Audit

Date: 2026-05-14

This report checks whether the goal package is clear enough to execute from the
current repository state. It does not claim IEEE Transactions submission
readiness and does not replace the objective or submission gates.

## Scope

Checked files:

- `paper/UXFD_paper/goal/README.md`
- `paper/UXFD_paper/goal/00_overall_goal.md`
- `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md`
- `paper/UXFD_paper/goal/02_1d2d_fusion.md`
- `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md`
- `paper/UXFD_paper/goal/04_moe_explainable.md`
- `paper/UXFD_paper/goal/05_fuzzy_xfd.md`
- `paper/UXFD_paper/goal/06_neuralsymbolic_theory.md`
- `paper/UXFD_paper/goal/07_tii_operator_attention.md`
- `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`
- `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`

## Findings

| Check | Result | Evidence |
|---|---|---|
| Named goal files exist | pass | `find paper/UXFD_paper/goal -maxdepth 1 -type f` lists all named goal files plus `09_gpu_execution_queue.yaml`. |
| Spec Kit path is declared | pass | `goal/README.md` points to `specs/006-uxfd-ieee-trans-submission-readiness/{spec,plan,tasks}.md`. |
| Six xhigh agents are evidenced | pass | `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/CODEX_SUBAGENT_LAUNCH.md` lists six xhigh read-only agents and scopes. |
| Per-paper baseline/ablation expectations are explicit | pass | Each paper goal has a Baseline Suite, Ablation Suite, SOTA Optimization Gate, TOP Recent-Work Quota, Compute Budget, Strict-Reviewer Risks, and Acceptance Gates. |
| 2x4090 resource constraint is explicit | pass | `00_overall_goal.md` and `09_gpu_execution_queue.yaml` bind local GPUs `0,1` and prohibit unapproved cloud/A100/H100 assumptions. |
| TOP recent-work policy is explicit | pass | `08_recent_work_citation_readme.md` defines exact-runnable, representative-runnable, literature-only, blocked, and resource-blocked statuses. |
| Low-tier source exclusion is explicit | pass | Scientific Reports, MDPI publisher-level journals, IEEE TIM, IEEE Access, Applied Sciences, Electronics, Sensors, and Mathematics appear only in exclusion/source-hygiene contexts in the goal package. |
| Stale execution paths in goal files | pass with note | No stale `copy 2`, `Paper/1D`, `--config_dir`, or `config_dir` execution command remains in `paper/UXFD_paper/goal`. The only `TODO` hit is a reviewer-risk sentence warning that TODO claims make manuscripts non-ready. |
| Paper02 pending planning update is visible | pass | `99_submission_readiness_matrix.md` marks `plan/EXPERIMENT_PLAN_补充.md` and `program.md` as a pending uncommitted planning update, not accepted evidence. |

## Current Non-Execution Blockers

The goal package is clear enough for staged execution, but execution is blocked
by current state:

- Experiment launch gate is not ready: it reports 3 launch blockers covering
  owner-review, static queue execution, and live 2x4090 preflight.
- GPU queue cannot execute: current preflight reports `nvidia-smi` driver
  failure and PyTorch `cuda_available=False`, `device_count=0`.
- Owner-review gate cannot pass: real
  `paper/UXFD_paper/results/submodule_owner_review_decisions.json` is missing,
  and 6 dirty-submodule review records remain pending.
- No accepted experiment evidence exists yet:
  `paper/UXFD_paper/results/accepted_runs` has zero accepted records.
- SOTA aggregate evidence cannot be accepted until accepted run references
  exist for the seven queue bindings.
- TOP representative artifacts remain pending for all seven queue bindings.
- Submodule working trees are not clean:
  `Explainable_FD_Toolkit:22`, `1D-2D_fusion_explainable:3`,
  `MOE_explainable:2` (27 dirty entries total).
- Recent-work policy is clear and source-hygiene checked, but representative
  TOP evidence is still blocked by missing GPU/artifact records.
- All seven paper matrices remain `submission_ready: false`.

## Verification Commands

```bash
rg -n "TODO|TBD|待定|未定|不清楚|copy 2|Paper/1D|config_dir|--config_dir" paper/UXFD_paper/goal
rg -n "Scientific Reports|MDPI|IEEE Transactions on Instrumentation and Measurement|IEEE TIM|IEEE Access|Applied Sciences|Electronics|Sensors|Mathematics" paper/UXFD_paper/goal/00_overall_goal.md paper/UXFD_paper/goal/08_recent_work_citation_readme.md paper/UXFD_paper/goal/99_submission_readiness_matrix.md
python -m pytest -q test/test_uxfd_goal_clarity.py
python -m scripts.uxfd_experiment_launch_gate --format markdown
python -m scripts.uxfd_objective_audit --format markdown --allow-not-achieved
python -m scripts.uxfd_submission_gate --format markdown --allow-not-ready
```

## Verdict

The goal package is clear enough to continue staged preparation and, after the
experiment launch gate passes without override flags, to execute the queue in
`09_gpu_execution_queue.yaml`. That launch gate must clear real owner
decisions, static queue execution, and live local GPU `0,1` RTX 4090 preflight.
It is not clear to treat any paper as submission-ready, because accepted
artifacts, TOP representative evidence, GPU metadata, real owner decisions,
clean submodule state, SOTA aggregates, and final cross-paper submission gates
are still missing.

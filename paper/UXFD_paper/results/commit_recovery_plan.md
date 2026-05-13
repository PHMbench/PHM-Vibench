# UXFD Commit Recovery Plan

Created: 2026-05-12
Updated: 2026-05-14

This file records the exact recovery path used for the parent checkpoint.
It is not experiment evidence and does not make any paper submission-ready.

## Current Checkpoint Status

The recovery sequence below is now historical. As of 2026-05-14:

- The parent UXFD goal-control checkpoint is clean.
- The current objective audit reports `parent UXFD goal-control checkpoint committed`
  as `met`.
- The latest parent UXFD control commits include:
  - `a366356 fix: surface UXFD source verification in aggregate gates`
  - `b208a39 test: require source verification in UXFD recent work gate`
  - `672b086 docs: add UXFD recent work source verification`
  - `b92a3d9 docs: require minimum seed coverage in UXFD backlog`
- The active blockers are no longer parent checkpoint recovery. They are:
  - dirty paper submodules requiring owner review,
  - unavailable local 2x4090 CUDA preflight,
  - missing accepted run artifacts under `paper/UXFD_paper/results/accepted_runs`,
  - missing SOTA aggregates under `paper/UXFD_paper/results/sota_aggregates`,
  - seven paper matrices still marked `submission_ready: false`.

Do not replay the historical `git add` commands below wholesale. Stage only the
specific file currently being changed, after checking the current dirty tree.

## Historical Recovery Notes

- Paper02 planning update is committed inside the submodule at `205aaea`:
  - `paper/UXFD_paper/1D-2D_fusion_explainable/plan/EXPERIMENT_PLAN_补充.md`
  - `paper/UXFD_paper/1D-2D_fusion_explainable/program.md`
- Paper02 runner-policy update is committed inside the submodule at `e3325fe`:
  - current-root `python main.py --config ...` runner policy.
  - local GPU `0,1` only policy.
  - local HDF5 dataset loader support for Paper02 dry-run validation.
- Paper02 control-doc update is committed inside the submodule at `18fec7c`:
  - current-root `PHM-Vibench_fix` normalization for README/CORE.
  - maintained `main.py --config ...` commands for VIBENCH and paper-local configs.
  - `innovation_contract.md` is bound from README, CORE, and paper blueprint.
- Paper02 short-input smoke-runner update is committed inside the submodule at
  `e959965`:
  - `code/models/two_d_branch.py` supports short smoke inputs by bounding STFT
    window sizes to the actual signal length.
  - `scripts/run_minimal_demo.py` writes model, figure, and metrics outputs
    under an explicit output root.
  - `scripts/run_fusion_ablation_smoke.py` and
    `scripts/test_fusion_ablation_smoke.py` record the current-root rewrite
    proxy without promoting it to accepted evidence.
- Paper02 truth-first evidence binder is committed inside the submodule at
  `6aff2ff`:
  - adds `scripts/bind_submission_ready_evidence.py` for paper-local
    claim-evidence binding.
  - forces external `submission_ready` to remain `False` until the parent UXFD
    accepted-run, 2x4090, and cross-paper gates pass.
- Paper02 truth-first audit/sync tooling is committed inside the submodule at
  `c17a823`:
  - adds `scripts/run_quantitative_explainability.py` for paper-local
    quantitative explainability summaries.
  - adds `scripts/sync_truth_first_manuscript.py` to replace unsupported
    manuscript claims with parent-gated paper-local evidence wording.
  - adds `scripts/truth_audit.py` to report unsupported claim drift without
    promoting local evidence to external submission-ready status.
- Paper02 comparison collectors are committed inside the submodule at
  `78b6cbb`:
  - replaces heavyweight local comparison scripts with paper-local collector
    packs for MoE, TSPN, and Operator Attention.
  - adds dry-run and explicit output-dir paths so validation does not create
    result artifacts in the submodule working tree.
  - keeps all comparison outputs explicitly outside accepted UXFD run evidence.
- Paper02 truth-first manuscript cleanup is committed inside the submodule at
  `da9f3b9`:
  - `scripts/sync_truth_first_manuscript.py`
  - `scripts/truth_audit.py`
  - `manuscript/final_tex/main.tex`
  - `manuscript/paper.md`
  - `paper_draft/NMI_Paper1_Fusion1D2D.tex`
  - `manuscript/experiments.md`
  - validation passed with `unsupported_claims_remaining=0` and
    `blocking_issue_count=0` under `/tmp/uxfd_paper02_truth_*_current`.
  - do not stage `best_model.pth`, untracked `EXPERIMENT_DESIGN.md`, or
    untracked `manuscript/AUTORESEARCH_EVIDENCE.md` in this checkpoint.
- Paper02 readiness-control text sync is committed inside the submodule at
  `df3e579`:
  - `README_T041_SUBMISSION_READINESS.md`
  - `manuscript/references.bib`
  - `paper_draft/references.bib`
  - `submission_prep/baseline_ablation_matrix.yaml`
  - `submission_prep/ieee_trans_readiness.md`
  - validation passed with `9 passed` for Paper02 control docs, runner policy,
    and low-tier source audit tests.
- Paper04 bounded probe runner update is committed inside the submodule at
  `b1f4084`:
  - current-root path policy for `run_real_dataset_probe.py`.
  - bounded dataset bridge and expert-count probe runner scripts.
  - parent static policy test blocks stale root, legacy GPU, and
    `main_com.py --config_dir` regressions.
- Paper04 control-doc update is committed inside the submodule at `90ed3fe`:
  - current-root `PHM-Vibench_fix` normalization for README/CORE/program.
  - maintained `main.py --config ...` and bounded expert-ablation probe commands.
  - `innovation_contract.md` is bound from README, CORE, and paper blueprint.
- Paper04 truth-first manuscript sync is committed inside the submodule at
  `c832060`:
  - removes appended Markdown run logs from `manuscript/final_tex/main.tex`.
  - keeps the draft explicitly non-submission-ready under the parent UXFD
    submission gate.
  - adds `scripts/sync_truth_first_manuscript.py` as the reproducible sync
    surface for the internal evidence checkpoint.
- Paper04 truth-first evidence binder is committed inside the submodule at
  `2faa58d`:
  - adds `scripts/bind_submission_ready_evidence.py` for internal
    claim-evidence binding.
  - forces external `submission_ready` to remain `False` until parent UXFD
    accepted-run, 2x4090, and cross-paper gates pass.
- Paper01 control-doc update is committed inside the submodule at `dff592b`:
  - current-root `PHM-Vibench_fix` normalization for README/CORE/program.
  - executable commands use lowercase `paper/UXFD_paper/...` paths.
  - `innovation_contract.md` is bound from README, CORE, and paper blueprint.
- Paper01 smoke-runner hardening is committed inside the submodule at
  `23fa1e0`:
  - `scripts/demo.py` and `scripts/run_benchmark_standalone.py` run in
    headless environments.
  - `scripts/run_unified_explain_eval.py` no longer requires optional
    `tabulate` for Markdown table generation.
  - `scripts/run_shap_lime_analysis.py` creates only a non-accepted synthetic
    SHAP/LIME smoke bundle and explicitly blocks SOTA/submission-ready claims.
- Parent goal/control checkpoint recovery was completed by later commits.
- The staging recipe below is retained only to document how that recovery was
  performed.

## Phase 1: Paper02 Submodule Planning Checkpoint

Status: completed at submodule SHA `205aaea`.

Executed from the parent repository root:

```bash
git -C paper/UXFD_paper/1D-2D_fusion_explainable status --short -- plan/EXPERIMENT_PLAN_补充.md program.md
git -C paper/UXFD_paper/1D-2D_fusion_explainable diff -- plan/EXPERIMENT_PLAN_补充.md program.md
git -C paper/UXFD_paper/1D-2D_fusion_explainable add -- plan/EXPERIMENT_PLAN_补充.md program.md
git -C paper/UXFD_paper/1D-2D_fusion_explainable diff --cached --check
git -C paper/UXFD_paper/1D-2D_fusion_explainable diff --cached --stat
git -C paper/UXFD_paper/1D-2D_fusion_explainable commit -m "docs: add fusion experiment supplement plan"
git -C paper/UXFD_paper/1D-2D_fusion_explainable rev-parse --short HEAD  # 205aaea
```

Do not stage Paper02 manuscript drafts, model weights, experiment outputs, or
unreviewed scripts in this checkpoint.

## Phase 2: Parent Matrix And Report Sync

Paper02 truth-first cleanup has now been committed as a separate submodule
checkpoint:

```bash
git -C paper/UXFD_paper/1D-2D_fusion_explainable add -- \
  scripts/sync_truth_first_manuscript.py \
  scripts/truth_audit.py \
  manuscript/final_tex/main.tex \
  manuscript/paper.md \
  paper_draft/NMI_Paper1_Fusion1D2D.tex \
  manuscript/experiments.md
git -C paper/UXFD_paper/1D-2D_fusion_explainable diff --cached --check
git -C paper/UXFD_paper/1D-2D_fusion_explainable diff --cached --name-status
git -C paper/UXFD_paper/1D-2D_fusion_explainable commit -m "docs: clean paper02 truth-first manuscript surfaces"
git -C paper/UXFD_paper/1D-2D_fusion_explainable rev-parse --short HEAD  # da9f3b9
```

Sync the parent matrix and reports against Paper02 submodule SHA `df3e579`.
Keep accepted baseline, ablation, TOP representative, GPU metadata, and SOTA
gates blocked unless real accepted artifacts exist.

Historical note: Phase 1 recorded Paper02 submodule SHA `205aaea`; later
Paper02 checkpoints have advanced the submodule beyond that planning-only SHA.

1. Update `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`:
   - replace stale Paper02 submodule SHA references with the latest committed
     Paper02 cleanup checkpoint SHA.
   - reference the truth-first manuscript cleanup as a committed checkpoint.
   - keep accepted baseline, ablation, TOP representative, GPU metadata, and
     SOTA gates blocked unless real accepted artifacts exist.
2. Regenerate parent reports:

```bash
python -m scripts.uxfd_submission_gate --format json --allow-not-ready --output paper/UXFD_paper/results/submission_gate_current.json
python -m scripts.uxfd_submission_gate --format markdown --allow-not-ready --output paper/UXFD_paper/results/submission_gate_current.md
python -m scripts.uxfd_readiness_backlog --format markdown --allow-not-ready --output paper/UXFD_paper/results/readiness_backlog.md
python -m scripts.uxfd_submodule_dirty_triage --format markdown --output paper/UXFD_paper/results/submodule_dirty_triage.md
```

`scripts.uxfd_submodule_dirty_triage` may exit non-zero while residual dirty
submodule files remain; inspect the generated report instead of treating that as
a script crash.

## Phase 3: Parent Checkpoint Commit

Status: historical/completed.

The following staging set was used during recovery for the parent goal/control
files plus the Paper01, Paper02, and Paper04 submodule gitlinks. Do not rerun it
without rechecking the current dirty tree:

```bash
git add -- \
  .claude/handoffs/2026-05-12-uxfd-goal-continuation.md \
  paper/UXFD_paper/Explainable_FD_Toolkit \
  paper/UXFD_paper/1D-2D_fusion_explainable \
  paper/UXFD_paper/MOE_explainable \
  paper/UXFD_paper/goal/README.md \
  paper/UXFD_paper/goal/99_submission_readiness_matrix.md \
  paper/UXFD_paper/results/gpu_queue_live_preflight.json \
  paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md \
  paper/UXFD_paper/results/accepted_run_templates \
  paper/UXFD_paper/results/artifact_gate_queue_coverage.md \
  paper/UXFD_paper/results/queue_launch_plan.sh \
  paper/UXFD_paper/results/queue_launch_shards/gpu0.sh \
  paper/UXFD_paper/results/submission_gate_current.json \
  paper/UXFD_paper/results/submission_gate_current.md \
  paper/UXFD_paper/results/submodule_dirty_triage.md \
  paper/UXFD_paper/results/goal_clarity_audit_current.md \
  paper/UXFD_paper/results/low_tier_source_audit.md \
  paper/UXFD_paper/results/low_tier_source_audit.json \
  paper/UXFD_paper/results/commit_recovery_plan.md \
  specs/006-uxfd-ieee-trans-submission-readiness/tasks.md \
  scripts/uxfd_artifact_gate.py \
  scripts/uxfd_artifact_scaffold.py \
  scripts/uxfd_low_tier_source_audit.py \
  scripts/uxfd_objective_audit.py \
  scripts/uxfd_readiness_backlog.py \
  scripts/uxfd_submission_gate.py \
  scripts/uxfd_submodule_dirty_triage.py \
  test/test_uxfd_artifact_gate.py \
  test/test_uxfd_artifact_scaffold.py \
  test/test_uxfd_low_tier_source_audit.py \
  test/test_uxfd_gpu_queue.py \
  test/test_uxfd_paper01_control_docs.py \
  test/test_uxfd_paper02_runner_policy.py \
  test/test_uxfd_paper04_runner_policy.py \
  test/test_uxfd_objective_audit.py \
  test/test_uxfd_readiness_backlog.py \
  test/test_uxfd_submission_gate.py \
  test/test_uxfd_submodule_dirty_triage.py \
  test/test_uxfd_goal_clarity.py
```

Do not stage:

- `paper/UXFD_paper/results/figures/`
- unrelated parent repository edits
- unreviewed Paper01/Paper02/Paper04 generated results or manuscript drafts

Validate and commit:

```bash
git diff --cached --check
python -m pytest -q test/test_uxfd_goal_clarity.py test/test_uxfd_objective_audit.py test/test_uxfd_submission_gate.py test/test_uxfd_gpu_queue.py test/test_uxfd_low_tier_source_audit.py test/test_uxfd_paper01_control_docs.py
git add -f -- paper/UXFD_paper/results/low_tier_source_audit.json
git status --short -- paper/UXFD_paper/Explainable_FD_Toolkit paper/UXFD_paper/1D-2D_fusion_explainable paper/UXFD_paper/MOE_explainable paper/UXFD_paper/goal paper/UXFD_paper/results scripts/uxfd_low_tier_source_audit.py scripts/uxfd_objective_audit.py scripts/uxfd_readiness_backlog.py scripts/uxfd_submission_gate.py scripts/uxfd_submodule_dirty_triage.py test/test_uxfd_artifact_gate.py test/test_uxfd_goal_clarity.py test/test_uxfd_gpu_queue.py test/test_uxfd_low_tier_source_audit.py test/test_uxfd_paper01_control_docs.py test/test_uxfd_paper02_runner_policy.py test/test_uxfd_paper04_runner_policy.py test/test_uxfd_objective_audit.py test/test_uxfd_readiness_backlog.py test/test_uxfd_submission_gate.py test/test_uxfd_submodule_dirty_triage.py
git commit -m "chore: sync paper02 planning and UXFD goal audits"
```

Do not include `paper/UXFD_paper/results/objective_audit_current.json`,
`paper/UXFD_paper/results/objective_audit_current.md`, or
`paper/UXFD_paper/results/readiness_backlog.md` in this commit. Those reports
include a parent-checkpoint cleanliness item, so they must be regenerated after
this commit lands.

## Phase 4: Objective Audit Refresh Commit

Status: historical/completed.

The following commands describe the audit refresh sequence after Phase 3:

```bash
python -m scripts.uxfd_objective_audit --format json --allow-not-achieved --output paper/UXFD_paper/results/objective_audit_current.json
python -m scripts.uxfd_objective_audit --format markdown --allow-not-achieved --output paper/UXFD_paper/results/objective_audit_current.md
python -m scripts.uxfd_readiness_backlog --format markdown --allow-not-ready --output paper/UXFD_paper/results/readiness_backlog.md
python -m pytest -q test/test_uxfd_objective_audit.py test/test_uxfd_readiness_backlog.py
git add -- paper/UXFD_paper/results/objective_audit_current.json paper/UXFD_paper/results/objective_audit_current.md paper/UXFD_paper/results/readiness_backlog.md
git diff --cached --check
git commit -m "docs: refresh UXFD objective and readiness audits"
```

At this point, `parent UXFD goal-control checkpoint committed` should report
`met` unless a new goal/control file was edited after Phase 3.

## Completion Reminder

Even after these commits, the active goal is not complete until:

- local GPUs `0,1` pass preflight,
- accepted experiment artifacts exist under `paper/UXFD_paper/results/accepted_runs`,
- TOP representative evidence is accepted,
- all seven paper matrices become `submission_ready: true`,
- objective and submission gates pass without `--allow-*`.

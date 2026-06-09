# Quickstart: PHM-GenBench Frontier

## Validate Existing Repo Gates

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m pytest test/
```

Use the project `LQ_signal` environment for the full repository test gate; the
base Python environment may not have optional test dependencies such as
`torchmetrics`. The Speckit prerequisite helper may reject the current
`Feature_factory-update` branch name even when this feature directory and its
checklists are present. Treat that as a branch-name caveat, not as M2 evidence
completion.

## Preflight Gates For GOAL-GEN

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
```

## Minimal Evidence Loop

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override trainer.num_epochs=1

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=sample \
  --override task.generative.allow_untrained_smoke=true \
  --override task.generative.condition_sampling_policy=grid

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=eval \
  --override task.generative.generated_path=<samples.pt>

python -m scripts.paperpack_generative --run_dir <run_dir>
```

The paperpack writes `figure_sources/manifest_index.json` with both synthetic
manifest paths and metric source paths. For M2 paper claims, run the guarded
draft generator against benchmark-effect summary/manifest files; it writes
`PAPER_DRAFT.md`, `evidence_gaps.md`, and `submission_readiness.md` together.

## Claude Teams Review

Use read-only plan/review mode first. Before launch, verify that the configured
Claude endpoint is approved for the workspace content in scope. If approval is
not available, do not launch Claude Teams; record `BLOCKED_NOT_RUN` under
`specs/002-phm-genbench-frontier/reviews/claude-team/<run-id>/` and continue
with local Codex verification.

Canonical review artifacts belong under `specs/002-phm-genbench-frontier/`.
`.codex/claude-team-runs/` may be used only as tool scratch or a mirror.

Example launcher after endpoint approval:

```bash
python /home/user/.codex/skills/claude-code-teams/scripts/launch_claude_team.py \
  --mode review \
  --objective "Review PHM-GenBench frontier roadmap for model coverage, benchmark evidence, and implementation risks" \
  --paths ".specify/goals,specs/002-phm-genbench-frontier,src/task_factory/Components/generative,src/task_factory/task/generative,src/model_factory/generative_model,configs/paper/phm_generative,scripts" \
  --teammates 3 \
  --permission-mode plan
```

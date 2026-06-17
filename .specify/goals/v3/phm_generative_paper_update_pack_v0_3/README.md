# PHM-Vibench Generative Paper Update Pack v0.3

Date: 2026-06-09

This update pack is written for the current `Feature_factory-update` branch,
which has moved beyond the v0.2 baseline.  The branch now contains:

- `main.py` strict preflight and a pipeline whitelist.
- `Pipeline_06_generative.py` with `train / sample / eval` modes.
- Generative task registrations for CFM, Rectified Flow, DDPM epsilon,
  Score SDE, and exploratory one-step families.
- Generative model backbones including MLP1D, UNet1D, DiT-style 1D,
  and SSM/Mamba-style placeholders.
- Synthetic-data manifest evidence gates.
- Generative benchmark-effect, sweep, paperpack, and paper-draft scripts.
- Six-dataset paper matrix targeting `CWRU / XJTU / FEMTO / UNSW / JUST / PU`.
- `.specify/goals/v2/` and `specs/002-phm-genbench-frontier/` process artifacts.

v0.3 is not a "start from scratch" pack.  It is a stabilization and
paper-readiness pack.  The key rule is:

```text
Freeze the existing generative direction.
Do not add more methods until evidence, manifests, metrics, run plans, and
paperpack paths are coherent.
```

## Recommended installation location

Copy this pack into:

```text
specs/002-phm-genbench-frontier/v0_3_update_pack/
```

or keep it outside the repo and feed the files to Codex/Claude Code as a
planning pack.

## What this pack provides

```text
00_BRANCH_STATE_AUDIT.md
01_PAPER_READINESS_GAPS.md
02_V3_ARCHITECTURE_CONTRACT.md
03_PIPELINE_GUIDE.md
04_MODEL_FACTORY_GUIDE.md
05_LOSS_TASK_REGISTRY_GUIDE.md
06_CONFIG_AND_MATRIX_GUIDE.md
07_TRAINING_EVIDENCE_GUIDE.md
08_EVAL_METRICS_GUIDE.md
09_PAPERPACK_GUIDE.md
10_PR_SPLIT_PLAN.md
11_CODEX_CLAUDE_HANDOFF.md
12_LITERATURE_MAP_2026.md
13_FINAL_REPO_TARGET.md
templates/
prompts/
checklists/
schemas/
```

## v0.3 operating policy

1. Keep `python main.py --config <yaml>` as the maintained entry path.
2. Keep the five config blocks: `environment / data / model / task / trainer`.
3. Keep runtime code under the existing factories.
4. Treat `Score SDE / MeanFlow / Drifting / Transition Flow / OT-NFM` as
   exploratory until method-specific evidence exists.
5. Treat CFM, Rectified Flow, and DDPM as the minimum paper baselines.
6. Do not mark anything `benchmark-valid` unless the full evidence chain is present.
7. Do not treat `paperpack` output as submission-ready unless the submission
   readiness gates pass.

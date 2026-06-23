---
name: phm-vibench-hydra-execution-machine
description: >-
  Use for implementing the PHM-Vibench Hydra execution-machine transformation:
  full Hydra config groups/defaults migration, fail-fast pipeline/runtime semantics,
  explicit contrastive objectives, stable run artifacts, demo matrix, and CI gates.
---

# PHM-Vibench Hydra Execution Machine

Use this skill when working on the PHM-Vibench transformation tracked under
`docs/ignore/04_16/hydra_execution_machine/`.

## Start Protocol

1. Read `docs/ignore/04_16/hydra_execution_machine/progress/MASTER.md`.
2. Read the active phase file linked from MASTER.
3. Read the relevant analysis/plan document only as needed:
   - `analysis/risk-assessment.md` for hotspots and priorities.
   - `plan/task-breakdown.md` for task dependencies and acceptance criteria.
   - `plan/dependency-graph.md` for parallel lane constraints.
4. Work only on the active task unless the user explicitly redirects.
5. After completing a task, update both the active phase file checkbox and MASTER phase count/current status.

## Task Goal

Turn PHM-Vibench into a scientific execution machine:

```text
config -> preflight -> pipeline -> trainer -> artifacts
```

There must be no implicit demo, implicit pipeline fallback, silent zero-loss target,
or missing parent-consumable manifest on maintained paths.

## S.U.P.E.R Architecture Principles

Write code like building with LEGO: each brick has a single job, a standard
interface, a clear direction, runs anywhere, and can be swapped.

### S — Single Purpose

- Each module, file, and function solves exactly one problem.
- Prefer decomposition; power comes from composition.
- Litmus test: if you cannot describe a module's responsibility in a single
  sentence, split it.

### U — Unidirectional Flow

- Data flows input -> processing -> output.
- Dependencies point inward; outer layers depend on inner layers.
- No reverse dependencies, no circular calls.
- Litmus test: can core logic run unit tests without external services?

### P — Ports over Implementation

- Define interface contracts before implementation.
- Use serializable data structures or schemas at module boundaries.
- Swapping an implementation must not require reading internal code to infer the format.

### E — Environment-Agnostic

- Configuration is injected through config files, environment variables, or CLI overrides.
- No hardcoded machine paths in committed demos.
- Dependencies are explicitly declared.
- Same codebase should run on another machine without source edits.

### R — Replaceable Parts

- Any layer can be replaced without affecting unrelated layers.
- Replacement cost is the architecture metric.
- If replacing one component triggers cascading changes, the boundary is wrong.

## S.U.P.E.R Code Review Checklist

Run this before marking any task complete:

1. Does each touched function/module have one clear responsibility?
2. Does runtime still flow `config -> preflight -> pipeline -> trainer -> artifacts`?
3. Are config inputs validated before trainer construction?
4. Are new cross-module inputs/outputs schema-defined or serializable?
5. Are all task-critical errors raised instead of warning-and-continuing?
6. Are machine-local paths absent from committed demos/configs?
7. Is the public CLI contract preserved or explicitly deprecated?
8. Can factories keep stable inputs while Hydra migration proceeds?
9. Does the task add or update focused tests for its acceptance criteria?
10. Did you update progress docs immediately after completion?

Scoring rule: all pass = proceed; 1-2 fail = fix before marking done; 3+ fail =
stop and refactor the task boundary.

## Implementation Rules

- Keep `python main.py --config <yaml> [--override key=value ...]` as the public command.
- `--config_path` is compatibility only and must warn.
- No `main.py` default config or default pipeline.
- Hydra/OmegaConf is the target config compose layer; Pydantic remains final schema validation.
- Add a compatibility bridge instead of deleting current YAML demos in the first pass.
- Training-critical failures must raise:
  - invalid config or pipeline
  - invalid data path or metadata
  - invalid shape/device/label state
  - invalid contrastive pairing
  - failed contrastive loss computation
- Best-effort is allowed only for optional artifacts such as explain/distilled/predictions. Manifest itself is required.
- Preserve unrelated dirty files and submodule state. Do not revert user changes.

## Known Hotspots

- `main.py`: implicit default demo and pipeline fallback.
- `src/Pipeline_02_pretrain_fewshot.py`: implicit single/staged/legacy mode and broad fallback.
- `src/Pipeline_03_*`, `Pipeline_04`, `Pipeline_05`: fallback patterns need review.
- `src/task_factory/Components/contrastive_losses.py`: unlabeled InfoNCE zero-loss path.
- `src/task_factory/task/pretrain/hse_contrastive.py`: training-critical zero fallback.
- `src/task_factory/Components/contrastive_strategies.py`: catches loss failures and returns zero.
- `src/trainer_factory/extensions/manifest.py`: manifest schema needs central enforcement.

## Validation Anchors

Use the smallest relevant validation for the active task:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1 --override data.num_workers=0
python -m pytest test/
```

Add task-specific tests for strict CLI, pipeline mode, InfoNCE pairing, HSE fail-fast,
manifest contract, and demo matrix gates as phases require.

## Progress Update Protocol

After a task is complete:

1. Check the task in `docs/ignore/04_16/hydra_execution_machine/progress/phase-*.md`.
2. Update the corresponding phase count in MASTER.
3. Update MASTER Current Status to the next task.
4. Add a short note with changed files and validation results.
5. If all tasks are done, trigger archive planning for the spec-driven artifacts.

# CLAUDE.md

This file documents PHM-Vibench’s intent, architecture, and change strategy. For “how to run” commands, see `AGENTS.md`.

## Repository Overview

PHM-Vibench is a configuration-first benchmark platform for industrial vibration signal analysis (fault diagnosis /
predictive maintenance). The core value is:
- Reproducible experiments (configs are the experiment contract).
- Modular extension (swap datasets/models/tasks/trainers via registries + factories, not by rewriting pipelines).

## Key Architecture Components

### Factory Design Pattern (wiring points)
- `src/data_factory/`: dataset loading, splits, preprocessing, dataloaders.
- `src/model_factory/`: backbones/foundation models + heads (registry-driven).
- `src/task_factory/`: training logic / metrics / losses (LightningModule-like tasks).
- `src/trainer_factory/`: PyTorch Lightning Trainer wiring (callbacks/loggers/devices).

Module deep dives live under:
- `src/data_factory/CLAUDE.md`
- `src/model_factory/CLAUDE.md`
- `src/task_factory/CLAUDE.md`
- `src/trainer_factory/CLAUDE.md`
- Config loader internals: `src/configs/CLAUDE.md`

### Pipeline System (orchestrators)
Pipelines assemble factories in a fixed order (load config → build data → build model → build task → build trainer).
Common pipelines:
- `src/Pipeline_01_default.py`: standard single-stage pipeline.
- `src/Pipeline_02_pretrain_fewshot.py`: pretrain + few-shot pipeline.
- `src/Pipeline_03_multitask_pretrain_finetune.py`: multi-task pretrain/fine-tune pipeline.
- `src/Pipeline_ID.py`: ID-based ingestion variant.

## Configuration System (v5.x)

### Single supported entrypoint (contract)
Run via:
```bash
python main.py --config <yaml> [--override key=value ...]
```
Pipeline is selected by YAML top-level `pipeline:` (not by a `--pipeline` CLI flag).

### 5-block config model
All maintained configs are organized into:
- `environment` / `data` / `model` / `task` / `trainer`

Composition rules (low → high precedence):
1) `base_configs.*` YAML files
2) The demo YAML’s own block overrides
3) Optional machine-local override `configs/local/local.yaml` (or `--local_config ...`)
4) CLI `--override key=value` (repeatable)

### Single Source of Truth (SSOT) + tooling
To reduce ambiguity and make configs traceable:
- Registry (authoritative index): `configs/config_registry.csv`
- Registry schema: `docs/config_registry_schema.md`
- Generated atlas (human-readable): `docs/CONFIG_ATLAS.md` (regen: `python -m scripts.gen_config_atlas`)
- Inspect tool: `python -m scripts.config_inspect` (resolved config + sources + instantiation targets + sanity)
- Schema validate: `python -m scripts.validate_configs` (loader-driven + pydantic; see `src/config_schema/`)

Start here for user-facing config docs: `configs/README.md`.

## Paper / Research Workflows (kept separate)

Paper-grade experiments live in a git submodule to avoid confusing the main onboarding path:
- `paper/2025-10_foundation_model_0_metric/` (init requires network):
  - `git submodule update --init --recursive paper/2025-10_foundation_model_0_metric`
  - See `paper/README_SUBMODULE.md`

Rule: do not make core repo validation depend on paper-only scripts/configs.

### HSE-Prompt research experiments (paper-only)
- Location: `paper/2025-10_foundation_model_0_metric/` (submodule)
- Goal: HSE/HSE-Prompt cross-system generalization studies
- If the submodule is not initialized, use `configs/demo/05_pretrain_fewshot/` and `configs/demo/06_pretrain_cddg/` as
  the runnable reference in this repo.

## Configuration Standards (reduce ambiguity)

### Template source
- Maintained templates: `configs/demo/`
- Local research variants: `configs/experiments/<task_dataset_variant>/`
- Legacy configs (planned migration/removal): `configs/reference/` (do not template from this directory)

### Model component naming
Use registry-style IDs:
- embeddings: `E_**_*`
- backbones: `B_**_*`
- heads: `H_**_*`

Example (correct):
```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
```

Example (incorrect; avoid ambiguous names):
```yaml
model:
  embedding: "HSE"      # ambiguous / not registry-addressed
  backbone: "Dlinear"   # missing prefix (B_**_)
  task_head: "Linear"   # missing prefix (H_**_)
```

### Dataset selection (traceable / reproducible)
Prefer numeric IDs from metadata:
```yaml
task:
  target_system_id: [1, 2]
```
Those IDs should come from the metadata file referenced by `data.metadata_file` (typically the `Dataset_id` column).

How to verify the mapping (example for Excel metadata):
```bash
python - <<'PY'
import pandas as pd
df = pd.read_excel("data/metadata.xlsx")
print(df[["Dataset_id", "Name"]].drop_duplicates().sort_values("Dataset_id"))
PY
```

## Common Pitfalls (and what to do)

### “Which value is actually used?”
Use inspect to get resolved config + field sources:
```bash
python -m scripts.config_inspect --config <yaml> --override key=value
```

### Component import errors
If a model component cannot be imported, it usually means the ID is not registered or mistyped. Use:
```bash
python -m scripts.config_inspect --config <yaml> --dump targets --format yaml
```

### Dataset / path errors
- Avoid hard-coded absolute paths in committed configs.
- Put machine-specific paths into `configs/local/local.yaml` or pass `--override data.data_dir=/path/to/...`.

## Common Development Commands

For practical runbook and copy-paste commands, see [@AGENTS.md - Quick Commands].

Key commands include:
- Smoke test: `python main.py --config configs/demo/00_smoke/dummy_dg.yaml`
- Config inspect: `python -m scripts.config_inspect --config <yaml>`
- Validate: `python -m scripts.validate_configs`
- Tests: `python -m pytest test/`

## Results and Output (where files go)
- Default base directory is `save/`.
- If `environment.output_dir` is set, outputs go under that directory instead (many demos use `results/demo/...`).
- Final run directory is `base_dir/<experiment_name>/iter_<k>/` (see `src/configs/config_utils.py:path_name`).

Typical artifacts depend on trainer/task, but usually include checkpoints, metrics/logs, and a config snapshot.

## Model Architecture (high-level map)

### Foundation models (ISFM family)
- `M_01_ISFM`, `M_02_ISFM`, `M_03_ISFM`

### Common backbones (examples)
- `B_04_Dlinear`, `B_06_TimesNet`, `B_08_PatchTST`, `B_09_FNO`

### Heads (examples)
- `H_01_Linear_cla` (classification), `H_03_Linear_pred` (prediction)

## Task Types (high-level map)
- Classification / DG / CDDG (domain generalization)
- FS / GFS (few-shot and generalized few-shot)
- Pretrain (self-supervised / contrastive pretraining)

## Environment Setup (practical notes)
- Core dependencies are listed in `requirements.txt`.
- Keep machine-specific paths out of committed configs; use `configs/local/local.yaml` or CLI `--override`.

## Important Notes (how to keep the repo maintainable)
- Always register new components via the appropriate factory/registry instead of hardcoding imports in pipelines.
- Keep demos runnable with minimal assumptions; prefer smoke-friendly defaults (e.g. `--override trainer.num_epochs=1`).
- When changing config structure, update SSOT (registry/atlas) and add a migration note if any user-facing key changes.
- Vibecoding (AI-assisted changes): default to the simplest viable implementation. Avoid over-engineering and
  unnecessary defensive design; apply Occam’s razor; reason from first principles; iterate incrementally.

## Hard Constraints (Do Not Break)
- Do not introduce breaking changes to `main.py` public CLI or core YAML keyspaces (`environment/data/model/task/trainer`)
  without a compatibility layer + migration notes.
- Docs must be “本科生能跑 + 博士生能改”: minimal runnable commands first, then field explanation + pitfalls + extension
  points, with links to consumer code.
- Config fields must be traceable: any key should answer “final value comes from where” + “consumed where”.

## Required Change Order (stable)
1) Registry → 2) Atlas → 3) Inspect → 4) Schema validate → 5) `configs/**/README.md` → 6) CI/tests + acceptance

## PR/Step Review Checklist
- Change list: what changed and why (tight scope).
- How to validate: copy-paste commands from `AGENTS.md`.
- Expected outputs: generated files + output directory pattern.

# Repository Guidelines (AGENTS)

This file is a practical runbook + double-check list for working in PHM-Vibench. For change strategy/constraints, see
`CLAUDE.md`.

## Project Meaning (what to remember)
- Config-first benchmark: experiments are defined by YAML configs (environment/data/model/task/trainer).
- Modular wiring: factories under `src/*_factory/` assemble data/model/task/trainer from registries.
- Single maintained entrypoint: `python main.py --config <yaml> [--override key=value ...]` (pipeline via YAML
  `pipeline:`).

## Quick Commands (copy-paste)
```bash
# Offline smoke run (repo-shipped dummy data)
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# Validate config schema (demos + active registry rows)
python -m scripts.validate_configs

# Inspect resolved config / field sources / instantiation targets
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1

# Registry → Atlas (docs/CONFIG_ATLAS.md must stay in sync)
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md

# Maintained tests
python -m pytest test/
```

## Project Structure & Module Organization
- `src/`: runnable pipelines + factories; extend via the matching factory to preserve modular wiring.
- `configs/`: experiment YAMLs
  - templates: `configs/demo/`
  - local variants: `configs/experiments/<task_dataset_variant>/`
  - legacy: `configs/reference/` (planned migration/removal; do not template from it)
- Runtime assets: raw inputs in `data/`, results in `save/` or `environment.output_dir`, visuals in `pic/`, docs in
  `docs/`.
- Tests: maintained suite lives in `test/` (optional legacy runner: `dev/test_history/`).

## Architecture Highlights
- Factory pattern with registries for data, models, tasks, and trainers.
- Pipelines: `Pipeline_01_default`, `Pipeline_02_pretrain_fewshot`, `Pipeline_03_multitask_pretrain_finetune`,
  `Pipeline_ID`.
- Config tooling: registry (`configs/config_registry.csv`) → atlas (`docs/CONFIG_ATLAS.md`) → inspect/validate scripts.

## Configuration System (what to enforce)
- Keep the 5-block model: `environment/data/model/task/trainer`.
- Prefer composable configs (`base_configs + overrides`) and CLI dot overrides.
- Keep configs traceable:
  - add maintained demos to `configs/config_registry.csv`
  - regenerate atlas with `python -m scripts.gen_config_atlas`
  - validate with `python -m scripts.validate_configs`

## Dataset Integration (how to extend)
- Raw inputs: `data/raw/<dataset_name>/`
- Metadata: spreadsheets (`metadata_*.xlsx`) or repo-provided CSV for smoke demos.
- Implement readers by inheriting `BaseReader` and register them in `src/data_factory/__init__.py`.
- Reader examples: `src/data_factory/reader/RM_*.py` and `src/data_factory/reader/Dummy_Data.py`.

## Model and Task Registry (naming discipline)
- Components are registry-addressed (e.g., `E_01_*`, `B_04_*`, `H_01_*`, `M_01_*`).
- Tasks are wired via `src/task_factory/task_registry.csv` (inspect tool shows target paths).

## Development Commands
- Smoke: `python main.py --config configs/demo/00_smoke/dummy_dg.yaml`
- Baseline demo: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml --override trainer.num_epochs=1`
- Streamlit UI: `streamlit run streamlit_app.py` (experimental; not a validation gate).

## Style and Testing
- Style: PEP 8, 100-char line limit; format with `black src/ test/` and `isort src/ test/` if available.
- Testing:
  - Maintained: `python -m pytest test/`
  - Legacy runner (optional): `python dev/test_history/run_tests.py --unit`

## Commit & PR Guidelines (keep changes reviewable)
- Keep changes focused (configs vs factories vs docs should be separable when possible).
- Vibecoding (AI-assisted changes): keep it simple (KISS). Avoid over-engineering, premature abstractions, and
  unnecessary defensive design; apply Occam’s razor; work from first principles; develop incrementally.
- Every PR/step should include:
  - What changed + why
  - How to validate (commands above)
  - Expected outputs (e.g. `docs/CONFIG_ATLAS.md` updated, output directory pattern)

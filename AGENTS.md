# Repository Guidelines (AGENTS)

This file is a practical runbook + double-check list for working in PHM-Vibench. For change strategy/constraints, see
`CLAUDE.md`. For the canonical project overview + onboarding path, see `README.md` (and `configs/README.md` for the
config system).

## Scope (what this file is for)
- Copy-paste commands and validation gates for day-to-day work.
- “What changed, and how do I prove it works?” checklist items.

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

# Validate documentation links / @README conventions
python -m scripts.validate_docs

# Maintained tests
python -m pytest test/
```

## Where to Work (quick map)
- `configs/demo/`: maintained runnable templates (copy from here)
- `configs/experiments/`: your local experiment variants
- `configs/reference/`: legacy (do not template from here)
- `src/*_factory/`: extension points (data/model/task/trainer wiring)
- `docs/`: maintained documentation; `docs/CONFIG_ATLAS.md` is generated from the registry

## Config Traceability (SSOT)
- Registry (authoritative index): `configs/config_registry.csv`
- Generated atlas (human-readable): `docs/CONFIG_ATLAS.md` (regen: `python -m scripts.gen_config_atlas`)
- Inspect: `python -m scripts.config_inspect` (resolved config + sources + instantiation targets)
- Validate: `python -m scripts.validate_configs` (loader + schema)

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

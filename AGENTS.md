# Repository Guidelines

## Project Structure & Module Organization
Framework code lives in `src/` with four core factories: `data_factory/`, `model_factory/`, `task_factory/`, and `trainer_factory/`. Configuration YAMLs sit in `configs/` (grouped by dataset, pipeline, and experiment), while automation utilities are under `script/`. Tests primarily live in `test/`; integration flows (e.g., `test_flow_*.py`) stay at the repo root. Shared assets and outputs belong in `data/`, `results/`, and `save/`. Update related docs in `doc/` or `docs/` when adding new capabilities.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`; add `-r requirements-test.txt` when working on the harness. Launch training via `python main.py +experiment=<config>` and queue batch runs with `python run_flow_experiment_batch.py --config-name <name>`. Run targeted validations using `python run_tests.py --pattern test_flow_*` and execute the full suite with `pytest`. Use `validate_flow_setup.py` after major config edits to verify entry points.

## Coding Style & Naming Conventions
Follow PEP 8, four-space indentation, and Black-compatible formatting; type hints are expected on new public APIs. Keep modules, configs, and Hydra groups in `snake_case` (or lowercase dash-separated), while classes stay `CamelCase` and functions `snake_case`. Document non-obvious PHM assumptions in short docstrings or inline comments. When extending factory registries, replicate the existing registration helper patterns.

## Testing Guidelines
Name tests `test_*.py` and colocate fixtures with the functionality they exercise. Cover factory registration, data loading, and trainer loops with fast smoke tests; tag slow or benchmark-heavy cases with `pytest.mark.slow` so they can be skipped in CI. Ensure deterministic behavior by seeding RNGs inside fixtures and reflecting any new CLI flags in `pytest.ini` when needed.

## Commit & Pull Request Guidelines
Write imperative, under-72-character commit subjects (e.g., "Add reader cache") and group related changes together. Pull requests should link issues, outline behavioral impact, mention required config migrations, and note any doc updates. Include metric tables or screenshots when altering dashboards or benchmark outputs, and state whether the relevant `pytest` targets were executed.

## Configuration & Experiment Tips
Treat Hydra configs as reusable templates: copy from `configs/benchmarks/` or `configs/pipelines/` and override parameters with CLI flags. Log reproducibility metadata—seed, config path, artifact location—in commit bodies or PR notes. Leverage existing W&B or TensorBoard hooks exposed by the trainer factories to keep long-running experiments auditable.

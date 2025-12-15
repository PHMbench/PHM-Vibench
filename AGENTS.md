# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds runnable pipelines and the data/model/task/trainer factories; implement new logic in the matching factory to preserve modular wiring.
- `configs/` stores experiment YAMLs—start from a demo under `configs/demo/` (e.g. `configs/demo/01_cross_domain/cwru_dg.yaml`), and keep local variants under a dedicated subfolder.
- Runtime assets stay outside Git: raw inputs in `data/`, results in `save/`, visuals in `pic/`, docs in `docs/`; active tests live in `test/` while legacy stress suites remain in `tests/`.
- see @CLAUDE.md for better understanding of the Vibench.

## Architecture Highlights
- Factory pattern with registries for data, models, tasks, and trainers (`src/*_factory/CLAUDE.md` for deep dives).
- Pipelines include `Pipeline_01_default`, `Pipeline_02_pretrain_fewshot`, `Pipeline_03_multitask_pretrain_finetune`, and `Pipeline_ID`.
- Configuration-first design via `load_config()` supporting presets, YAML files, dictionaries, and `ConfigWrapper` overrides.
- Save artifacts under `save/{metadata}/{model}/{task_trainer_timestamp}/` with checkpoints, metrics, logs, figures, and config backup.

## Configuration System
- Unified loader handles preset aliases plus recursive dot-notation overrides: `load_config('isfm', {'model.d_model': 512})`.
- Keep YAML keys lowercase with hyphen-separated values to match samples in `configs/demo/`.
- Pipelines read full experiment context from config; avoid hard-coded paths or hyperparameters.
- Extended guide in `src/configs/CLAUDE.md` covers chaining (`copy().update()`), multi-stage pipelines, and override precedence.

## Dataset Integration
- Raw inputs belong in `data/raw/<dataset_name>/` with metadata spreadsheets (`metadata_*.xlsx`) and processed H5 files.
- Implement readers by inheriting `BaseReader` and register them inside `src/data_factory/__init__.py`.
- Reference examples in `src/data_factory/reader/RM_*.py` and document quirks in dataset-specific notes.
- (TODO) Research/paper-specific validation scripts are planned to live in the paper submodule; keep core repo validation via `python -m pytest test/`.

## Model and Task Registry
- Foundation models live under `model_factory` (e.g., `M_01_ISFM`, `M_02_ISFM`, `M_03_ISFM`) alongside backbone networks (`B_08_PatchTST`, `B_09_FNO`, etc.).
- Attach heads such as `H_01_Linear_cla` or `H_03_Linear_pred` for classification vs prediction workloads.
- Tasks cover classification, cross-dataset domain generalization, few-shot (FS/GFS), and pretraining—wire them via `task_factory`.
- Trainer implementations extend PyTorch Lightning; keep Lightning callbacks and loggers configurable.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` then `pip install -r requirements.txt`.
- Run baselines with `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`; use `configs/demo/` as the template source for new configs.
- `streamlit run streamlit_app.py` is an experimental UI (TODO: visualization is incomplete); do not rely on it for automated validation.
- Use `python -m pytest test/` for routine checks.

## Coding Style & Naming Conventions
- Follow PEP 8 with a 100-character limit, grouped imports, and NumPy-style docstrings for any public API.
- Classes use `PascalCase`, functions and variables `snake_case`, constants `UPPER_CASE`, and config folders follow `task_dataset_variant`.
- Format before committing: `black src/ test/`, `isort src/ test/`; enforce linting with `flake8` and static checks through `mypy src/`.
- Keep YAML keys lowercase with hyphen-separated values to match the existing samples in `configs/demo/`.

## Testing Guidelines
- The maintained pytest suite sits in `test/` with unit, integration, and performance markers; migrate refreshed stress tests from `tests/` as they stabilise.
- Name files `test_<feature>.py`, tag long cases `@pytest.mark.slow`, and guard GPU paths with `@pytest.mark.gpu` to keep automation green.
- Target coverage on critical pipelines via `pytest --cov=src --cov-report=term`, and note accuracy or latency outcomes alongside the command in pull requests.

## Commit & Pull Request Guidelines
- Mirror recent history: imperative subjects, optional scoped prefixes (`docs(hse):`, `refactor:`), and keep English summaries unless you are updating Chinese-only docs.
- Keep commits focused—split config updates, factory changes, and docs so each diff stays reviewable and reversible.
- PRs should include a problem statement, model/dataset impact summary, reproduction commands, and links to tracked issues.
- Attach artifact paths under `save/<metadata>/<model>/<experiment>` or UI screenshots, and confirm any data-source change complies with `SECURITY.md`.

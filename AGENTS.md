# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds runnable pipelines and the data/model/task/trainer factories; implement new logic in the matching factory to preserve modular wiring.
- `configs/` stores experiment YAMLs—start from `configs/demo/Single_DG/CWRU.yaml`, clone templates in `configs/experiments/`, and keep local variants under a dedicated subfolder.
- Runtime assets stay outside Git: raw inputs in `data/`, results in `save/`, visuals in `pic/`, docs in `docs/`; active tests live in `test/` while legacy stress suites remain in `tests/`.
- see @CLAUDE.md for better understanding of the Vibench.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` then `pip install -r requirements.txt` (add `dev/test_history/requirements-test.txt` when evolving pytest suites).
- Run baselines with `python main.py --config configs/demo/Single_DG/CWRU.yaml`; swap in unified metric configs when reproducing cross-domain benchmarks.
- `python scripts/hse_synthetic_demo.py` validates the HSE pipeline quickly, and `streamlit run streamlit_app.py` launches the monitoring UI for manual QA.
- Use `python -m pytest test/` for routine checks; call `python dev/test_history/run_tests.py --unit` or append `--coverage` when mirroring the historical matrix.

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

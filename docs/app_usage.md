# Streamlit Console Usage

The Streamlit console is experimental and is not a validation gate. Launch it with:

```bash
streamlit run frontend/streamlit_app.py
```

## What It Does

The console is a research control surface, not a second execution stack.

- `Workbench`: maintained demos, recent runs, and recent evidence
- `Compose`: config selection, explicit preflight state, resolved YAML, field sources, sanity checks, and CLI launch
- `Runs`: recent run discovery, shared run filters, and evidence-first detail tabs
- `Compare`: protocol-aware compare guard rails with baseline selection and evidence warnings
- `Registry`: data/model/task/trainer registries and presets
- `Artifacts`: shared run filters, present/missing inventories, and previews for `config_snapshot.yaml`, `test_result_*.csv`, `artifacts/manifest.json`, `figures/`, and related files

## Execution Contract

- The CLI remains authoritative: `python main.py --config <yaml> [--override key=value ...]`
- The frontend only composes, inspects, launches, compares, and traces
- Launch always shows the exact CLI command before execution
- Compare keeps one explicit baseline run and warns when selected evidence is incomplete

## Artifact Contract

The console follows the repo's current run artifacts:

- `config_snapshot.yaml`
- `test_result_*.csv`
- `artifacts/manifest.json`
- `figures/`
- optional `logs/**/metrics.csv`
- optional `artifacts/predictions.npz`

If Streamlit is not installed or fails to start, use the maintained CLI demos under `configs/demo/`.

Legacy Streamlit experiments are archived under `frontend/legacy/`.

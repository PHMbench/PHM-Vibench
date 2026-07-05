# CLAUDE.md — plot_factory

This module gives architecture guidance for the plotting/visualization factory.
For the runnable quick-start, component table, and layout, see [@README.md].

## Intent
Centralize plotting/visualization helpers so they live under `src/` with the
other factories (`data_factory`, `model_factory`, `task_factory`, `trainer_factory`,
`explain_factory`) instead of in a top-level `plot/` directory.

## Architecture
- `plot_config.py` — pure styling helpers. No project deps (matplotlib + seaborn
  + scienceplots only). Safe to import anywhere.
- `registry.py` — `PLOT_REGISTRY` built on `src/utils/registry.py`. Eagerly
  registers the lightweight `P_00_*` helpers; pipelines resolve lazily.
- `plot_factory.py` — façade. `get_plotter(name)` lazily imports pipeline
  modules so importing `plot_factory` does not pull in torch / data factory.
- `pretraining_plot.py` — the runnable prediction-visualization pipeline. Uses
  `src.data_factory.build_data` + `src.model_factory.build_model` (modern API).

## Naming convention
- Helpers: `P_00_<name>`
- Pipelines: `P_0N_<name>` (N >= 1)

## When extending
1. Add the plotting function in a module under `plot_factory/`.
2. If it is a lightweight helper, register it eagerly in `registry.py`.
3. If it is a heavy pipeline (torch/data deps), add a lazy entry to
   `_PIPELINE_MODULES` in `plot_factory.py` instead.
4. Add a row to the README component table.

## Migration provenance
- `plot_config.py`         <- `plot/A1_plot_config.py` (verbatim logic).
- `pretraining_plot.py`    <- `plot/pretraining_plot.py` (import path fixed:
  `plot.A1_plot_config` -> `.plot_config`; save dir `plot/output` ->
  `results/plot_factory/output`).
- `plot/A3-A8, A10`        -> archived to `obsidian/history/dev/plot_legacy/`
  (they imported removed legacy modules `model.*`, `model_collection.*`,
  `trainer.*`, `configs.config`).
- `plot/A9_model_loss_curve_TODO.py` dropped (was a near-empty TODO stub).

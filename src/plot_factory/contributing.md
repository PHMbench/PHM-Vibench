# Contributing — plot_factory

## Adding a new plotter
1. Create `src/plot_factory/<name>.py` with a top-level function (the pipeline).
2. Pick an ID: `P_0N_<name>` (next free `N`).
3. Light helper (no torch / factory deps): register it eagerly in `registry.py`:
   ```python
   PLOT_REGISTRY._items["P_0N_<name>"] = <fn>
   ```
4. Heavy pipeline (torch / data / model deps): add a lazy entry in
   `plot_factory.py::_PIPELINE_MODULES`:
   ```python
   "P_0N_<name>": ("<module>", "<callable>"),
   ```
5. Add a row to the README component table.

## Style rules
- Default output dir: `results/plot_factory/output/` (or env override).
- Always call `configure_matplotlib(...)` before plotting for consistent style.
- Never re-introduce a top-level `plot/` package — keep plotters under `src/`.

## Tests
- Smoke-import the package: `python -c "from src.plot_factory import configure, available_plotters; print(available_plotters())"`
- For the pretraining pipeline, run with a small config + checkpoint.

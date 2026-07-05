# plot_factory

Plotting / visualization utilities for PHM-Vibench. Follows the repo's factory
convention (see `src/data_factory`, `src/model_factory`, `src/explain_factory`).

## Quick start

```python
from src.plot_factory import configure, get_plotter

# Apply publication styling (matplotlib + scienceplots).
configure(style="ieee", font_lang="en")

# Run a registered plot pipeline lazily.
plot = get_plotter("P_01_pretraining_prediction")
plot(args)
```

Or run the pretraining-prediction visualizer as a script:

```bash
python -m src.plot_factory.pretraining_plot \
    --config_path <config.yaml> --ckpt_path <ckpt> --file_ids 10
```

Plots are written under `results/plot_factory/output/` (override with the
`PHM_VIBENCH_PLOT_DIR` env var).

## Layout

- `plot_config.py`     — `configure_matplotlib` / `set_chinese_font` styling helpers.
- `registry.py`        — `PLOT_REGISTRY` of reusable plotters (`P_<id>_<name>`).
- `plot_factory.py`    — `configure`, `get_plotter`, `available_plotters`.
- `pretraining_plot.py`— pretraining-prediction visualization pipeline.

## Component IDs

| ID                          | Kind     | Description                              |
|-----------------------------|----------|------------------------------------------|
| `P_00_configure_matplotlib` | helper   | matplotlib/seaborn styling               |
| `P_00_set_chinese_font`     | helper   | Chinese font registration                |
| `P_01_pretraining_prediction` | pipeline | visualize masked-prediction reconstructions |

## Notes

- Legacy one-off plot scripts (A3-A8, A10) referenced the pre-factory `model.`
  / `model_collection.` / `trainer.` modules that no longer exist; they are
  archived under `obsidian/history/dev/plot_legacy/`, not carried into `src/`.
- `app/` uses `plotly` directly and does not depend on this package.

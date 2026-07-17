# `configs/experiments/`

Local research configs live here.

- Start from a template under `configs/demo/`.
- Copy into a dedicated subfolder (e.g. `configs/experiments/<task_dataset_variant>/...`) and iterate locally.
- If a config becomes a maintained demo, move it into `configs/demo/` and add a row to `configs/config_registry.csv`.

Recommended workflow:

```bash
# Inspect sources/targets before running
python -m scripts.config_inspect --config configs/experiments/<name>/exp.yaml --override trainer.num_epochs=1

# Validate schema (demos are checked by default; add new configs to the registry if you want CI coverage)
python -m scripts.validate_configs
```

Current 2025/2026 experimental contracts:

- `classification/tsl_transformer_dummy.yaml`: clean-room TSL-style classifier;
- `classification/dlinear_fic_dummy.yaml`: CE plus FIC gradient constraint;
- `foundation_models/mantis_v1_local.yaml` and `mantis_v2_local.yaml`: frozen
  local-only adapters with no bundled checkpoints;
- `pretraining/ppt_time_order_univariate.yaml`: backward-compatible time-only PPT;
- `pretraining/ppt_order_multichannel.yaml`: time/channel PPT objective.

The classification templates use `environment.iterations: 5` for seeds 42-46.
Pipeline 01 also writes `all_results.csv` and `run_summary.json` to the common
experiment directory; a single run records `sample_std: null` rather than an
invented dispersion estimate.

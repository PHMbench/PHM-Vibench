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

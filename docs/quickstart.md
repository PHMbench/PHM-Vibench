# Run the First PHM-Vibench Experiment

This page is the canonical first-run guide. It uses the repository-shipped dummy
data and does not require an external dataset or GPU.

## Prerequisites

- repository root is the current working directory;
- Python 3.10 environment is active;
- dependencies are installed as described in [Installation](installation.md).

## 1. Confirm the public entrypoint

```bash
python main.py --help
```

The maintained runtime contract is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

## 2. Validate and inspect the configuration

Validate maintained configuration schemas:

```bash
python -m scripts.validate_configs
```

Inspect the resolved smoke configuration, field sources, import targets, and
sanity checks:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

A non-zero inspector exit code means at least one sanity check failed. Fix that
failure before starting the run.

## 3. Run one offline epoch

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

The maintained smoke config already selects CPU execution and repository-shipped
Dummy data. The explicit overrides keep the command suitable for quick local
verification.

## 4. Confirm the result

A successful command should exit with code `0`, print the final completion
message, and create output below:

```text
results/demo/dummy_dg_smoke/
```

The exact nested run directory and artifacts are controlled by the environment,
trainer, and logger configuration. Treat command success, logs, checkpoints, and
metrics as functional evidence only; the Dummy run is not algorithm-performance
evidence.

Useful checks:

```bash
find results/demo/dummy_dg_smoke -maxdepth 4 -type f | sort
```

Record the repository commit and exact command when sharing the result:

```bash
git rev-parse HEAD
```

## 5. Try a local experiment variant

Do not edit a maintained demo for personal paths or hyperparameters. Copy the
nearest template into `configs/experiments/`:

```bash
cp configs/demo/00_smoke/dummy_dg.yaml \
  configs/experiments/my_dummy_experiment.yaml
```

Inspect and run the copied config:

```bash
python -m scripts.config_inspect \
  --config configs/experiments/my_dummy_experiment.yaml \
  --override trainer.num_epochs=1

python main.py \
  --config configs/experiments/my_dummy_experiment.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

A config belongs under `configs/demo/` only after it has a maintained purpose,
registry entry, documentation, and runtime evidence.

## 6. Use an external dataset

Read the [data directory policy](../data/README.md), then point an existing demo
to a local data root with overrides:

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/PHM-Vibench-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

External-data demos are not offline tests. Record the data source, license,
metadata version, split, preprocessing, seed, and overrides.

## Next steps

- [Configuration system](../configs/README.md)
- [Supported components](../SUPPORTED_COMPONENTS.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [Troubleshooting](troubleshooting.md)
- [Testing and evidence](testing.md)
- [Optional Streamlit workspace](app_usage.md)

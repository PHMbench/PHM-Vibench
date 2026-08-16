# Demo: Dummy DG Smoke (`demo_00_smoke_dummy_dg`)

Purpose: one-command end-to-end run using repository-shipped dummy metadata and raw CSV files.

## Minimal run

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

The installed command is equivalent:

```bash
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
```

## Expected outputs

- Output base directory: `results/demo/dummy_dg_smoke/`
- A subfolder `{experiment_name}/iter_0/` with Lightning logs and checkpoints
- Finite test metrics and an aggregate run summary

## Recommended override

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

## Input contract

The smoke run consumes the packaged files:

```text
data/metadata_dummy.csv
data/raw/Dummy_Data/dummy1.csv
data/raw/Dummy_Data/dummy2.csv
```

Each signal CSV must contain numeric `ch1` and `ch2` columns in that order. Missing,
empty, malformed, or non-finite files fail at the reader boundary. PHMFactory does not
generate substitute signals, guess columns, pad channels, or silently repair the fixture.

## Common failures

1. Running from a different working directory while using repository-relative paths.
2. Deleting `data/metadata_dummy.csv` or either packaged signal CSV.
3. Using a very large `data.batch_size` for the small smoke dataset.
4. Increasing `data.window_size` beyond the available signal length.

This command is an execution smoke. It proves the maintained data → model → task →
trainer path can run; it does not establish a real-data benchmark or algorithm claim.

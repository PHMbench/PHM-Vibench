# Offline Dummy smoke paths

These configurations use repository-shipped metadata and raw CSV files. They require no
external dataset and exercise the same Data, Task, Trainer, and Pipeline contracts.

## ISFM/HSE smoke

```bash
phmfactory --config configs/demo/00_smoke/dummy_dg.yaml
```

This path uses:

```text
model.type=ISFM
model.name=M_01_ISFM
embedding=E_01_HSE
backbone=B_04_Dlinear
```

## Transparent model-replacement smoke

```bash
phmfactory --config configs/demo/00_smoke/dummy_global_average_linear.yaml
```

This path changes only the Model Factory configuration:

```text
model.type=Baseline
model.name=GlobalAverageLinear
```

It is the shortest user-visible proof of the replacement invariant:

```text
replace one model
=> change only the model configuration
```

The transparent model averages each `[B, L, C]` window over time and applies one linear
classification head. It is intended for wiring and protocol diagnosis, not as a claim of
strong diagnostic accuracy.

## Expected outputs

Each command writes below its declared `environment.output_dir` and produces an iteration
directory with Lightning logs, a best checkpoint, finite test metrics, and an aggregate
run summary.

## Input contract

Both paths consume:

```text
data/metadata_dummy.csv
data/raw/Dummy_Data/dummy1.csv
data/raw/Dummy_Data/dummy2.csv
```

Each signal CSV must contain numeric `ch1` and `ch2` columns in that order. Missing,
empty, malformed, or non-finite files fail at the reader boundary. PHMFactory does not
generate substitute signals, guess columns, pad channels, or silently repair the fixture.

## Common failures

1. Running from another working directory while using repository-relative paths.
2. Deleting the packaged metadata or either signal CSV.
3. Using a batch size too large for the small smoke dataset.
4. Increasing `data.window_size` beyond the available signal length.
5. Setting `model.input_dim` to a value other than the actual two channels.

These are execution smokes. They prove bounded Factory assembly and runtime execution;
they do not establish real-data benchmark validity or an algorithm-performance claim.

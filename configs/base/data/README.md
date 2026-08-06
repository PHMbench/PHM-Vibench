# `configs/base/data/`

## What This Block Controls

`data` controls:

- metadata and raw-data locations;
- window generation;
- train/validation/test proportions;
- per-window normalization;
- explicit training augmentation or evaluation corruption;
- DataLoader batch and worker settings.

## Maintained Example

```yaml
data:
  data_dir: "/path/to/PHM-Vibench"
  metadata_file: "metadata.xlsx"
  batch_size: 256
  num_workers: 8

  window_size: 4096
  window_sampling_strategy: "evenly_spaced"
  num_window: 64

  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  unused_ratio: 0.0

  normalization: "per_window_standardization"
  train_noise_snr: null
  evaluation_noise_snr: null
```

## Windowing

| Field | Meaning |
|---|---|
| `window_size` | Number of raw samples in one window; must be positive. |
| `num_window` | Maximum/generated window count; must be positive. |
| `window_sampling_strategy` | `evenly_spaced`, `random`, or `sequential`. |
| `stride` | Used only by `sequential`; it is invalid for other strategies. |
| `window_sampling_seed` | Seed used by deterministic random window selection. |

`evenly_spaced` distributes `num_window` starts across the complete source file. It
does not consume `stride`.

## Split Proportions

```text
train_ratio + val_ratio + test_ratio + unused_ratio = 1
```

Use `unused_ratio` only when data is intentionally reserved. An unnamed remainder is
not accepted.

Window-list disjointness does not imply independent raw samples or independent files.
The default data factory prints the actual raw-interval and file-overlap facts it can
resolve.

## Normalization

`per_window_standardization` computes the mean and standard deviation from each window
itself, including validation and test windows. This is test-sample adaptive
preprocessing and removes absolute offset and scale information. It is not equivalent
to train-fitted dataset normalization.

Accepted maintained values:

```text
per_window_standardization
per_window_minmax
none
```

The historical names `standardization` and `minmax` remain implementation aliases for
older configs, but maintained configs use the explicit per-window names.

## Noise

Noise is never applied through a shared ambiguous `noise_snr` field.

```yaml
data:
  train_noise_snr: 20       # training windows only
  evaluation_noise_snr: 10  # validation/test windows only
  evaluation_noise_seed: 42 # deterministic evaluation corruption
```

Configure only the split whose scientific protocol requires noise. Invalid SNR values,
non-finite signals, and zero-power signals with requested SNR augmentation fail at the
data boundary instead of silently returning an unmodified window.

## Typical Overrides

```bash
python main.py --config <yaml> --override data.data_dir=/path/to/PHM-Vibench
python main.py --config <yaml> --override data.metadata_file=metadata.xlsx
python main.py --config <yaml> --override data.num_workers=0
```

## How to Extend

- Add a reader under `src/data_factory/reader/` whose module name matches metadata
  `Name`.
- Register the task-specific dataset adapter through `register_dataset_adapter(...)`.
- Do not modify model, task, trainer, Pipeline, or CLI code for a compatible dataset.

## Common Failures

1. `window_size` exceeds the source signal length.
2. `stride` is supplied for a non-sequential strategy.
3. split ratios do not account for all data.
4. metadata `Name` or raw file path does not resolve.
5. reader output contains non-numeric, NaN, or Inf values.
6. requested noise cannot be applied to a zero-power signal.

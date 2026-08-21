# MFPT + GlobalAverageLinear baseline candidate

This directory contains a transparent real-data protocol candidate. The model averages
each one-channel window over time and applies one linear classifier.

The candidate is currently:

```text
execution_status: sanity_ok
protocol_status: smoke_only
```

It was previously promoted as `baseline_valid`, but recent changes altered metric
lifecycle, checkpoint selection, and repeated-run aggregation. The exact experiment must
be rerun on the current source before that claim can be restored.

## Frozen protocol

```text
provider: mathworks/RollingElementBearingFaultDiagnosis-Data
provider revision: d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d
train population: 14 provider train_data MAT files
held-out test population: 6 provider test_data MAT files
train/val split: grouped by source File, stratified by Label
labels: 0 normal, 1 inner-race fault, 2 outer-race fault
seeds: 17, 18, 19
model: Baseline/GlobalAverageLinear
loss: CE
metrics: accuracy, macro-F1
```

Provider test files must never participate in fitting, validation, early stopping, or
checkpoint selection.

## Prepare data

```bash
python -m scripts.prepare_mfpt_baseline --output data/mfpt
```

The command obtains the pinned provider revision, requires the exact 20 public MAT files,
validates each payload, and creates:

```text
data/mfpt/
├── metadata_mfpt.csv
└── raw/RM_007_MFPT/
    ├── train_data/*.mat
    └── test_data/*.mat
```

The output directory must not already exist. The command never overwrites user data. The
dataset is provided under CC BY-NC-SA 4.0; review the provider license before use.

## Preflight and run

```bash
phmfactory preflight \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml

phmfactory \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
```

Machine-specific paths remain explicit:

```bash
phmfactory \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml \
  --override data.data_dir=/absolute/path/to/mfpt \
  --override environment.output_dir=/absolute/path/to/results \
  --override data.split.manifest_path=/absolute/path/to/results/split_manifest.json
```

## Current-source promotion gate

The candidate may return to `baseline_valid` only when the unchanged current-source run
satisfies:

```text
20 strict reader successes
14 provider-train files partitioned into disjoint train/val file groups
6 provider-test files used only for test
three completed seeds: 17, 18, 19
one best checkpoint restored before every test
accuracy and macro-F1 present for every seed
non-empty finite test metrics
run_summary count=3 with finite mean and sample_std
independent workflow-only accuracy/F1 recomputation agrees with framework output
```

Do not change the data population, split, model, objective, metrics, epochs, or seeds to
recover a preferred result. A failed requalification is evidence and must leave the
candidate unpromoted.

## Claim boundary

Historical results remain useful protocol evidence, but they are not current-source
promotion evidence. This candidate does not claim state-of-the-art accuracy, a strong
signal representation, or universal MFPT protocol superiority. `GlobalAverageLinear`
deliberately ignores sampling-rate and fault-frequency metadata; those omissions are
visible baseline limitations.

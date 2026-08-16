# MFPT + GlobalAverageLinear baseline v1

This directory contains PHMFactory's first scientifically closed real-data baseline
candidate. It is intentionally transparent: the model averages each one-channel window
over time and applies one linear classifier.

## Protocol

```text
provider: mathworks/RollingElementBearingFaultDiagnosis-Data
provider revision: d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d
train population: 14 provider train_data MAT files
held-out test population: 6 provider test_data MAT files
train/val split: grouped by source File, stratified by Label
labels: 0 normal, 1 inner-race fault, 2 outer-race fault
seeds: 17, 18, 19
model: Baseline/GlobalAverageLinear
```

The provider test files never participate in fitting, validation, early stopping, or
checkpoint selection.

## Prepare data

```bash
python -m scripts.prepare_mfpt_baseline --output data/mfpt
```

The command downloads the pinned provider revision, copies only the exact 20 public
bearing-test-rig MAT files, validates each MAT payload, and creates:

```text
data/mfpt/
├── metadata_mfpt.csv
└── raw/RM_007_MFPT/
    ├── train_data/*.mat
    └── test_data/*.mat
```

The output directory must not already exist. The command never overwrites user data.
The dataset is provided under CC BY-NC-SA 4.0; review the provider license before use,
especially for commercial work.

## Preflight and run

```bash
phmfactory preflight \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml

phmfactory \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
```

Machine-specific locations remain explicit:

```bash
phmfactory \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml \
  --override data.data_dir=/absolute/path/to/mfpt \
  --override environment.output_dir=/absolute/path/to/results \
  --override data.split.manifest_path=/absolute/path/to/results/split_manifest.json
```

## Success criteria

A baseline-valid execution requires all of the following:

```text
20 strict reader successes
14 provider-train files partitioned into non-overlapping train/val file groups
6 provider-test files used only for test
three completed seeds: 17, 18, 19
best checkpoint restored before every test
non-empty finite test metrics for every seed
run_summary.json with count=3 and finite mean/sample_std
```

## Claim boundary

This baseline establishes a real-data execution and estimator contract. It does not claim
state-of-the-art accuracy, a strong signal representation, or universal MFPT protocol
superiority. `GlobalAverageLinear` deliberately ignores the available sampling-rate and
fault-frequency metadata; those omissions are baseline limitations, not hidden behavior.

# 06. Config and Matrix Guide

## Config families currently present

### Demo configs

```text
configs/demo/10_generative/dummy_generative_cfm.yaml
configs/demo/10_generative/dummy_generative_rectified_flow.yaml
configs/demo/10_generative/dummy_generative_ddpm.yaml
configs/demo/10_generative/dummy_generative_score_sde.yaml
configs/demo/10_generative/dummy_generative_meanflow.yaml
configs/demo/10_generative/dummy_generative_drifting_flow.yaml
configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml
configs/demo/10_generative/dummy_generative_ot_nfm.yaml
```

Demo configs are smoke only.  They should stay CPU-friendly and exploratory.

### Paper configs

```text
configs/paper/phm_generative/
  six_dataset_benchmark_matrix.yaml
  cfm_train_grid_seed0.yaml
  cfm_train_grid_seed1.yaml
  cfm_sample_grid_seed0.yaml
  cfm_eval_train_reference_seed0.yaml
  rectified_flow_train_grid_seed0.yaml
  ddpm_train_distribution_seed0.yaml
  benchmark_effect_matrix.yaml
```

Paper configs are not smoke tests.  They require real datasets and GPU evidence.

## Six-dataset matrix contract

The matrix should define:

```yaml
benchmark:
  id:
  output_dir:
  baseline_method:
  min_datasets: 6
  seeds: [0, 1]
  resource:
    gpu_ids: [6, 7]
    max_parallel_runs: 2
    require_cuda: true

datasets:
  - dataset:
    dataset_id:
    name:
    overrides:
      task.target_system_id:
      task.source_domain_id:
      task.target_domain_id:
    protocol:
      utility:
      notes:

methods:
  - method:
    label:
    train_config:
    condition_sampling_policy:
```

## Current paper dataset set

```text
RM_001_CWRU
RM_002_XJTU
RM_003_FEMTO
RM_008_UNSW
RM_024_JUST
RM_027_PU
```

## Required matrix validation

Add a command:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --allow-missing-data
```

Then add strict mode for real execution:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run
```

Strict mode must fail if metadata paths are missing.

## Problem to fix: placeholder paths

The dry-run planner emits `<experiment_name>` placeholders for checkpoint and
sample paths.  v0.3 should not execute these blindly.  Add one of:

```text
A. artifact resolver script that resolves actual run dirs after train/sample.
B. stage ledger written by every stage.
C. explicit --checkpoint_path / --generated_path handoff file.
```

Preferred: stage ledger.

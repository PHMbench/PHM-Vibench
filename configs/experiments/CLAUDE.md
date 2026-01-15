# Experiment Configurations - CLAUDE.md

This module provides architecture guidance for local experiment configurations in PHM-Vibench. For usage, see [@README.md].

## Overview

The `configs/experiments/` directory contains local experiment configurations and variants.

## Purpose

This directory is for:
- **Personal research experiments**
- **Hyperparameter tuning**
- **One-off tests**
- **Local configuration overrides**

## Best Practices

### 1. Naming Convention
```
{task}_{dataset}_{description}.yaml

Examples:
- dg_cwru_adamw_1e3.yaml
- fs_thu_10way_5shot_contrastive.yaml
- ablation_cwru_no_norm.yaml
```

### 2. Organization
Create subdirectories for organization:
```
configs/experiments/
├── dg_experiments/
├── fs_experiments/
├── ablation_studies/
└── hyperparam_tuning/
```

### 3. Use Base Configs
Avoid duplication by using base configs:
```yaml
base_configs:
  - configs/base/environment.yaml
  - configs/base/data.yaml

# Your experiment-specific overrides
model:
  d_model: 256  # Your experiment value
```

### 4. Document Your Experiments
Add comments explaining:
- Purpose of experiment
- What you're testing
- Expected outcomes

## Tracking policy

This directory is tracked by git in this repository. Keep configs here portable (no machine-specific absolute paths).

For machine-local paths (e.g. `data.data_dir`), prefer:
- `configs/local/local.yaml` (git-ignored), or
- CLI overrides: `python main.py --config <yaml> --override data.data_dir=/path/to/...`

## Related Documentation

- [@README.md] - Experiment Configuration Guide
- [@../README.md] - Configuration System
- [@../demo/CLAUDE.md] - Demo Configurations

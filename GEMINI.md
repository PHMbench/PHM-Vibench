# PHM-Vibench Project Context

## Project Overview
**PHM-Vibench** is a comprehensive, modular benchmark platform designed for industrial equipment vibration signal fault diagnosis and predictive maintenance (PHM). It addresses challenges in fragmented experimental environments and reproducibility by providing a standardized framework.

*   **Type:** Python / PyTorch Deep Learning Project
*   **Goal:** Enable fair comparison and rapid prototyping of PHM algorithms.
*   **Core Philosophy:** Configuration-first, factory-based modular design.

## Key Features

### 1. Extensive Industrial Dataset Support
The platform integrates loaders for over 20 standard industrial datasets, covering diverse equipment like bearings, gears, and pumps:
*   **Classics:** CWRU, XJTU, IMS, FEMTO (bearing/gear faults).
*   **University/Lab:** THU/THU24, Ottawa19/23, JNU, SEU, HUST23/24, HIT/HIT23, KAIST, PU.
*   **Others:** MFPT, UNSW, DIRG, JUST, Pump data.
*   **Mechanism:** Implemented via `src/data_factory/reader/` inheriting from `BaseReader`.

### 2. Diverse Algorithm Implementation
Supports a wide range of deep learning architectures tailored for 1D signal processing:
*   **Backbones:** CNN (ResNet, VGG), RNN (LSTM, GRU), MLP, Transformers (ViT variants).
*   **Advanced Models:** ISFM (Industrial Signal Foundation Models), ISFM_Prompt (Prompt tuning).
*   **Custom:** Extensible via `src/model_factory` and `model_registry.csv`.

### 3. Modular "5-Block" Configuration System
Experiments are defined by YAML files composed of 5 independent but interacting blocks:
1.  **Environment:** Global settings (seeds, project paths, output directories, logging).
2.  **Data:** Dataset selection, preprocessing (windowing, normalization, truncation), batch size, workers.
3.  **Model:** Network architecture hyperparameters (layers, heads, dimensions, dropout).
4.  **Task:** Learning objective (Classification, RUL, Anomaly Detection), loss functions, optimizers, schedulers, metrics. Supports complex scenarios like Domain Generalization (DG) and Cross-Domain DG (CDDG).
5.  **Trainer:** Training loop controls (epochs, GPUs, early stopping, checkpointing, WandB integration).

### 4. Reproducible Workflows
*   **Unified Pipelines:** Pre-defined pipelines (`Pipeline_01_default`, `Pipeline_02_pretrain_fewshot`, etc.) ensure consistent execution logic.
*   **Result Management:** Structured output in `save/{metadata}/{model}/{task_trainer_timestamp}/` containing checkpoints, logs, and config backups.

## Directory Structure
*   `src/`: Core logic.
    *   `data_factory/`: Dataset readers (`reader/`), samplers, and data processing.
    *   `model_factory/`: Model architectures and registry.
    *   `task_factory/`: Task definitions (loss, metrics).
    *   `trainer_factory/`: Training loops.
*   `configs/`: Experiment configurations.
    *   `demo/`: **Start here.** Maintained templates for smoke tests and demos.
    *   `base/`: Reusable config fragments (data, model, etc.).
    *   `experiments/`: User-specific local experiments (git-ignored).
    *   `config_registry.csv`: Index of maintained configurations.
*   `data/`: Storage for raw dataset files (git-ignored).
*   `save/`: Experiment artifacts (git-ignored).
*   `test/`: Pytest suite for regression testing.
*   `scripts/`: Utilities for config validation and inspection.

## Workflows & Commands

### 1. Environment Setup
```bash
conda create -n PHM python=3.10
conda activate PHM
pip install -r requirements.txt
```

### 2. Running Experiments
The entry point is `main.py`. It requires a config file.
```bash
# Minimal Smoke Test (No data download required)
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# Standard Run (e.g., Cross-Domain Generalization on CWRU)
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml --override trainer.num_epochs=50
```

### 3. Configuration Management
Use the provided scripts to debug and validate configurations:
```bash
# Inspect how a config is resolved (merged layers + overrides)
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml

# Validate all registry configs against the schema
python -m scripts.validate_configs
```

### 4. Development & Testing
```bash
# Run the test suite
python -m pytest test/
```

## Development Guidelines
*   **Adding Datasets:** Create a new reader in `src/data_factory/reader/RM_XXX.py` and register it.
*   **Adding Models:** Add the model class in `src/model_factory/` and update `model_registry.csv`.
*   **Configs:** Do not modify `configs/base/` directly if possible. Create new compositions in `configs/experiments/` or use `--override` for quick tests.
*   **Documentation:** Update `configs/config_registry.csv` and run `python -m scripts.gen_config_atlas` when adding new stable configurations.

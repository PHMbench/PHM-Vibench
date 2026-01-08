# PHM-Vibench Project Context (GEMINI)

This file provides project context for AI assistants working on PHM-Vibench. For comprehensive documentation, see [@README.md]; for development commands, see [@AGENTS.md].

## Quick Overview

**PHM-Vibench** is a modular benchmark platform for industrial vibration signal fault diagnosis and predictive maintenance.

*   **Type:** Python / PyTorch Deep Learning Project
*   **Goal:** Enable fair comparison and rapid prototyping of PHM algorithms
*   **Core Philosophy:** Configuration-first, factory-based modular design

For detailed project overview and features, see [@README.md - Project Overview].

## Key Context Points

### Extensive Dataset Support
The platform integrates loaders for 20+ industrial datasets via `src/data_factory/reader/`:
*   **Classics:** CWRU, XJTU, IMS, FEMTO
*   **University/Lab:** THU/THU24, Ottawa19/23, JNU, SEU, HUST23/24, HIT/HIT23, KAIST
*   **Others:** MFPT, UNSW, DIRG, JUST, Pump data

### Algorithm Support
*   **Backbones:** CNN (ResNet, VGG), RNN (LSTM, GRU), MLP, Transformers
*   **Advanced Models:** ISFM, ISFM_Prompt
*   **Extensible via:** `src/model_factory` and `model_registry.csv`

### 5-Block Configuration System
Experiments defined by YAML: `environment | data | model | task | trainer`
*   **Environment:** Global settings (seeds, paths, output dirs, logging)
*   **Data:** Dataset selection, preprocessing, batch size
*   **Model:** Network architecture hyperparameters
*   **Task:** Learning objective, loss functions, optimizers, metrics
*   **Trainer:** Training loop controls (epochs, GPUs, checkpointing)

See [@configs/README.md] for configuration system details.

## Directory Structure

```
src/
├── data_factory/    # Dataset readers, samplers, processing
├── model_factory/   # Model architectures and registry
├── task_factory/    # Task definitions (loss, metrics)
└── trainer_factory/ # Training loops

configs/
├── demo/            # Start here - maintained templates
├── base/            # Reusable config fragments
├── experiments/     # User-specific local experiments
└── config_registry.csv  # Index of maintained configurations
```

For complete directory structure, see [@README.md - Directory Structure].

## Running Experiments

The entry point is `main.py`. For detailed workflows, see [@README.md - Quick Start] and [@AGENTS.md - Quick Commands].

```bash
# Smoke test (no data download required)
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# Standard run
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml --override trainer.num_epochs=50
```

## Configuration Management

```bash
# Inspect config resolution
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml

# Validate registry configs
python -m scripts.validate_configs
```

## Development Guidelines

For contribution guidelines:
*   **Adding Datasets:** Create reader in `src/data_factory/reader/RM_XXX.py` and register
*   **Adding Models:** Add class in `src/model_factory/` and update `model_registry.csv`
*   **Configs:** Use `configs/experiments/` for local variants or `--override` for quick tests
*   **Documentation:** Update `configs/config_registry.csv` and run `python -m scripts.gen_config_atlas`

See [@README.md - Contributing] and [docs/developer_guide.md](docs/developer_guide.md) for more details.

## Testing

```bash
# Run the test suite
python -m pytest test/
```

## Cross-Reference

| Topic | Reference |
|-------|-----------|
| Project overview | [@README.md] |
| Quick commands | [@AGENTS.md] |
| Configuration system | [@configs/README.md] |
| Change strategy | [@CLAUDE.md] |
| Developer guide | [docs/developer_guide.md](docs/developer_guide.md) |

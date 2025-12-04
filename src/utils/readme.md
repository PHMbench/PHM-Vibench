# Utils API Reference

> **Note**: This is the English API reference. For comprehensive documentation, please see [README.md](README.md).

## Overview

This directory provides helper utilities for configuration management, experiment logging, training orchestration, and model evaluation in the PHM-Vibench framework.

## Quick Links

- **[Comprehensive Documentation (中文)](README.md)** - Main documentation with decision trees and guides
- **[Architecture Guide](../../../CLAUDE.md)** - Project-wide architecture documentation

## Core Utilities

### Configuration Management (`config_utils.py`)

#### Functions

- **`load_config(config_path)`**
  Reads a YAML configuration file with automatic encoding fallback (GB18030) and raises an error when the file is missing.

  ```python
  config = load_config('config.yaml')
  ```

- **`makedir(path)`**
  Creates the directory at `path` if it does not already exist.

  ```python
  makedir('/path/to/experiment/results')
  ```

- **`path_name(configs, iteration=0)`**
  Constructs a timestamped experiment name from dataset, model, task, and trainer details in `configs`. Returns the created result directory and experiment name.

  ```python
  result_dir, exp_name = path_name(configs)
  ```

- **`transfer_namespace(raw_arg_dict)`**
  Converts a dictionary to a `types.SimpleNamespace` for attribute-style access.

  ```python
  namespace = transfer_namespace(config_dict)
  print(namespace.model.name)  # Access as attributes
  ```

### Model and Training Utilities (`utils.py`)

#### Functions

- **`load_best_model_checkpoint(model, trainer)`**
  Retrieves the `ModelCheckpoint` callback from a PyTorch Lightning `Trainer` and loads weights from the best checkpoint into `model`.

  ```python
  from src.utils.utils import load_best_model_checkpoint
  load_best_model_checkpoint(model, trainer)
  ```

- **`init_lab(args_environment, cli_args, experiment_name)`**
  Initializes optional `wandb` and `swanlab` loggers based on configuration flags and command-line notes, handling missing libraries gracefully.

  ```python
  init_lab(config.environment, args, experiment_name)
  ```

- **`close_lab()`**
  Finalizes active `wandb` and `swanlab` sessions if they were initialized.

  ```python
  close_lab()
  ```

### Registry System (`registry.py`)

#### Class: `Registry`

A generic registry pattern implementation for dynamic component registration and retrieval.

#### Methods

- **`register_module()`** (decorator)
  Registers a class or function in the registry.

  ```python
  from src.utils.registry import Registry

  MODEL_REGISTRY = Registry('model')

  @MODEL_REGISTRY.register_module()
  class MyModel:
      pass
  ```

- **`build(config)`**
  Builds an instance from configuration dictionary.

  ```python
  model = MODEL_REGISTRY.build(config.model)
  ```

- **`get(name)`**
  Retrieves a registered class by name.

  ```python
  model_class = MODEL_REGISTRY.get('MyModel')
  ```

### Training Orchestration (`training/two_stage_orchestrator.py`)

#### Class: `MultiStageOrchestrator`

Orchestrates multi-stage training workflows with checkpoint management and configuration inheritance.

#### Methods

- **`run_stages()`**
  Executes all configured training stages in sequence.

  ```python
  from src.utils.training.two_stage_orchestrator import MultiStageOrchestrator

  orchestrator = MultiStageOrchestrator(config)
  orchestrator.run_stages()
  ```

### HSE Utilities (`hse/`)

#### HSE Prompt Validator (`hse/prompt_validator.py`)

##### Class: `HSPPromptValidator`

Validates HSE (Harmonic Spectral Enhancement) prompt configurations.

##### Methods

- **`validate_config(config)`**
  Validates HSE prompt configuration parameters.

  ```python
  from src.utils.hse import HSPPromptValidator

  validator = HSPPromptValidator()
  is_valid = validator.validate_config(config)
  ```

#### HSE Integration Utils (`hse/integration_utils.py`)

##### Class: `HSEIntegrationUtils`

Provides integration utilities for HSE with training pipelines.

##### Methods

- **`create_pretraining_config(base_configs, backbone, target_systems, pretraining_config)`**
  Creates pretraining configuration for HSE-guided models.

- **`create_finetuning_config(base_configs, checkpoint_path, backbone, target_system, finetuning_config)`**
  Creates finetuning configuration with checkpoint loading.

### Evaluation (`evaluation/ZeroShotEvaluator.py`)

#### Class: `ZeroShotEvaluator`

Performs zero-shot evaluation using linear probes on frozen pretrained models.

#### Methods

- **`evaluate(model, dataloaders)`**
  Evaluates model performance on specified datasets.

### Validation (`validation/OneEpochValidator.py`)

#### Class: `OneEpochValidator`

Provides rapid 1-epoch training validation for early issue detection.

#### Methods

- **`validate(model, dataloader)`**
  Runs validation with memory monitoring and performance benchmarking.

## Conventions and Usage Patterns

### Configuration Access
- Configuration dictionaries are converted to namespaces for attribute-style access
- Most utilities expect arguments with attributes: `config.model.name`
- Dictionary access is also supported: `config['model']['name']`

### Experiment Management
- Result paths follow standardized structure with timestamped names
- `path_name()` uses `makedir()` to ensure directories exist
- Logging helpers enable/disable external services based on configuration flags

### Registration Pattern
- Use registry for dynamic component discovery
- Decorator-based registration is preferred
- Configuration-driven component building

## Error Handling

- Configuration loading handles encoding fallbacks automatically
- Missing libraries are handled gracefully in logging utilities
- Registry operations provide clear error messages for missing components

## Dependencies

### Required
- Python 3.8+
- PyTorch
- PyYAML
- PyTorch Lightning

### Optional
- `wandb` - Experiment tracking
- `swanlab` - Alternative experiment tracking

## Migration Notes

**Deprecated Modules** (see [README.md](README.md) for migration guide):
- `pipeline_config.py` → Use `utils.py` and `pipeline_config/base_utils.py`
- `config/hse_prompt_validator.py` → Use `hse/prompt_validator.py`
- `pipeline_config/hse_prompt_integration.py` → Use `hse/integration_utils.py`

## Support

For comprehensive documentation, decision trees, and troubleshooting guides, please refer to the [main documentation](README.md).

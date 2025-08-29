# CLAUDE.md (change before user confirm)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PHM-Vibench is a comprehensive benchmark platform for industrial equipment vibration signal analysis, focusing on fault diagnosis and predictive maintenance. It features a modular factory design pattern with extensive support for multiple datasets, models, and tasks.

## Key Architecture Components

### Factory Design Pattern
The codebase uses factory patterns for maximum modularity:
- **data_factory/**: Dataset loading and processing with 30+ industrial datasets (CWRU, XJTU, FEMTO, etc.)
- **model_factory/**: Neural network architectures including Transformers, CNNs, RNNs, and specialized foundation models
- **task_factory/**: Task definitions (classification, prediction, few-shot learning, domain generalization)
- **trainer_factory/**: Training orchestration with PyTorch Lightning

### Pipeline System
The framework supports multiple experimental pipelines:
- `Pipeline_01_default`: Standard training pipeline
- `Pipeline_02_pretrain_fewshot`: Two-stage pretraining + few-shot learning
- `Pipeline_03_multitask_pretrain_finetune`: Multi-task foundation model training
- `Pipeline_ID`: ID-based data processing pipeline

### Configuration-Driven Experiments
All experiments use the unified configuration system v5.0 with flexible loading options:
- **YAML Templates**: Built-in presets like `quickstart`, `isfm`, `gfs` for common patterns
- **File Overrides**: Use any YAML file to override base configurations
- **Dictionary Overrides**: Programmatic configuration via Python dictionaries
- **ConfigWrapper Chaining**: Multi-stage pipeline configuration inheritance

Configuration sections include:
- `data`: Dataset configuration and preprocessing parameters
- `model`: Model architecture and hyperparameters
- `task`: Task type, loss functions, and training settings
- `trainer`: Training orchestration and hardware settings

## Module-Specific Documentation

For detailed guidance on specific components, see:
- [Configuration System](./src/configs/CLAUDE.md) - Unified configuration management, YAML templates, and multi-stage pipelines
- [Data Factory](./src/data_factory/CLAUDE.md) - Dataset integration, processing, and reader implementation
- [Model Factory](./src/model_factory/CLAUDE.md) - Model architectures, ISFM foundation models, and implementations  
- [Task Factory](./src/task_factory/CLAUDE.md) - Task definitions, training logic, and loss functions
- [Trainer Factory](./src/trainer_factory/CLAUDE.md) - Training orchestration and PyTorch Lightning integration
- [Utils](./src/utils/CLAUDE.md) - Utility functions, configuration helpers, and registry patterns

## Common Development Commands

### Running Experiments
```bash
# Basic single dataset experiment
python main.py --config configs/demo/Single_DG/CWRU.yaml

# Cross-dataset domain generalization
python main.py --config configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml

# Pretraining + Few-shot pipeline
python main.py --pipeline Pipeline_02_pretrain_fewshot --config_path configs/demo/Pretraining/Pretraining_demo.yaml --fs_config_path configs/demo/GFS/GFS_demo.yaml

# All datasets experiment
python main.py --config configs/demo/Multiple_DG/all.yaml
```

### Testing
```bash
# Run comprehensive test suite
python run_tests.py

# Run specific test categories
pytest test/ -m "not slow"  # Skip slow tests
pytest test/ -m "unit"      # Unit tests only
pytest test/ -m "gpu" --tb=short  # GPU tests
```

### Streamlit GUI
```bash
# Launch interactive experiment interface
streamlit run streamlit_app.py
```

### Test Configuration
- Tests are configured in `pytest.ini` with comprehensive coverage settings
- Test requirements in `requirements-test.txt` include pytest, coverage, and ML testing tools
- Minimum 80% code coverage required

## Dataset Integration

### Data Structure
- Raw datasets in `data/raw/<DATASET_NAME>/`
- Metadata files: `metadata_*.xlsx` describing dataset structure
- H5 processed files for efficient loading
- Readers in `src/data_factory/reader/RM_*.py` for each dataset

### Adding New Datasets
1. Place raw data in `data/raw/<DATASET_NAME>/`
2. Create metadata file describing structure
3. Implement reader class inheriting from BaseReader
4. Register in `data_factory/__init__.py`

## Model Architecture

### Foundation Models (ISFM - Industrial Signal Foundation Model)
- **M_01_ISFM**: Basic transformer-based foundation model
- **M_02_ISFM**: Advanced multi-modal foundation model
- **M_03_ISFM**: Specialized temporal dynamics model

### Backbone Networks
- **B_08_PatchTST**: Patch-based time series transformer
- **B_04_Dlinear**: Direct linear forecasting model  
- **B_06_TimesNet**: Time series analysis network
- **B_09_FNO**: Fourier Neural Operator for signal processing

### Task Heads
- **H_01_Linear_cla**: Linear classification head
- **H_09_multiple_task**: Multi-task learning head
- **H_03_Linear_pred**: Linear prediction head

## Task Types and Use Cases

### Supported Tasks
- **Classification**: Fault diagnosis and equipment state classification
- **CDDG**: Cross-Dataset Domain Generalization for robustness
- **FS/GFS**: Few-Shot and Generalized Few-Shot Learning
- **Pretrain**: Self-supervised pretraining for foundation models

### Domain Generalization
- Single domain: Training and testing on same dataset
- Cross-dataset: Training on one dataset, testing on another
- Multiple domain: Using multiple source domains for robustness

## Environment Setup

### Dependencies
- Python 3.8+, PyTorch 2.6.0, PyTorch Lightning
- Scientific computing: numpy, pandas, scipy, scikit-learn
- Visualization: matplotlib, seaborn, plotly
- ML utilities: wandb, tensorboard, transformers, timm

### Key Environment Variables
Set `data_dir` in config files to point to your data directory containing the metadata Excel files and H5 dataset files.

## Results and Output

### Directory Structure
Results are saved in hierarchical structure under `save/`:
```
save/{metadata_file}/{model_name}/{task_type}_{trainer_name}_{timestamp}/
├── checkpoints/     # Model weights
├── metrics.json     # Performance metrics  
├── log.txt         # Training logs
├── figures/        # Visualization plots
└── config.yaml     # Experiment configuration backup
```

### Logging and Monitoring
- WandB integration for experiment tracking
- Comprehensive metrics logging
- Automatic figure generation for analysis

## Important Notes

- The codebase uses factory patterns extensively - always register new components in the appropriate factory
- Configuration files drive all experiments - avoid hardcoding parameters
- Results are automatically organized by metadata file, model, and timestamp
- The framework supports both traditional ML approaches and modern foundation models
- Multi-task and cross-dataset capabilities are core features for industrial applications
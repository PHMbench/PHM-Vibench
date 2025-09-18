# Project Structure - PHM-Vibench

## Directory Organization

### Root Level Structure
```
PHM-Vibench/
├── main.py                    # Primary entry point
├── main_dummy.py              # Testing entry point
├── streamlit_app.py           # Web interface
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── CLAUDE.md                  # Project-specific Claude instructions
├── .claude/                   # Claude Code configuration
│   ├── steering/              # Steering documents
│   ├── agents/                # Specialized agent definitions
│   ├── commands/              # Custom commands
│   └── templates/             # Code templates
├── configs/                   # Experiment configurations
├── src/                       # Source code modules
├── data/                      # Dataset storage (user-configured)
├── save/                      # Experiment results
├── test/                      # Testing suite
├── scripts/                   # Utility scripts
└── pic/                       # Documentation assets
```

## Source Code Architecture (src/)

### Factory Pattern Structure
```
src/
├── configs/                   # Configuration management
│   ├── __init__.py           # Unified config loading
│   ├── config_utils.py       # Configuration utilities
│   ├── presets/              # Built-in configuration templates
│   └── CLAUDE.md             # Configuration system documentation
├── data_factory/             # Dataset management
│   ├── __init__.py           # Data factory registration
│   ├── data_factory.py       # Main data factory class
│   ├── base_data.py          # Base dataset interface
│   ├── H5DataDict.py         # H5 data dictionary
│   ├── ID_data_factory.py    # ID-based data processing
│   ├── reader/               # Dataset readers
│   │   ├── RM_*.py           # Reader implementations
│   │   └── BaseReader.py     # Base reader class
│   └── CLAUDE.md             # Data factory documentation
├── model_factory/            # Model architecture management
│   ├── __init__.py           # Model factory registration
│   ├── model_factory.py      # Main model factory class
│   ├── base_model.py         # Base model interface
│   ├── models/               # Model implementations
│   │   ├── backbone/         # Backbone networks (B_*)
│   │   ├── embedding/        # Embedding layers (E_*)
│   │   ├── task_head/        # Task-specific heads (H_*)
│   │   └── full_models/      # Complete model architectures (M_*)
│   └── CLAUDE.md             # Model factory documentation
├── task_factory/             # Training task management
│   ├── __init__.py           # Task factory registration
│   ├── task_factory.py       # Main task factory class
│   ├── base_task.py          # Base task interface
│   ├── tasks/                # Task implementations
│   │   ├── classification/   # Classification tasks
│   │   ├── few_shot/         # Few-shot learning tasks
│   │   ├── domain_gen/       # Domain generalization tasks
│   │   └── pretraining/      # Pretraining tasks
│   └── CLAUDE.md             # Task factory documentation
├── trainer_factory/          # Training orchestration
│   ├── __init__.py           # Trainer factory registration
│   ├── trainer_factory.py    # Main trainer factory class
│   ├── base_trainer.py       # Base trainer interface
│   ├── trainers/             # Trainer implementations
│   └── CLAUDE.md             # Trainer factory documentation
└── utils/                    # Shared utilities
    ├── __init__.py           # Utility registration
    ├── reproducibility.py    # Seed and determinism management
    ├── logger.py             # Logging utilities
    ├── visualization.py      # Plotting and analysis
    └── CLAUDE.md             # Utils documentation
```

## Configuration System

### Configuration Hierarchy
```
configs/
├── demo/                     # Example configurations
│   ├── Single_DG/            # Single dataset experiments
│   │   ├── CWRU.yaml
│   │   ├── MFPT.yaml
│   │   └── ...
│   ├── Multiple_DG/          # Cross-dataset experiments
│   │   ├── CWRU_THU_using_ISFM.yaml
│   │   ├── all.yaml
│   │   └── ...
│   ├── FewShot/              # Few-shot learning
│   ├── Pretraining/          # Pretraining experiments
│   └── dummy_test.yaml       # Testing configuration
└── experiments/              # Production configurations
```

### YAML Structure Pattern
```yaml
environment:                  # Environment settings
  name: 'experiment_name'
  seed: 42
  project: 'PHM-Vibench'

data:                        # Dataset configuration
  data_dir: '/path/to/data'
  metadata_file: 'metadata.xlsx'
  batch_size: 64
  # ... data processing parameters

model:                       # Model architecture
  name: 'M_01_ISFM'
  backbone: 'B_01_basic_transformer'
  task_head: 'H_01_Linear_cla'
  # ... model hyperparameters

task:                        # Training task
  name: 'classification'
  type: 'CDDG'
  optimizer: 'adam'
  lr: 0.001
  # ... training parameters

trainer:                     # Training orchestration
  name: 'Default_trainer'
  args:
    num_epochs: 100
    gpus: 1
    # ... trainer settings
```

## Naming Conventions

### Component Naming
- **Models**: `M_XX_ModelName` (e.g., `M_01_ISFM`)
- **Backbones**: `B_XX_BackboneName` (e.g., `B_08_PatchTST`)
- **Task Heads**: `H_XX_HeadName` (e.g., `H_01_Linear_cla`)
- **Embeddings**: `E_XX_EmbeddingName` (e.g., `E_01_HSE`)
- **Readers**: `RM_DatasetName` (e.g., `RM_CWRU`)

### File Naming
- **Configuration files**: `dataset_name.yaml`, `experiment_description.yaml`
- **Pipeline modules**: `Pipeline_XX_description`
- **Documentation**: `CLAUDE.md` for module-specific docs
- **Test files**: `test_component.py`

### Directory Naming
- **Factory modules**: `component_factory/`
- **Implementation subdirs**: `components/`, `models/`, `tasks/`, `trainers/`
- **Result organization**: `{metadata_file}/{model_name}/{experiment_timestamp}/`

## Result Organization

### Hierarchical Structure
```
save/
└── {metadata_file}/          # Dataset metadata grouping
    └── {model_name}/          # Model architecture grouping
        └── {task}_{trainer}_{timestamp}/  # Experiment instance
            ├── checkpoints/   # Model weights (.pth, .ckpt)
            ├── metrics.json   # Performance metrics
            ├── log.txt        # Training logs
            ├── figures/       # Visualizations (.png, .pdf)
            │   ├── confusion_matrix.png
            │   ├── learning_curve.png
            │   └── loss_curve.png
            └── config.yaml    # Experiment configuration backup
```

## Module Registration Patterns

### Factory Registration
Each factory uses a centralized registration system:
```python
# In __init__.py files
from .component_factory import ComponentFactory

# Register all components
COMPONENTS = {
    'component_name': ComponentClass,
    # ...
}

def get_component(name, **kwargs):
    return COMPONENTS[name](**kwargs)
```

### Pipeline Registration
```python
# Pipeline modules follow naming: src.Pipeline_XX_name
# Imported dynamically in main.py:
pipeline = importlib.import_module(f'src.{args.pipeline}')
```

## Extension Points

### Adding New Components
1. **Datasets**: Implement `BaseReader` in `data_factory/reader/RM_NewDataset.py`
2. **Models**: Add to appropriate subdirectory in `model_factory/models/`
3. **Tasks**: Implement `BaseTask` in `task_factory/tasks/`
4. **Trainers**: Implement `BaseTrainer` in `trainer_factory/trainers/`

### Configuration Extension
1. Add new YAML files in `configs/demo/` or `configs/experiments/`
2. Use preset templates from `src/configs/presets/`
3. Follow hierarchical parameter structure

### Pipeline Extension
1. Create new `src/Pipeline_XX_description.py` module
2. Implement `pipeline(args)` function
3. Register in argument parser options

## Testing Structure
```
test/
├── test_data_factory.py      # Data loading tests
├── test_model_factory.py     # Model architecture tests
├── test_task_factory.py      # Training task tests
├── test_trainer_factory.py   # Training orchestration tests
├── test_configs.py           # Configuration system tests
├── test_integration.py       # End-to-end tests
└── fixtures/                 # Test data and configurations
```

## Documentation Standards

### Module Documentation
- Each major module has `CLAUDE.md` with component-specific guidance
- API documentation in docstrings following Google style
- Configuration parameter tables in README.md

### Code Documentation
- **Docstrings**: All public classes and methods
- **Type hints**: All function signatures
- **Inline comments**: Complex logic explanation only
- **Configuration comments**: Parameter explanations in YAML
# Technology Stack - PHM-Vibench

## Core Technologies

### Programming Language & Runtime
- **Python 3.8+**: Primary development language
- **conda/pip**: Package management and virtual environments

### Deep Learning Framework
- **PyTorch 2.6.0**: Core deep learning framework
- **PyTorch Lightning**: Training orchestration and experiment management
- **torchvision 0.21.0**: Computer vision utilities
- **torchaudio 2.6.0**: Audio signal processing

### Scientific Computing Stack
- **numpy >= 1.20.0**: Numerical computing foundation
- **pandas >= 1.3.0**: Data manipulation and analysis
- **scipy**: Scientific computing algorithms
- **scikit-learn >= 1.0.0**: Classical machine learning
- **h5py**: HDF5 file format for efficient data storage

### Visualization & Monitoring
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations
- **tensorboard**: Training monitoring and logging
- **wandb**: Experiment tracking and collaboration
- **swanlab**: Alternative experiment tracking

### ML Ecosystem Libraries
- **transformers**: Hugging Face transformer models
- **timm**: PyTorch image models library
- **einops**: Tensor operations with readable notation
- **reformer_pytorch**: Memory-efficient transformer implementation
- **umap-learn**: Dimensionality reduction and visualization

### Data Processing
- **xlrd > 2.0.0**: Excel file reading
- **openpyxl**: Excel file writing and manipulation
- **modelscope**: Model and dataset management
- **urllib3**: HTTP client library

### Web Interface
- **streamlit**: Interactive web applications for experiments

### Specialized Libraries
- **scienceplots**: Publication-quality matplotlib styles

## Architecture Patterns

### Factory Design Pattern
- **data_factory**: Manages dataset loading and preprocessing
- **model_factory**: Handles model architecture registration and creation
- **task_factory**: Defines training tasks and loss functions
- **trainer_factory**: Orchestrates training processes

### Configuration Management
- **YAML-based**: Human-readable configuration files
- **ConfigWrapper**: Unified configuration handling with v5.0 system
- **Hierarchical merging**: Support for nested configuration overrides
- **Preset templates**: Built-in configuration templates (quickstart, isfm, gfs)

### Pipeline Architecture
- **Pipeline_01_default**: Standard training pipeline
- **Pipeline_02_pretrain_fewshot**: Two-stage pretraining + few-shot
- **Pipeline_03_multitask_pretrain_finetune**: Multi-task foundation models
- **Pipeline_ID**: ID-based data processing pipeline

## Infrastructure Requirements

### Hardware Requirements
- **GPU**: CUDA 11.1+ compatible GPUs recommended
- **Memory**: Minimum 8GB GPU memory for large models
- **Storage**: SSD recommended for dataset loading performance
- **CPU**: Multi-core processor for data loading parallelization

### Distributed Training
- **PyTorch DDP**: Distributed data parallel training
- **Multi-GPU support**: Automatic GPU detection and utilization
- **Memory optimization**: Mixed precision training support

### Data Storage
- **H5 format**: Efficient numerical data storage and access
- **Hierarchical organization**: Structured result storage system
- **Metadata management**: Excel-based dataset metadata tracking

## Development Tools

### Testing Framework
- **pytest**: Unit and integration testing
- **coverage**: Code coverage analysis
- **requirements-test.txt**: Separate testing dependencies

### Code Quality
- **Black**: Code formatting (implied by project structure)
- **Type hints**: Python type annotations for better code clarity
- **Modular design**: Clean separation of concerns

### Documentation
- **Markdown**: Documentation in .md format
- **Module-specific docs**: CLAUDE.md files for each major component
- **Configuration guides**: Comprehensive YAML configuration documentation

## Integration Points

### External Services
- **WandB API**: Experiment tracking and visualization
- **ModelScope**: Chinese model and dataset hub
- **Hugging Face**: Transformer model ecosystem

### File Formats
- **YAML**: Configuration files
- **H5/HDF5**: Dataset storage
- **Excel (.xlsx)**: Metadata files
- **JSON**: Result metrics storage
- **PNG/PDF**: Visualization outputs

## Performance Considerations

### Memory Optimization
- **Lazy loading**: On-demand dataset loading
- **Batch processing**: Configurable batch sizes
- **Mixed precision**: fp16 training support
- **Gradient accumulation**: Large effective batch sizes

### Storage Optimization
- **H5 compression**: Efficient dataset storage
- **Result hierarchy**: Organized output structure
- **Checkpoint management**: Model state persistence

## Security & Reproducibility

### Reproducibility
- **Seed management**: Configurable random seeds
- **Deterministic operations**: Consistent results across runs
- **Environment tracking**: Complete dependency versioning
- **Configuration backup**: Automatic experiment config storage

### Data Security
- **No hardcoded credentials**: Environment-based configuration
- **Local storage**: No external data transmission by default
- **Access control**: File-system based security model
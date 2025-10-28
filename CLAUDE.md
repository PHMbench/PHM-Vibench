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
PHM-Vibench v5.0 配置系统提供了极简而强大的实验管理能力：

**核心优势**:
- **统一接口**: 单一`load_config()`函数处理所有配置需求
- **4×4灵活性**: 支持预设/文件/字典/ConfigWrapper × 4种覆盖方式
- **智能合并**: 递归合并嵌套配置，点号展开自动处理
- **链式操作**: 支持copy().update()链式配置构建
- **100%兼容**: 所有现有Pipeline无需修改即可使用

**快速示例**:
```python
from src.configs import load_config
# 从预设加载并覆盖参数
config = load_config('isfm', {'model.d_model': 512, 'task.lr': 0.001})
```

📖 **详细文档**: [配置系统v5.0完整指南](./src/configs/CLAUDE.md)

Configuration sections include:
- `data`: Dataset configuration and preprocessing parameters
- `model`: Model architecture and hyperparameters
- `task`: Task type, loss functions, and training settings
- `trainer`: Training orchestration and hardware settings

## Paper Experiments Configuration

### HSE-Prompt Research Experiments
- **Location**: `paper/2025-10_foundation_model_0_metric/`
- **Configuration**: `paper/2025-10_foundation_model_0_metric/configs/vbench_standard/` (Vbench standard format)
- **Scripts**: `paper/2025-10_foundation_model_0_metric/scripts/` (Standardized experiment scripts)
- **Target**: Validate HSE-Prompt cross-domain generalization capability
- **Performance Goal**: 92.8% cross-domain accuracy, 87.6% 5-shot learning accuracy

### Key Configuration Standards
- **Model**: M_02_ISFM_Prompt (Simplified HSE + Learnable prompts)
- **Embedding**: E_01_HSE_v2 (ISFM_Prompt version) or E_01_HSE (Standard)
- **Backbone**: B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO
- **Task Head**: H_01_Linear_cla (Classification), H_02_distance_cla (Metric learning)
- **Task**: hse_contrastive (HSE contrastive learning)
- **Datasets**: Use target_system_id [1,2,6,5,12] for CWRU, XJTU, THU, Ottawa, JNU

### Quick Start for Paper Experiments
```bash
# 1. Environment validation (5 minutes)
bash paper/2025-10_foundation_model_0_metric/scripts/validation/quick_validation.sh

# 2. Run baseline experiments
python paper/2025-10_foundation_model_0_metric/scripts/baseline_experiments/run_all_baseline.py --quick

# 3. Collect and analyze results
python paper/2025-10_foundation_model_0_metric/scripts/analysis/collect_results.py
```

## Important Configuration Standards

### Paper vs Script Configuration Differences
- **Paper Configs**: Located in `paper/2025-10_foundation_model_0_metric/configs/vbench_standard/` - ✅ RECOMMENDED
  - Uses Vbench standard format
  - Standard component naming (E_01_HSE_v2, B_08_PatchTST, etc.)
  - Consistent data set mapping (target_system_id)
  - Complete documentation and examples
- **Script Configs**: Located in `script/unified_metric/configs/` - ⚠️ LEGACY FORMAT
  - Legacy format, may have compatibility issues
  - Use only for reference or migration purposes
- **Demo Configs**: Located in `configs/demo/` - ✅ VBECH STANDARD
  - Good reference for Vbench configuration patterns

### Model Component Naming Standards
When creating or modifying configuration files, use the following Vbench standard format:

#### ✅ CORRECT Format
```yaml
model:
  embedding: "E_01_HSE_v2"     # ✅ Correct: Vbench standard component name
  backbone: "B_08_PatchTST"    # ✅ Correct: Standard naming with prefix
  task_head: "H_01_Linear_cla" # ✅ Correct: Complete task head name

  # Standard ISFM models
  name: "M_01_ISFM"           # ✅ Correct: Standard ISFM
  name: "M_02_ISFM_Prompt"    # ✅ Correct: ISFM with prompt support
```

#### ❌ INCORRECT Format (AVOID)
```yaml
model:
  embedding: "HSE_Prompt"     # ❌ Wrong: Not a valid component name
  embedding: "HSE_v2"         # ❌ Wrong: Missing E_01_ prefix

  backbone: "Dlinear"        # ❌ Wrong: Missing B_04_ prefix
  backbone: "PatchTST"       # ❌ Wrong: Missing B_08_ prefix
  backbone: "TimesNet"       # ❌ Wrong: Missing B_06_ prefix

  task_head: "Linear_cla"    # ❌ Wrong: Missing H_01_ prefix
  task_head: "classification" # ❌ Wrong: Not a valid component name

  name: "ISFM_Prompt"        # ❌ Wrong: Missing M_02_ prefix
```

### Dataset Configuration Standards

#### ✅ CORRECT Dataset Configuration
```yaml
task:
  # Use system_id for dataset specification
  target_system_id: [1, 2, 6, 5, 12]  # ✅ Correct: Use system ID

# Dataset mapping:
# System ID 1: CWRU      (Case Western Reserve University)
# System ID 2: XJTU      (Xi'an Jiaotong University)
# System ID 5: Ottawa    (University of Ottawa)
# System ID 6: THU       (Tsinghua University)
# System ID 12: JNU      (Jinan University)
```

#### ❌ INCORRECT Dataset Configuration (AVOID IN PAPER CONFIGS)
```yaml
task:
  # ❌ WRONG: Do not use dataset names in paper configs
  target_domains: ["CWRU", "Ottawa-19", "THU-2", "HUST"]

  # ❌ WRONG: Do not mix ID and name formats
  target_system_id: [1, "CWRU", 6]

  # ❌ WRONG: Do not use descriptive names
  datasets: ["bearing_faults", "gear_faults"]
```

### Configuration File Best Practices

1. **Always use Vbench standard naming** for components
2. **Prefer system_id over dataset names** for reproducibility
3. **Validate YAML syntax** before running experiments
4. **Use paper configs** (`configs/vbench_standard/`) for new experiments
5. **Document custom configurations** with clear comments

### Configuration Validation Commands
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Validate configuration loading
python -c "from src.configs import load_config; print('✅ Config loaded successfully')"

# Test specific components
python -c "
from src.model_factory import build_model
from src.configs import load_config
config = load_config('your_config.yaml')
model = build_model(config)
print('✅ Model built successfully')
"
```

### Common Configuration Pitfalls and Solutions

#### Problem: Import Error for Components
```
ImportError: No module named 'src.model_factory.ISFM_Prompt'
```
**Solution**: Use correct component names:
- `embedding: "E_01_HSE_v2"` instead of `"HSE_Prompt"`
- `backbone: "B_08_PatchTST"` instead of `"PatchTST"`

#### Problem: Dataset Loading Error
```
FileNotFoundError: Dataset name not found
```
**Solution**: Use system_id instead of dataset names:
- `target_system_id: [1, 2, 6, 5, 12]` instead of `target_domains: ["CWRU", "XJTU"]`

#### Problem: Model Configuration Conflicts
```
ValueError: Invalid model configuration
```
**Solution**: Ensure consistent model naming:
- `name: "M_02_ISFM_Prompt"` with `type: "ISFM_Prompt"`
- Check that all components exist in the factory registry

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
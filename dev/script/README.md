# PHM-Vibench Scripts Directory

## Overview
This directory contains all execution scripts for PHM-Vibench experiments, organized by research focus and experimental approach.

## Directory Structure

### 📁 Core Experiment Scripts
```
script/
├── demo/                            # Quick start examples and demonstrations
│   └── demo.sh                      # Main demo script with common use cases
├── unified_metric/                  # Unified metric learning pipeline
│   ├── README.md                    # Comprehensive pipeline guide
│   ├── configs/                     # Self-contained configs
│   ├── run_unified_experiments.py  # Main orchestrator
│   ├── quick_validate.py            # Pre-flight validation
│   ├── sota_comparison.py           # Baseline comparisons
│   ├── collect_results.py           # Results aggregation
│   ├── paper_visualization.py       # Publication figures
│   └── test_1epoch.py               # Quick testing
├── Vibench_paper/                   # Paper-specific experiments
│   ├── CDDG/                        # Cross-Dataset Domain Generalization
│   ├── DG/                          # Domain Generalization
│   ├── foundation\ model/           # Multi-task foundation models
│   └── GFS/                         # Generalized Few-Shot learning
├── cross-system-generalization/     # Cross-system transfer experiments
├── flow_loss_pretraining/           # Flow loss pretraining methods
└── LQ1/                             # Custom research pipelines
```

## 🚀 Quick Start

### Basic PHM-Vibench Usage
All experiments use `main.py` as the entry point with standardized YAML configurations:

```bash
# Basic single dataset experiment
python main.py --config configs/demo/Single_DG/CWRU.yaml

# Cross-dataset domain generalization
python main.py --config configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml

# Pretraining experiments
python main.py --config configs/demo/Pretraining/Pretraining_demo.yaml

# Few-shot learning
python main.py --config configs/demo/Few_Shot/CWRU.yaml

# Multi-stage pretraining + few-shot
python main.py \
    --pipeline Pipeline_02_pretrain_fewshot \
    --fs_config_path configs/demo/GFS/GFS_demo.yaml \
    --config_path configs/demo/Pretraining/Pretraining_demo.yaml
```

### Run Demo Script
```bash
# Execute all basic examples
bash script/demo/demo.sh
```

## 🧪 Specialized Pipelines

### 1. Unified Metric Learning Pipeline
**Location**: `script/unified_metric/`
**Purpose**: Two-stage unified training with 82% computational savings

```bash
# Quick validation (5 minutes)
python script/unified_metric/test_1epoch.py

# Full pipeline (22 hours)
python script/unified_metric/run_unified_experiments.py --mode complete

# Individual stages
python script/unified_metric/run_unified_experiments.py --mode pretraining
python script/unified_metric/run_unified_experiments.py --mode zero_shot_eval
python script/unified_metric/run_unified_experiments.py --mode finetuning

# Analysis and visualization
python script/unified_metric/collect_results.py --mode analyze
python script/unified_metric/paper_visualization.py --demo

# SOTA comparisons
python script/unified_metric/sota_comparison.py --methods all
```

**Key Features**:
- **Stage 1**: Unified pretraining on all 5 datasets simultaneously
- **Stage 2**: Dataset-specific fine-tuning using unified model
- **Zero-shot evaluation**: >80% accuracy without dataset-specific training
- **Statistical rigor**: Built-in significance testing and effect size analysis
- **Publication ready**: LaTeX tables and 300 DPI figures

### 2. Vibench Paper Experiments
**Location**: `script/Vibench_paper/`
**Purpose**: Systematic evaluation across different model architectures

```bash
# Cross-Dataset Domain Generalization (CDDG)
python main.py --config script/Vibench_paper/CDDG/config_CDDG_B_04_Dlinear.yaml
python main.py --config script/Vibench_paper/CDDG/config_CDDG_B_06_TimesNet.yaml
python main.py --config script/Vibench_paper/CDDG/config_CDDG_B_08_PatchTST.yaml
python main.py --config script/Vibench_paper/CDDG/config_CDDG_B_09_FNO.yaml

# Domain Generalization (DG) - target-specific experiments
python main.py --config script/Vibench_paper/DG/config_DG_B_06_TimesNet_target_1.yaml

# Foundation Model Multi-task Training
bash script/Vibench_paper/foundation\ model/run_multitask_experiments.sh

# Quick foundation model test
bash script/Vibench_paper/foundation\ model/test_multitask.sh
```

**Available Models**:
- **B_04_Dlinear**: Direct linear forecasting
- **B_06_TimesNet**: Time series analysis network
- **B_08_PatchTST**: Patch-based time series transformer
- **B_09_FNO**: Fourier Neural Operator

### 3. Cross-System Generalization
**Location**: `script/cross-system-generalization/`
**Purpose**: Transfer learning across different industrial systems

### 4. Flow Loss Pretraining
**Location**: `script/flow_loss_pretraining/`
**Purpose**: Advanced pretraining with flow-based objectives

## 📊 Configuration Management

### Standard Config Locations
```bash
configs/
├── demo/                           # Demonstration configs
│   ├── Single_DG/                  # Single dataset experiments
│   ├── Multiple_DG/                # Cross-dataset experiments
│   ├── Pretraining/                # Pretraining configs
│   ├── Few_Shot/                   # Few-shot learning
│   └── GFS/                        # Generalized few-shot
├── baseline/                       # Baseline model configs
├── dev/                           # Development configs
└── template/                      # Template configs
```

### Self-Contained Pipeline Configs
```bash
script/unified_metric/configs/      # Unified metric configs (local management)
script/Vibench_paper/*/            # Paper experiment configs (embedded)
```

### Data Configuration
All experiments use the standardized data configuration:
```yaml
data:
  data_dir: "/mnt/crucial/LQ/PHM-Vibench"
  metadata_file: "metadata_6_11.xlsx"
  unified_datasets: ["CWRU", "XJTU", "THU", "Ottawa", "JNU"]
```

## 🎯 Results Organization

### Output Structure
```bash
results/
├── unified_metric_learning/        # Unified metric results
│   ├── pretraining/
│   ├── finetuning/
│   ├── analysis/
│   └── logs/
├── experiments/                    # Paper experiment results
├── save/                          # Standard PHM-Vibench saves
│   └── {metadata_file}/{model_name}/{task_type}_{trainer_name}_{timestamp}/
└── baseline_comparison/           # SOTA comparison results
```

### Key Result Files
- **metrics.json**: Performance metrics
- **config.yaml**: Experiment configuration backup
- **log.txt**: Training logs
- **figures/**: Visualization plots
- **checkpoints/**: Model weights

## 🔧 Development Guidelines

### Adding New Experiments

#### 1. Standard PHM-Vibench Experiments
```bash
# Create config in appropriate directory
configs/your_experiment_type/your_config.yaml

# Run using main.py
python main.py --config configs/your_experiment_type/your_config.yaml
```

#### 2. Self-Contained Pipelines
```bash
# Create directory structure
script/your_pipeline/
├── README.md
├── configs/
├── your_main_script.py
└── analysis/

# Follow unified_metric/ as template
```

### Configuration Best Practices
1. **Use absolute paths** in configs when possible
2. **Set data_dir environment variable** or in config
3. **Include metadata_file** specification
4. **Follow naming conventions**: `config_TaskType_Model_Details.yaml`
5. **Archive old configs** in `.archive/` directories

### Testing Before Full Runs
```bash
# Quick validation
python script/unified_metric/quick_validate.py --mode health_check

# 1-epoch tests
python script/unified_metric/test_1epoch.py

# Single dataset validation
python main.py --config configs/demo/Single_DG/CWRU.yaml
```

## 📈 Performance Optimization

### GPU Memory Management
```yaml
# For 8GB GPUs
data:
  batch_size: 16
  num_workers: 4

# For 16GB+ GPUs
data:
  batch_size: 32
  num_workers: 8
```

### Training Acceleration
```yaml
trainer:
  gradient_checkpointing: true
  mixed_precision: true
  compile_model: true  # PyTorch 2.0+
```

## 🐛 Troubleshooting

### Common Issues

#### GPU Memory Errors
```bash
# Reduce batch size
sed -i 's/batch_size: 32/batch_size: 16/' your_config.yaml

# Enable memory optimization
sed -i 's/gradient_checkpointing: false/gradient_checkpointing: true/' your_config.yaml
```

#### Missing Data
```bash
# Verify data directory
ls /mnt/crucial/LQ/PHM-Vibench/metadata_6_11.xlsx

# Check dataset availability
python -c "
from src.data_factory import data_factory
datasets = ['CWRU', 'XJTU', 'THU', 'Ottawa', 'JNU']
for ds in datasets:
    try:
        reader = data_factory(ds)
        print(f'✅ {ds}: Available')
    except:
        print(f'❌ {ds}: Missing')
"
```

#### Configuration Errors
```bash
# Validate YAML syntax
python -c "
import yaml
with open('your_config.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Config valid')
"

# Test config loading
python -c "
from src.configs import load_config
config = load_config('your_config.yaml')
print('✅ Config loaded successfully')
"
```

## 📚 Additional Resources

### Documentation
- **Main README**: `../README.md` - Project overview
- **Config System**: `../src/configs/CLAUDE.md` - Configuration documentation
- **Data Factory**: `../src/data_factory/CLAUDE.md` - Dataset integration
- **Model Factory**: `../src/model_factory/CLAUDE.md` - Model architectures

### Example Workflows
- **Quick Start**: Use `script/demo/demo.sh`
- **Research Pipeline**: Follow `script/unified_metric/README.md`
- **Paper Experiments**: Check `script/Vibench_paper/readme.md`

### Support
- **Issues**: Report at GitHub repository
- **Questions**: Check existing documentation first
- **Contributing**: Follow project contribution guidelines

---

*PHM-Vibench Team | Updated: 2025-09-16*
# English Prompt: Paper Project README Generator

## Complete Prompt Template

### Version 1: Detailed Version (Recommended for Complex Projects)

```
# Task: Create Paper Project README Documentation

You are a professional scientific documentation assistant. Please create a comprehensive project README for a paper about "[Research Topic]".

## Project Information
- Paper Title: [Full Paper Title]
- Project Name: [Project Name/Repository Name]
- Research Field: [e.g., Fault Diagnosis, Machine Learning, Deep Learning, Computer Vision]
- Research Goal: [Describe the core research goal in 1-2 sentences]
- Main Contributions: [List 2-3 main innovations]
- Project Status: [In Development/Released/Maintained]

## Requirements

Please create the README documentation following the structure below, written in English:

### 1. Project Title & Overview
```
# [Project Name] - [Short Description]

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Version](https://img.shields.io/badge/Version-v1.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Paper](https://img.shields.io/badge/Paper-arXiv-red)

## 📋 Overview

[Brief project background (2-3 paragraphs), including:]
- Current state and challenges in the research field
- Limitations of existing methods
- Our proposed solution and motivation

**Key Features**:
- ✅ [Feature 1: Specific description, e.g., "Novel attention mechanism for enhanced feature extraction"]
- ✅ [Feature 2: Specific description, e.g., "Strong cross-domain generalization for various industrial scenarios"]
- ✅ [Feature 3: Specific description, e.g., "Few-shot learning capability reducing annotation costs"]
- ✅ [Feature 4: Specific description, e.g., "Complete experimental framework for rapid validation"]
- ✅ [Feature 5: Specific description, e.g., "Detailed documentation and example code"]

**Project Structure**:
```
[ProjectName]/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── setup.py                 # Installation script
├── configs/                 # Configuration files
│   ├── baseline.yaml        # Baseline configuration
│   ├── experiment_1.yaml    # Experiment 1 config
│   └── ...
├── src/                     # Source code
│   ├── models/              # Model implementations
│   ├── data/                # Data processing
│   ├── utils/               # Utility functions
│   └── train.py             # Training script
├── scripts/                 # Execution scripts
│   ├── run_all.sh           # Complete experiment script
│   └── run_baseline.sh      # Baseline experiment script
├── experiments/             # Experiment designs
│   ├── experiment_design.md
│   └── results/             # Experiment results
├── docs/                    # Detailed documentation
│   ├── api.md               # API documentation
│   └── tutorials/           # Tutorials
└── tests/                   # Test code
```

### 2. Research Framework
```
## 🎯 Research Framework

### Core Research Questions
This study addresses the following core research questions through systematic experimental design:

#### Question 1: [Question 1 Title, e.g., "Baseline Method Performance Evaluation"]
**Core Question**: [Detailed description of research question]

**Specific Hypotheses**:
- **H0**: [Null hypothesis, e.g., "Traditional methods have limited performance in cross-domain scenarios (accuracy < 70%)"]
- **H1**: [Alternative hypothesis 1, e.g., "Proposed feature learning method improves cross-domain performance (70-80%)"]
- **H2**: [Alternative hypothesis 2, e.g., "Introducing attention mechanism further enhances performance (80-85%)"]
- **H3**: [Alternative hypothesis 3, e.g., "Complete method achieves optimal performance (> 90%)"]

#### Question 2: [Question 2 Title, e.g., "Method Generalization Validation"]
**Core Question**: [Detailed description of research question]

**Specific Hypotheses**:
- [List other hypotheses]

### Experimental Design
[Describe the core ideas of experimental design, including:]
- Progressive validation strategy
- Controlled experiment setup
- Evaluation metric selection
- Statistical significance testing methods

### Expected Contributions
The main contributions of this study include:
1. [Contribution 1: Theoretical innovation]
2. [Contribution 2: Methodological innovation]
3. [Contribution 3: Practical value]
4. [Contribution 4: Open-source contribution]
```

### 3. Experimental System
```
## 📊 Experimental Design

### Precise Experiment Matrix
| Experiment | Research Goal | Method Comparison | Dataset | Expected Performance | Runs | Config File |
|------------|--------------|-------------------|---------|---------------------|------|-------------|
| Exp 0 | Baseline Establishment | Backbone+Head | Dataset A | 65-70% | 5 | configs/exp0.yaml |
| Exp 1 | Feature Learning | +Feature Extraction | Dataset A | 70-75% | 5 | configs/exp1.yaml |
| Exp 2 | Attention Mechanism | +Attention Module | Dataset A | 75-80% | 5 | configs/exp2.yaml |
| Exp 3 | Complete Method | All Components | Dataset A | 80-90% | 5 | configs/exp3.yaml |
| Exp 4 | Generalization Validation | Complete Method | Dataset B | >75% | 5 | configs/exp4.yaml |
| Exp 5 | Ablation Study | Component Combinations | Dataset A | Quantitative Analysis | 30 | configs/exp5.yaml |

### Resource Allocation & Time Estimation
Based on single NVIDIA RTX 3090:

| Experiment | GPU Time/Run | Total GPU Time | Memory | Batch Size | Epochs |
|------------|--------------|----------------|---------|------------|---------|
| Exp 0 | 0.5 hours | 2.5 hours | 8GB | 32 | 100 |
| Exp 1 | 0.6 hours | 3.0 hours | 10GB | 32 | 100 |
| Exp 2 | 0.8 hours | 4.0 hours | 12GB | 32 | 100 |
| Exp 3 | 1.0 hours | 5.0 hours | 12GB | 32 | 100 |

**Total Resource Requirements**:
- GPU Hours: ~15 hours
- Memory Requirement: Up to 12GB
- Storage Space: ~20GB

### Paper Table Correspondence
| Table No. | Table Title | Corresponding Experiment | Validation Focus | Metrics |
|-----------|-------------|--------------------------|------------------|---------|
| Table 1 | Baseline Method Comparison | Exp 0 | Performance Floor | Accuracy, F1 |
| Table 2 | Ablation Study Results | Exp 5 | Component Contributions | Performance Gain |
| Table 3 | Cross-Dataset Generalization | Exp 4 | Generalization Ability | Domain Adaptability |
```

### 4. Quick Start
```
## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/[username]/[project-name].git
cd [project-name]

# Create conda environment
conda create -n [env-name] python=3.9
conda activate [env-name]

# Install PyTorch (adjust based on CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. Data Preparation
```bash
# Download datasets
python scripts/download_data.py --dataset [dataset_name]

# Data preprocessing
python scripts/preprocess_data.py --input_dir [raw_data] --output_dir [processed_data]
```

### 3. Quick Run
```bash
# Run baseline experiment (single dataset, quick verification)
python src/train.py --config configs/baseline.yaml \
                   --dataset [dataset_name] \
                   --epochs 10 \
                   --debug

# Run complete experiments (all configs, multiple datasets)
bash scripts/run_all_experiments.sh
```

### 4. Result Visualization
```bash
# Generate performance report
python scripts/generate_report.py --results_dir experiments/results \
                                 --output_dir reports \
                                 --format pdf
```
```

### 5. Configuration System
```
## ⚙️ Configuration System

### Configuration File Structure
```yaml
# configs/experiment_template.yaml
# =============================================================================
# Experiment [ID]: [Experiment Name]
# Goal: [Experiment Objective]
# =============================================================================

# Environment Configuration
environment:
  project: "[Project Name]"
  seed: 42
  output_dir: "results/experiment_[id]"
  wandb_project: "wandb_project_name"

# Data Configuration
data:
  data_dir: "/path/to/dataset"
  dataset_name: "[Dataset Name]"
  batch_size: 32
  num_workers: 8
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  normalization: "standardization"
  augmentation: true

# Model Configuration
model:
  name: "[Model Name]"
  backbone: "[Backbone Network]"
  embedding_dim: 256
  num_layers: 4
  dropout: 0.1
  # Experiment-specific parameters
  use_attention: true
  attention_heads: 8
  prompt_dim: 128

# Training Configuration
training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.0001
  max_epochs: 100
  early_stopping: true
  patience: 15
  scheduler: "cosine"

# Experiment Configuration
experiment:
  name: "Experiment Name"
  description: "Experiment Description"
  target_metrics: ["accuracy", "f1_score"]
  baseline_comparison: true
```

### Parameter Override System
```bash
# Dynamic configuration adjustment using --override
python src/train.py --config configs/base.yaml \
                   --override data.batch_size=64 \
                   --override training.learning_rate=0.0005 \
                   --override model.use_attention=true \
                   --override environment.seed=123
```

### Configuration Validation
```python
# Validate configuration file completeness
from utils.config import validate_config

config = load_config("configs/experiment.yaml")
is_valid, issues = validate_config(config)
if not is_valid:
    print("Configuration validation failed:", issues)
```
```

### 6. Execution Guide
```
## 🎯 Unambiguous Execution Guide

### Complete Execution Workflow
```bash
# Navigate to project directory
cd /path/to/project

# Phase 1: Environment Verification (5 minutes)
python scripts/check_environment.py

# Phase 2: Single Dataset Validation (30 minutes)
python src/train.py --config configs/quick_test.yaml \
                   --dataset test_dataset \
                   --epochs 5

# Phase 3: Baseline Experiments (2 hours)
for dataset in dataset1 dataset2 dataset3; do
    python src/train.py --config configs/baseline.yaml \
                       --dataset $dataset \
                       --seed 42
done

# Phase 4: Complete Experiments (4 hours)
bash scripts/run_full_experiments.sh

# Phase 5: Result Collection (10 minutes)
python scripts/collect_results.py \
    --input_dir results \
    --output_dir final_results \
    --format both
```

### Batch Experiment Script Example
```bash
#!/bin/bash
# scripts/run_experiments.sh

# Define parameters
datasets=("dataset1" "dataset2" "dataset3")
seeds=(42 123 456)
configs=("baseline.yaml" "method1.yaml" "method2.yaml")

# Loop through all experiment combinations
for config in "${configs[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: config=$config, dataset=$dataset, seed=$seed"
            python src/train.py \
                --config configs/$config \
                --dataset $dataset \
                --seed $seed \
                --output_dir results/${config%.*}/$dataset/seed_$seed
        done
    done
done
```
```

### 7. Result Organization Standards
```
## 📊 Result Organization Standards

### File Naming Conventions
```
results/
├── experiment_[id]_[name]/
│   ├── dataset_[dataset_name]/
│   │   ├── seed_[random_seed]/
│   │   │   ├── config.yaml          # Used configuration
│   │   │   ├── model.pth            # Model weights
│   │   │   ├── training_log.csv     # Training logs
│   │   │   ├── metrics.json         # Evaluation metrics
│   │   │   ├── predictions.npy      # Prediction results
│   │   │   └── visualizations/      # Visualization results
│   │   │       ├── confusion_matrix.png
│   │   │       └── learning_curve.png
│   │   └── aggregated_results.json  # Multi-seed aggregation
│   └── summary.json                # Experiment summary
└── all_experiments_summary.csv      # All experiments summary
```

### Metrics File Format
```json
{
  "experiment_name": "experiment_3_full_method",
  "dataset": "dataset1",
  "seed": 42,
  "model_config": {
    "backbone": "ResNet50",
    "embedding_dim": 256,
    "use_attention": true
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_epochs": 100
  },
  "results": {
    "accuracy": 0.9234,
    "f1_macro": 0.9198,
    "precision_macro": 0.9212,
    "recall_macro": 0.9185,
    "auc_macro": 0.9678,
    "training_time": 5423.7,
    "inference_time": 15.6,
    "peak_memory": 12288
  },
  "class_wise_results": {
    "class_0": {"precision": 0.95, "recall": 0.92, "f1": 0.93},
    "class_1": {"precision": 0.89, "recall": 0.94, "f1": 0.91}
  },
  "timestamp": "2025-01-29T10:30:45Z",
  "git_commit": "abc123def456",
  "hardware": "NVIDIA RTX 3090"
}
```
```

### 8. Troubleshooting
```
## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Environment Issues
**Issue**: Dependency version conflicts
```bash
# Solution: Use virtual environment
conda create -n [project_name] python=3.9
conda activate [project_name]
pip install -r requirements.txt
```

**Issue**: CUDA version mismatch
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Install correct PyTorch version
# Visit https://pytorch.org/get-started/locally/ for correct command
```

#### 2. Memory Issues
**Issue**: GPU out of memory (OOM)
```yaml
# Solution: Adjust configuration
data:
  batch_size: 16  # Reduce from 32 to 16
model:
  gradient_checkpointing: true  # Enable gradient checkpointing
training:
  accumulate_grad_batches: 2  # Gradient accumulation
```

#### 3. Data Issues
**Issue**: Data path errors
```bash
# Check data path
python -c "import os; print(os.path.exists('/path/to/data'))"

# Use absolute paths or set environment variable
export DATA_DIR="/absolute/path/to/data"
```

#### 4. Performance Issues
**Issue**: Slow training speed
```python
# Solution: Performance optimization
# 1. Use mixed precision training
training:
  precision: 16  # Use FP16

# 2. Increase num_workers
data:
  num_workers: 8  # Adjust based on CPU cores

# 3. Use pin_memory
data:
  pin_memory: true
```

### Debugging Tools
```bash
# 1. View real-time logs
tail -f logs/experiment.log

# 2. Monitor GPU usage
watch -n 1 nvidia-smi

# 3. Performance profiling
python -m torch.utils.bottleneck src/train.py --config configs/debug.yaml

# 4. Memory analysis
python -c "import torch; print(torch.cuda.memory_summary())"

# 5. Configuration validation
python scripts/validate_config.py --config configs/experiment.yaml
```

### Performance Optimization Tips
1. **Data Loading Optimization**:
   - Use HDF5 or LMDB format for data storage
   - Pre-compute data augmentation
   - Use multiprocess data loading

2. **Training Optimization**:
   - Use mixed precision training
   - Implement gradient accumulation
   - Enable gradient checkpointing

3. **Model Optimization**:
   - Use more efficient backbones
   - Implement model pruning
   - Quantize model weights
```

### 9. Documentation & Contributing
```
## 📚 Documentation Structure

### Detailed Documentation
- **[Installation Guide](docs/installation.md)** - Detailed environment setup instructions
- **[API Documentation](docs/api.md)** - Complete API reference
- **[Tutorials](docs/tutorials/)** - Tutorials from beginner to advanced
- **[FAQ](docs/faq.md)** - Frequently Asked Questions

### Example Code
- **[Basic Usage](examples/basic_usage.py)** - Simple usage examples
- **[Advanced Usage](examples/advanced_usage.py)** - Complex scenario usage
- **[Custom Model](examples/custom_model.py)** - How to extend the model

## 🤝 Contributing Guide

We welcome all forms of contributions!

### How to Contribute
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Environment Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Code formatting
black .
isort .

# Code linting
flake8 .
mypy .
```

### Commit Convention
- feat: new feature
- fix: bug fix
- docs: documentation update
- style: code formatting adjustment
- refactor: code refactoring
- test: testing related
- chore: build process or auxiliary tool changes
```

### 10. Appendix
```
## 📄 Appendix

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this project in your research, please cite:

```bibtex
@article{[Author]2025,
  title={Paper Title},
  author={[Author List]},
  journal={Journal Name},
  year={2025}
}
```

### Contact
- **Project Maintainer**: [Your Name]
- **Email**: [Your Email]
- **GitHub Issues**: [Project Issues Link]
- **Discussions**: [Discussions Link]

### Acknowledgments
Thanks to the following open-source projects and contributors:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Weights & Biases](https://wandb.ai/) - Experiment tracking platform
- All researchers who contributed to this project

### Changelog
- **v1.0.0** (2025-01-29) - Initial release
- **v1.1.0** (Planned) - New features
- **v1.2.0** (Planned) - Performance optimizations

---

**Last Updated**: 2025-01-29
**Version**: v1.0
**Project Status**: Active Development
**CI/CD**: [![CI](https://github.com/[user]/[project]/workflows/CI/badge.svg)](https://github.com/[user]/[project]/actions)
```

## Special Requirements

1. **Language Requirements**:
   - Write in English
   - Maintain accuracy of technical terms
   - Clear and professional expression

2. **Formatting Requirements**:
   - Use emojis to enhance readability (📋, 🎯, 🚀, ⚙️, 🔧, etc.)
   - Use **bold** for important information
   - Specify language types for code blocks (bash, yaml, python, json, etc.)
   - Align tables properly using Markdown table syntax

3. **Content Requirements**:
   - Provide specific executable command examples
   - Include actual project structure and configurations
   - Scientifically sound experimental design
   - Practical and effective troubleshooting

4. **Style Requirements**:
   - Professional yet easy to understand
   - Clear logic, well-structured
   - Avoid redundancy, highlight key points
   - Maintain positive and friendly tone

Please generate the complete README document based on the provided project information.
```

### Version 2: Simplified Version (For Quick Generation)

```
Please create a README document for the paper project "[Project Name]".

Project Information:
- Research Topic: [Research Topic]
- Main Goal: [1-2 sentences describing goal]
- Innovations: [List 2-3 innovations]
- Project Status: [Development Status]

Requirements: Include the following core sections:
1. Project Overview (with feature list and project structure)
2. Research Framework (research questions and hypotheses)
3. Experimental Design (experiment matrix table)
4. Quick Start (environment setup and run commands)
5. Configuration Guide (YAML configuration examples)
6. Execution Guide (phased execution flow)
7. Result Organization (file naming conventions)
8. Troubleshooting (common problem solutions)

Write in English, include specific code examples, and use emojis to enhance readability.
```

### Version 3: Minimal Version (For Quick Prototyping)

```
# Generate README - [Project Name]

Project: [Project Name]
Topic: [Research Topic]
Goal: [Main Goal]
Innovations: [Innovation 1, Innovation 2, Innovation 3]

Please create a README containing:
- Project brief (within 100 words)
- 3-5 key features (✅ list)
- Installation commands
- Usage examples
- Experiment table
- Contact information

Requirements: English, concise, professional.
```

## Usage Guide

1. **Choose the appropriate version**:
   - Use complete version for complex projects
   - Use simplified version for medium projects
   - Use minimal version for prototype projects

2. **Fill in placeholders**:
   - Replace content in `[ ]` with actual information
   - Maintain format consistency

3. **Customize adjustments**:
   - Add or remove sections based on project characteristics
   - Adjust technical detail depth

4. **Post-generation checks**:
   - Verify commands are executable
   - Check link validity
   - Ensure correct formatting

---

**Created**: 2025-01-29
**Version**: v1.0
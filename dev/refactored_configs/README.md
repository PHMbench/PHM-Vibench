# PHM-Vibench Refactored Framework

## Overview

This directory contains the refactored PHM-Vibench framework that transforms the original research prototype into publication-quality scientific software. The refactoring addresses critical issues in configuration management, factory patterns, scientific code standards, reproducibility, and testing infrastructure.

## Key Improvements

### 1. **Configuration-Driven Architecture** (`config_schema.py`)
- **Hierarchical Configuration**: Structured configuration with validation using Pydantic
- **Type Safety**: Comprehensive type hints with automatic validation
- **Cross-Platform**: Relative paths and automatic device detection
- **Documentation**: Clear parameter descriptions and examples

### 2. **Optimized Factory Patterns** (`improved_factory.py`)
- **Unified Registry**: Single pattern for all component types
- **Rich Metadata**: Scientific references and parameter documentation
- **Better Error Messages**: Detailed error information with suggestions
- **Extensibility**: Easy addition of new components

### 3. **Scientific Code Standards** (`scientific_resnet1d.py`, `scientific_standards_summary.md`)
- **Mathematical Documentation**: Equations and algorithmic steps clearly documented
- **Type Safety**: Complete type annotations including tensor shapes
- **Academic References**: Proper citations for all implemented algorithms
- **Input Validation**: Robust parameter validation with meaningful error messages

### 4. **Reproducibility Framework** (`reproducibility_framework.py`)
- **Deterministic Setup**: Comprehensive seed management across all libraries
- **Environment Tracking**: Complete environment and dependency tracking
- **Configuration Hashing**: Unique identifiers for experiment configurations
- **Result Validation**: Tools for comparing and validating experimental results

### 5. **Testing Infrastructure** (`testing_framework.py`)
- **Unit Tests**: Comprehensive tests for individual components
- **Property-Based Tests**: Mathematical invariant validation using Hypothesis
- **Integration Tests**: End-to-end workflow testing
- **Performance Benchmarks**: Timing and memory usage validation

## File Structure

```
refactored_configs/
├── README.md                           # This file
├── config_schema.py                    # Configuration schema with validation
├── improved_factory.py                 # Optimized factory pattern implementation
├── scientific_resnet1d.py             # Example scientific implementation
├── scientific_standards_summary.md    # Scientific coding standards summary
├── reproducibility_framework.py       # Comprehensive reproducibility framework
├── testing_framework.py               # Testing infrastructure
├── example_experiment.yaml            # Example configuration file
├── usage_example.py                   # Complete usage example
└── comprehensive_refactoring_summary.md # Complete analysis and proposal
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pytorch-lightning pydantic hypothesis pytest numpy pandas scikit-learn
```

### 2. Run the Usage Example

```bash
# Run with default configuration
python refactored_configs/usage_example.py --config refactored_configs/example_experiment.yaml

# Run with custom parameters
python refactored_configs/usage_example.py \
    --config refactored_configs/example_experiment.yaml \
    --num-runs 3 \
    --output-dir custom_results \
    --debug

# Run tests only
python refactored_configs/usage_example.py \
    --config refactored_configs/example_experiment.yaml \
    --test-only
```

### 3. Create Your Own Configuration

```yaml
# my_experiment.yaml
name: "my_custom_experiment"
description: "Custom experiment description"
tags: ["custom", "experiment"]

reproducibility:
  global_seed: 42
  torch_deterministic: true

data:
  data_root: "data/"
  metadata_file: "metadata.xlsx"
  batch_size: 64
  window_size: 1024

model:
  name: "ResNet1D"
  type: "CNN"
  input_dim: 3
  num_classes: 10

optimization:
  optimizer: "adam"
  learning_rate: 0.001
  max_epochs: 100

task:
  name: "classification"
  type: "DG"
```
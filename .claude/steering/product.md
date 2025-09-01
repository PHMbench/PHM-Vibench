# Product Vision - PHM-Vibench

## Overview
PHM-Vibench is a comprehensive benchmark platform for industrial equipment vibration signal analysis, focusing on fault diagnosis and predictive maintenance. It serves as the standardized foundation for PHM (Prognostics and Health Management) research and industrial applications.

## Problem Statement
Industrial equipment fault diagnosis research faces fragmented evaluation environments, non-reproducible results, and unfair algorithm comparisons due to inconsistent data preprocessing, evaluation metrics, and experimental setups.

## Target Users

### Primary Users
- **Academic Researchers**: PhD students, professors conducting PHM research
- **Industrial Engineers**: Equipment maintenance specialists, reliability engineers
- **ML Practitioners**: Data scientists working on industrial fault diagnosis

### User Needs
- Standardized benchmark environment for fair algorithm comparison
- Reproducible experimental framework with consistent evaluation
- Easy integration of new datasets and models
- Quick prototyping of fault diagnosis solutions

## Core Features

### Data Integration
- 15+ industrial datasets (CWRU, XJTU, FEMTO, MFPT, etc.)
- Unified data preprocessing and loading pipeline
- Support for bearings, gears, motors, and other components
- H5 format for efficient data access

### Model Support
- Foundation models (ISFM series) for industrial signals
- Classical ML and modern deep learning architectures
- Transformer, CNN, RNN, and hybrid models
- Modular backbone + task head architecture

### Task Types
- **Classification**: Multi-class fault diagnosis
- **CDDG**: Cross-Dataset Domain Generalization
- **Few-Shot Learning**: Limited sample scenarios
- **Pretraining**: Self-supervised foundation model training

### Experimental Framework
- Configuration-driven experiments via YAML
- Hierarchical result organization and tracking
- Comprehensive evaluation metrics and visualization
- WandB integration for experiment monitoring

## Success Metrics

### Technical Metrics
- **Dataset Coverage**: 15+ integrated datasets
- **Model Variety**: 30+ implemented algorithms
- **Reproducibility**: All experiments fully reproducible
- **Performance**: Efficient training and inference

### Adoption Metrics
- Research paper citations using PHM-Vibench
- Active community contributions (datasets, models)
- Industrial deployment cases
- Documentation quality and completeness

## Business Objectives

### Academic Impact
- Standardize PHM research evaluation methodology
- Accelerate fault diagnosis algorithm development
- Enable fair comparison of research contributions
- Build recognized benchmark in PHM community

### Industry Adoption
- Provide production-ready fault diagnosis solutions
- Reduce development time for industrial applications
- Support equipment maintenance optimization
- Enable transfer learning across different equipment types

## Value Proposition
- **For Researchers**: Focus on algorithm innovation instead of infrastructure
- **For Industry**: Quick deployment of proven fault diagnosis solutions
- **For Community**: Standardized platform advancing the entire field

## Competitive Advantages
- Comprehensive dataset integration (15+ sources)
- Modular factory design for easy extension
- Configuration-driven experiments requiring no code changes
- Strong focus on reproducibility and fair comparison
- Foundation model support for few-shot scenarios
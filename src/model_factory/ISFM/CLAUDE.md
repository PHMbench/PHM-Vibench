# ISFM - CLAUDE.md

This module provides architecture guidance for the Industrial Signal Foundation Model (ISFM) framework in PHM-Vibench. For configuration interface and component details, see [@README.md].

## Architecture Overview

ISFM implements a modular, hierarchical foundation model architecture designed specifically for industrial vibration signal analysis:

```
Input Signal
     ↓
┌─────────────────────────────────────┐
│  Embedding Layer (E_XX_*)           │
│  - Hierarchical Signal Embedding    │
│  - Multi-scale patch extraction      │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Backbone Network (B_XX_*)          │
│  - Transformer / CNN / FNO, etc.    │
│  - Temporal modeling                 │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Task Head (H_XX_*)                 │
│  - Classification / Prediction       │
│  - Multi-task learning support       │
└─────────────────────────────────────┘
     ↓
Output
```

## Design Principles

### 1. Modularity
Each component (embedding, backbone, task_head) is independently configurable and swappable via the registry system.

### 2. Hierarchical Design
- **Embedding**: Handles signal preprocessing and feature extraction
- **Backbone**: Performs temporal/frequency modeling
- **Task Head**: Maps to task-specific outputs

### 3. Registry-Based Configuration
Components are registered with prefixes:
- `E_XX_*`: Embedding layers
- `B_XX_*`: Backbone networks
- `H_XX_*`: Task heads
- `M_XX_*`: Complete ISFM models

## Model Variants

### M_01_ISFM
Basic transformer-based foundation model with modular components.

### M_02_ISFM
Enhanced ISFM with support for heterogeneous batch processing and advanced features.

### M_03_ISFM
Specialized variant for specific industrial scenarios with improved temporal dynamics modeling.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `embedding/` | Signal embedding layers (E_XX_*) |
| `backbone/` | Backbone networks (B_XX_*) |
| `task_head/` | Task-specific output layers (H_XX_*) |
| `layers/` | Shared neural network layers |
| `component/` | Core component documentation |

## Design Decisions

### Why Hierarchical Architecture?
- **Flexibility**: Easy to swap components for different tasks
- **Reusability**: Components can be shared across different ISFM variants
- **Maintainability**: Clear separation of concerns

### Why Registry Pattern?
- **Configuration-driven**: Models assembled from YAML config
- **Extensibility**: New components added without modifying core code
- **Traceability**: Component lineage tracked via registry

## Integration

The ISFM framework integrates with:
- **Data Factory**: Receives windowed signal data
- **Task Factory**: Wrapped in task-specific training logic
- **Configs**: Assembled via 5-block config model

## Related Documentation

- [@README.md] - ISFM User Guide and Configuration
- [readme.md](readme.md) - Detailed Component Documentation
- [@../../README.md] - Model Factory Overview
- [@../Transformer/README.md] - Transformer Backbone Details

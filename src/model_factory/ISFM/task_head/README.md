# ISFM Task Heads

This directory contains task-specific output heads for the Industrial Signal Foundation Model (ISFM).

## Overview

Task heads map the backbone features to task-specific outputs (classification, prediction, etc.). Each head is designed for a specific PHM task type.

## Available Task Heads

| File | Head ID | Task Type | Description |
|------|---------|-----------|-------------|
| `H_01_Linear_cla.py` | `H_01_Linear_cla` | Classification | Basic linear classification head |
| `H_02_distance_cla.py` | `H_02_distance_cla` | Classification | Distance-based classification |
| `H_02_Linear_cla_heterogeneous_batch.py` | `H_02_Linear_cla_heterogeneous` | Classification | Heterogeneous batch handling |
| `H_03_Linear_pred.py` | `H_03_Linear_pred` | Prediction | Linear regression head |
| `H_04_VIB_pred.py` | `H_04_VIB_pred` | Prediction | VIB-specific prediction |
| `H_05_RUL_pred.py` | `H_05_RUL_pred` | RUL Prediction | Remaining Useful Life prediction |
| `H_06_Anomaly_det.py` | `H_06_Anomaly_det` | Anomaly Detection | Anomaly detection head |
| `H_09_multiple_task.py` | `H_09_multiple_task` | Multi-Task | Combined classification/prediction |
| `H_10_ProjectionHead.py` | `H_10_ProjectionHead` | Representation | Projection for contrastive learning |
| `multi_task_head.py` | - | Multi-Task | General multi-task framework |

## Task Categories

### Classification (cla)
Fault classification and diagnosis:
- `H_01_Linear_cla`: Standard linear classifier
- `H_02_distance_cla`: Metric learning classifier
- `H_02_Linear_cla_heterogeneous`: For batch-size mismatch scenarios

### Prediction (pred)
Value and Remaining Useful Life prediction:
- `H_03_Linear_pred`: Standard regression head
- `H_04_VIB_pred`: VIB-specific prediction
- `H_05_RUL_pred`: RUL prediction with uncertainty

### Detection
- `H_06_Anomaly_det`: Anomaly and outlier detection

### Multi-Task
- `H_09_multiple_task`: Handles multiple tasks simultaneously
- `multi_task_head.py`: Framework for defining custom multi-task heads

## Configuration Example

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"

  # Task head selection
  task_head: "H_01_Linear_cla"  # or any H_XX_* ID

  # Head-specific parameters
  num_classes: 10        # For classification
  # OR
  output_dim: 96         # For prediction

  # Multi-task configuration
  task_head: "H_09_multiple_task"
  tasks:
    classification:
      num_classes: 10
    prediction:
      pred_len: 96
```

## Related Documentation

- [@../README.md] - ISFM Module Overview
- [@../backbone/README.md] - Backbone Networks
- [@../../task_factory/README.md] - Task Factory (task definitions)

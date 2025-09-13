# Multi-Task PHM Foundation Model Documentation

## Overview

This document provides comprehensive documentation for the multi-task Prognostics and Health Management (PHM) foundation model implementation within the PHM-Vibench framework. The model simultaneously performs three distinct tasks using the ISFM (Intelligent Signal Foundation Model) architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Components](#components)
3. [Configuration](#configuration)
4. [Usage Guide](#usage-guide)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Architecture Overview

The multi-task PHM foundation model is designed to simultaneously handle three critical PHM tasks:

1. **Fault Classification**: Multi-class classification to identify different types of faults
2. **Remaining Useful Life (RUL) Prediction**: Regression task to predict remaining operational time before failure
3. **Anomaly Detection**: Binary classification to detect abnormal operating conditions

### Model Architecture

```
Input Signal (B, L, C)
        ↓
    Embedding Layer
        ↓
    Backbone Network (ISFM)
        ↓
    Shared Features (B, hidden_dim)
        ↓
    ┌─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    │  Classification │  RUL Prediction │ Anomaly Detection│
    │     Head        │      Head       │      Head       │
    │                 │                 │                 │
    └─────────────────┴─────────────────┴─────────────────┘
```

## Components

### 1. MultiTaskHead (`src/model_factory/ISFM/task_head/multi_task_head.py`)

The core multi-task head module that takes shared feature representations and outputs predictions for all three tasks.

**Key Features:**
- Separate output layers for each task with appropriate dimensions
- Configurable activation functions and hidden dimensions
- Proper weight initialization using Xavier/Glorot method
- Support for both 2D and 3D input tensors
- Comprehensive error handling and input validation

**Parameters:**
- `output_dim`: Dimension of input features from backbone
- `hidden_dim`: Dimension of hidden layers (default: 256)
- `dropout`: Dropout probability (default: 0.1)
- `num_classes`: Dictionary mapping system IDs to number of classes
- `rul_max_value`: Maximum RUL value for output scaling (default: 1000.0)
- `use_batch_norm`: Whether to use batch normalization (default: True)
- `activation`: Activation function name (default: 'relu')

### 2. MultiTaskLightningModule (`src/task_factory/multi_task_lightning.py`)

PyTorch Lightning module that handles multi-task training with combined loss functions and separate metrics tracking.

**Key Features:**
- Configurable loss weights for each task
- Separate metrics tracking (accuracy, F1-score, MSE, MAE, R2, AUROC)
- Support for different optimizers and learning rate schedulers
- Regularization support (L1, L2)
- Comprehensive logging and monitoring

**Supported Loss Functions:**
- Classification: CrossEntropyLoss
- RUL Prediction: MSELoss or L1Loss (MAE)
- Anomaly Detection: BCEWithLogitsLoss

### 3. Configuration File (`configs/multi_task_config.yaml`)

Comprehensive YAML configuration file that defines all model, training, and evaluation parameters.

**Key Sections:**
- Environment configuration
- Data preprocessing parameters
- Model architecture settings
- Task-specific configurations
- Training hyperparameters
- Logging and monitoring settings

## Configuration

### Basic Configuration

```yaml
# Model Configuration
model:
  name: "M_01_ISFM"
  task_head: MultiTaskHead
  hidden_dim: 512
  activation: "gelu"
  rul_max_value: 2000.0

# Task Configuration
task:
  enabled_tasks: ['classification', 'rul_prediction', 'anomaly_detection']
  task_weights:
    classification: 1.0
    rul_prediction: 0.8
    anomaly_detection: 0.6
```

### Advanced Configuration

```yaml
# Multi-task specific settings
task:
  # Loss function configuration
  classification_loss: "CE"
  rul_loss: "MSE"
  anomaly_loss: "BCE"
  
  # Task-specific parameters
  rul_prediction:
    max_rul_value: 2000.0
    normalize_targets: true
    loss_weight: 0.8
  
  anomaly_detection:
    threshold: 0.5
    class_weights: [1.0, 2.0]
    loss_weight: 0.6
```

## Usage Guide

### 1. Basic Usage

```python
from src.model_factory.ISFM.task_head.multi_task_head import MultiTaskHead
from argparse import Namespace

# Configure model
args = Namespace(
    output_dim=512,
    hidden_dim=256,
    num_classes={'system1': 5, 'system2': 3},
    rul_max_value=1000.0
)

# Create model
model = MultiTaskHead(args)

# Forward pass for all tasks
x = torch.randn(16, 128, 512)  # (batch, sequence, features)
outputs = model(x, system_id='system1', task_id='all')

# Individual task outputs
classification_output = outputs['classification']  # (16, 5)
rul_output = outputs['rul_prediction']            # (16, 1)
anomaly_output = outputs['anomaly_detection']     # (16, 1)
```

### 2. Training with Lightning

```python
from src.task_factory.multi_task_lightning import MultiTaskLightningModule
import pytorch_lightning as pl

# Create Lightning module
lightning_module = MultiTaskLightningModule(
    network=model,
    args_data=data_args,
    args_model=model_args,
    args_task=task_args,
    args_trainer=trainer_args,
    args_environment=env_args,
    metadata=metadata
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    precision=16
)

# Train model
trainer.fit(lightning_module, train_dataloader, val_dataloader)
```

### 3. Configuration-based Training

```bash
# Using the provided configuration file
python main.py --config configs/multi_task_config.yaml
```

## Training

### Loss Function

The total loss is computed as a weighted combination of individual task losses:

```
L_total = w_cls * L_classification + w_rul * L_rul + w_anom * L_anomaly + L_reg
```

Where:
- `w_cls`, `w_rul`, `w_anom` are configurable task weights
- `L_reg` is the regularization loss (L1 + L2)

### Metrics Tracking

The model tracks the following metrics for each task:

**Classification:**
- Accuracy
- F1-Score
- Precision
- Recall

**RUL Prediction:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

**Anomaly Detection:**
- Accuracy
- F1-Score
- Area Under ROC Curve (AUROC)

### Learning Rate Scheduling

Supported schedulers:
- ReduceLROnPlateau
- CosineAnnealingLR
- StepLR

## Evaluation

### Model Evaluation

```python
# Evaluate on test set
trainer.test(lightning_module, test_dataloader)

# Get predictions
model.eval()
with torch.no_grad():
    predictions = model(test_batch, system_id='system1', task_id='all')
```

### Metrics Computation

```python
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Classification metrics
y_pred_cls = torch.argmax(predictions['classification'], dim=1)
accuracy = accuracy_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')

# RUL prediction metrics
mse = mean_squared_error(y_true_rul, predictions['rul_prediction'])

# Anomaly detection metrics
y_pred_anom = torch.sigmoid(predictions['anomaly_detection']) > 0.5
anom_accuracy = accuracy_score(y_true_anom, y_pred_anom)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Enable gradient accumulation
   - Use mixed precision training (precision=16)

2. **Convergence Issues**
   - Adjust learning rate
   - Modify task weights
   - Check data normalization

3. **Imbalanced Tasks**
   - Adjust task weights in configuration
   - Use class weights for classification
   - Apply different loss functions

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: DEBUG
  
trainer:
  log_every_n_steps: 10
```

## API Reference

### MultiTaskHead

```python
class MultiTaskHead(nn.Module):
    def __init__(self, args_m):
        """Initialize multi-task head.
        
        Args:
            args_m: Configuration namespace with model parameters
        """
    
    def forward(self, x, system_id=None, task_id=None, return_feature=False):
        """Forward pass through multi-task head.
        
        Args:
            x: Input tensor (B, L, C) or (B, C)
            system_id: System identifier for classification
            task_id: Task to execute ('classification', 'rul_prediction', 
                    'anomaly_detection', 'all')
            return_feature: Return intermediate features if True
            
        Returns:
            Task predictions or feature representations
        """
```

### MultiTaskLightningModule

```python
class MultiTaskLightningModule(pl.LightningModule):
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        """Initialize Lightning module for multi-task training.
        
        Args:
            network: Backbone network (ISFM model)
            args_data: Data configuration
            args_model: Model configuration
            args_task: Task configuration
            args_trainer: Trainer configuration
            args_environment: Environment configuration
            metadata: Dataset metadata
        """
```

## Best Practices

1. **Data Preparation**
   - Ensure balanced datasets across tasks
   - Normalize RUL targets appropriately
   - Handle missing labels gracefully

2. **Training Strategy**
   - Start with equal task weights, then adjust based on performance
   - Use learning rate scheduling for better convergence
   - Monitor individual task losses during training

3. **Model Selection**
   - Validate on held-out data from each task
   - Consider task-specific early stopping criteria
   - Save models based on combined validation performance

4. **Deployment**
   - Test inference speed for real-time applications
   - Validate model outputs across different systems
   - Implement proper error handling for production use

## Future Enhancements

1. **Additional Tasks**
   - Severity estimation
   - Failure mode classification
   - Maintenance scheduling

2. **Architecture Improvements**
   - Attention mechanisms for task weighting
   - Dynamic task balancing
   - Meta-learning for few-shot adaptation

3. **Training Enhancements**
   - Curriculum learning
   - Multi-task uncertainty estimation
   - Domain adaptation techniques

# Task Components - CLAUDE.md

This module provides architecture guidance for the core training components in PHM-Vibench. For loss functions, metrics, and usage details, see [@README.md].

## Architecture Overview

The Components module contains reusable building blocks for task implementations:

```
Task Implementation
     ↓
┌─────────────────────────────────────┐
│  Loss Functions                       │
│  - Cross-entropy, MSE, Contrastive    │
│  - Task-specific losses               │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Metrics                              │
│  - Accuracy, F1, AUC, etc.           │
│  - Domain-specific metrics            │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Regularization                       │
│  - Mixup, Label smoothing             │
│  - Gradient penalties                 │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Contrastive Strategies               │
│  - SimCLR, MoCo, BYOL                │
│  - Pretraining strategies             │
└─────────────────────────────────────┘
```

## Key Design Principles

### 1. Modularity
Each component is independently usable:
- Loss functions can be mixed and matched
- Metrics can be combined for multi-task learning
- Regularization methods are composable

### 2. Registry Pattern
Components are registered for easy access:
- `get_loss_fn(name)` - Retrieve loss function
- `get_metrics(names)` - Retrieve metric objects
- `get_regularizer(name)` - Retrieve regularizer

### 3. Extensibility
New components added via registration:
```python
from src.task_factory import register_loss

@register_loss("custom_loss")
class CustomLoss(nn.Module):
    # Implementation
```

## Component Categories

### Loss Functions (`loss.py`, `prediction_loss.py`)
- **Classification**: Cross-entropy, Focal Loss
- **Regression**: MSE, MAE, Huber Loss
- **Contrastive**: SimCLR, MoCo, SupCon
- **Prediction**: Prediction-specific losses

### Metrics (`metrics.py`)
- **Classification**: Accuracy, F1, Precision, Recall, AUC
- **Regression**: MSE, MAE, R², RMSE
- **Domain**: Domain accuracy, discrepancy metrics

### Regularization (`regularization.py`)
- **Mixup**: Data augmentation
- **Label Smoothing**: Regularization for classification
- **Dropout**: Neural network regularization
- **Gradient Penalty**: Domain adversarial penalties

### Contrastive Strategies (`contrastive_strategies.py`)
- **SimCLR**: Simple contrastive learning
- **MoCo**: Momentum contrast
- **BYOL**: Bootstrap your own latent
- **SupCon**: Supervised contrastive

## Usage Example

```python
from src.task_factory.Components.loss import get_loss_fn
from src.task_factory.Components.metrics import get_metrics
from src.task_factory.Components.regularization import MixUpAugmentation

# In your task
self.loss_fn = get_loss_fn("CE")
self.train_metrics = get_metrics(["acc", "f1"])
self.mixup = MixUpAugmentation(alpha=0.2)

def training_step(self, batch, batch_idx):
    x, y = batch
    x, y_a, y_b, lam = self.mixup(x, y)
    logits = self(x)
    loss = self.loss_fn(logits, y_a, y_b, lam)
    return loss
```

## Integration

Components are used by:
- `Default_task` - Standard training tasks
- `ID_task` - ID-based tasks
- Custom task implementations

## Related Documentation

- [@README.md] - Component Usage Guide
- [@README_CONTRASTIVE_STRATEGIES.md](README_CONTRASTIVE_STRATEGIES.md) - Contrastive Learning
- [@../README.md] - Task Factory Overview

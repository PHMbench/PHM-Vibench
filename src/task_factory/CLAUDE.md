# Task Factory - CLAUDE.md

This module provides guidance for working with the task factory system in PHM-Vibench, which defines training objectives, loss functions, and evaluation metrics for different industrial fault diagnosis tasks.  
**Canonical reference:** see `src/task_factory/readme.md` for the up-to-date task format, configuration fields, and examples; this file is a quick overview that points you there.

## Architecture Overview

The task factory implements PyTorch Lightning modules with modular components:
- **task_factory.py**: Core task instantiation and registry
- **Default_task.py**: General-purpose task implementation
- **multi_task_lightning.py**: Multi-task foundation model training
- **ID_task.py**: Memory-efficient task for large datasets
- **task/**: Specific task implementations by category
- **Components/**: Reusable loss functions, metrics, and utilities

## Core Task Categories

### Classification Tasks

#### Standard Classification
```python
# Single-dataset fault classification
task:
  name: "classification"
  type: "Default_task"
  loss: "CE"                    # Cross-entropy loss
  metrics: ["acc", "f1"]        # Accuracy and F1-score
  num_classes: 10              # Automatic from metadata
```

#### Cross-Dataset Domain Generalization (CDDG)
```python
# Multi-domain robustness training
task:
  type: "CDDG"
  name: "classification"
  source_domain_id: [1, 5, 6]  # Training domains
  target_domain_id: 19         # Test domain
  loss: "CE"
  domain_adaptation_loss: "MMD" # Domain adaptation
```

#### Domain Generalization (DG)
```python
# Single-source to single-target domain transfer
task:
  type: "DG"
  name: "classification"
  target_domain_id: 13
  loss: "CE"
  regularization: ["domain_penalty"]
```

### Few-Shot Learning

#### Prototypical Networks
```python
task:
  type: "FS"
  name: "prototypical_network"
  num_support: 5               # Support samples per class
  num_query: 15                # Query samples per class
  num_episodes: 1000           # Training episodes
  distance_metric: "euclidean"
```

#### Generalized Few-Shot (GFS)
```python
task:
  type: "GFS"
  name: "classification"
  num_support: 5
  num_query: 15
  base_classes: 8              # Base classes for training
  novel_classes: 2             # Novel classes for testing
```

### Pretraining Tasks

#### Masked Reconstruction
```python
task:
  type: "pretrain"
  name: "masked_reconstruction"
  mask_ratio: 0.25             # Fraction of signal to mask
  reconstruction_loss: "MSE"    # Mean squared error
  contrastive_weight: 0.1      # Contrastive learning weight
```

#### Multi-Task Pretraining
```python
task:
  type: "pretrain"
  name: "classification_prediction"
  tasks:
    classification:
      loss: "CE"
      weight: 1.0
    prediction:
      loss: "MSE"
      weight: 0.5
      pred_len: 96             # Prediction horizon
```

## Task Components

### Loss Functions (Components/loss.py)
```python
# Available loss functions
loss_functions = {
    "CE": CrossEntropyLoss,      # Classification
    "MSE": MSELoss,              # Regression/Prediction
    "MAE": L1Loss,               # Robust regression
    "Huber": HuberLoss,          # Robust regression
    "Focal": FocalLoss,          # Imbalanced classification
    "Triplet": TripletLoss,      # Metric learning
}

# Usage in configuration
task:
  loss: "CE"
  loss_params:
    label_smoothing: 0.1
    class_weights: [1.0, 2.0, 1.5]  # For imbalanced data
```

### Metrics (Components/metrics.py)
```python
# Available metrics
metrics = {
    "acc": Accuracy,             # Classification accuracy
    "f1": F1Score,              # F1-score (macro/micro)
    "precision": Precision,      # Precision score
    "recall": Recall,           # Recall score
    "mse": MeanSquaredError,    # Regression MSE
    "mae": MeanAbsoluteError,   # Regression MAE
    "r2": R2Score,              # R-squared
}

# Configuration
task:
  metrics: ["acc", "f1", "precision", "recall"]
  metric_params:
    f1:
      average: "macro"          # Macro-averaged F1
    precision:
      average: "weighted"
```

### Regularization (Components/regularization.py)
```python
# Available regularization methods
regularization_methods = {
    "l1": L1Regularization,
    "l2": L2Regularization,
    "dropout": DropoutRegularization,
    "domain_penalty": DomainAdversarialPenalty,
    "mixup": MixupAugmentation,
}

# Configuration
task:
  regularization: ["l2", "dropout"]
  regularization_params:
    l2:
      weight: 1e-4
    dropout:
      p: 0.1
```

## Lightning Module Structure

### Default Task Implementation
```python
class Default_task(pl.LightningModule):
    def __init__(self, network, args_data, args_model, 
                 args_task, args_trainer, args_environment, metadata):
        super().__init__()
        self.network = network
        self.loss_fn = get_loss_fn(args_task.loss)
        self.metrics = get_metrics(args_task.metrics)
        
    def training_step(self, batch, batch_idx):
        (x, y), data_name = batch  # Standard batch format
        logits = self.network(x)
        loss = self.loss_fn(logits, y)
        
        # Log metrics
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.args_task.lr,
            weight_decay=self.args_task.weight_decay
        )
        return optimizer
```

### Multi-Task Implementation
```python
class MultiTaskLightningModule(pl.LightningModule):
    def __init__(self, network, task_configs, **kwargs):
        super().__init__()
        self.network = network
        
        # Set up tasks
        self.tasks = {
            'classification': ClassificationTask(),
            'prediction': PredictionTask(), 
            'anomaly_detection': AnomalyDetectionTask()
        }
        
    def training_step(self, batch, batch_idx):
        (x, y_dict), data_name = batch
        
        # Forward pass
        outputs = self.network(x)
        
        # Calculate task-specific losses
        total_loss = 0
        for task_name, task in self.tasks.items():
            task_loss = task.compute_loss(outputs, y_dict[task_name])
            self.log(f'{task_name}_loss', task_loss)
            total_loss += self.task_weights[task_name] * task_loss
            
        return total_loss
```

## Configuration Patterns

### Basic Task Configuration
```yaml
task:
  name: "classification"
  type: "Default_task"
  
  # Training parameters
  epochs: 100
  lr: 1e-3
  weight_decay: 1e-4
  batch_size: 32
  
  # Loss and metrics
  loss: "CE"
  metrics: ["acc", "f1"]
  
  # Optimization
  optimizer: "adam"
  scheduler: true
  scheduler_type: "cosine"
  
  # Early stopping
  early_stopping: true
  es_patience: 10
```

### Advanced Multi-Domain Task
```yaml
task:
  type: "CDDG"
  name: "classification"
  
  # Domain configuration
  source_domain_id: [1, 5, 6, 13]
  target_domain_id: 19
  num_domains: 5
  
  # Loss configuration
  loss: "CE"
  domain_loss: "MMD"           # Maximum Mean Discrepancy
  domain_loss_weight: 0.1
  
  # Advanced optimization
  lr: 5e-4
  lr_scheduler: "warmup_cosine"
  warmup_epochs: 10
  
  # Regularization
  regularization: ["mixup", "label_smoothing"]
  mixup_alpha: 0.2
  label_smoothing: 0.1
```

### Multi-Task Configuration
```yaml
task:
  name: "multitask"
  type: "pretrain"
  
  # Task definitions
  task_list: ["classification", "prediction"]
  task_weights:
    classification: 1.0
    prediction: 0.5
    
  # Task-specific configs
  classification:
    loss: "CE"
    num_classes: 10
    metrics: ["acc", "f1"]
    
  prediction:
    loss: "MSE" 
    pred_len: 96
    metrics: ["mae", "mse"]
```

## Creating Custom Tasks

### 1. Basic Task Implementation
```python
# In src/task_factory/task/YourCategory/your_task.py
import pytorch_lightning as pl
from ...Components.loss import get_loss_fn
from ...Components.metrics import get_metrics

class YourTask(pl.LightningModule):
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        super().__init__()
        self.network = network
        self.save_hyperparameters(ignore=['network'])
        
        # Initialize components
        self.loss_fn = get_loss_fn(args_task.loss)
        self.train_metrics = get_metrics(args_task.metrics, prefix='train_')
        self.val_metrics = get_metrics(args_task.metrics, prefix='val_')
        
    def training_step(self, batch, batch_idx):
        # Implement your training logic
        pass
        
    def validation_step(self, batch, batch_idx):
        # Implement your validation logic  
        pass
        
    def configure_optimizers(self):
        # Configure your optimizer and scheduler
        pass
```

### 2. Register Task
```python
# Register using decorator
from ...task_factory import register_task

@register_task("YourCategory", "your_task")
class YourTask(pl.LightningModule):
    # Implementation
```

### 3. Add Configuration
```yaml
# In configs/demo/YourCategory/your_task.yaml
task:
  type: "YourCategory"
  name: "your_task"
  # Your task parameters
```

## Advanced Features

### Custom Loss Functions
```python
# In src/task_factory/Components/loss.py
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        # Implement your loss logic
        return loss

# Register in loss dictionary
loss_functions["custom"] = CustomLoss
```

### Metric Learning Tasks
```python
# Triplet loss for representation learning
task:
  type: "pretrain"
  name: "triplet_learning"
  loss: "triplet"
  margin: 0.5
  mining_strategy: "hard_negative"
```

### Hierarchical Classification
```python
# Multi-level fault classification
task:
  type: "Default_task"  
  name: "hierarchical_classification"
  loss: "hierarchical_ce"
  hierarchy_levels: [3, 10]    # Coarse to fine
  level_weights: [0.3, 0.7]
```

## Best Practices

### Task Design
- Use modular components for loss functions and metrics
- Implement proper logging for all important quantities
- Support both single and multi-task scenarios
- Handle variable batch sizes and sequence lengths

### Configuration Management
- Provide sensible defaults for all parameters
- Use hierarchical configs for complex tasks
- Validate configuration consistency
- Document all task-specific parameters

### Training Strategies
- Implement proper learning rate scheduling
- Use early stopping to prevent overfitting
- Log validation metrics for monitoring
- Support gradient accumulation for large batches

## Troubleshooting

### Common Issues
1. **Metric Computation Errors**: Check metric compatibility with output format
2. **Loss Explosion**: Adjust learning rate or add gradient clipping
3. **Memory Issues**: Use gradient checkpointing or reduce batch size
4. **Convergence Problems**: Check loss function and data preprocessing

### Debug Commands
```python
# Test task instantiation
python -c "from src.task_factory.task_factory import task_factory; task = task_factory(...)"

# Check task parameters
print(f"Task parameters: {task.hparams}")
```

## Integration with Other Modules

- **Data Factory**: Receives dataloaders and metadata
- **Model Factory**: Wraps models with task-specific training logic
- **Trainer Factory**: Provides PyTorch Lightning trainers
- **Utils**: Uses registry patterns and configuration utilities

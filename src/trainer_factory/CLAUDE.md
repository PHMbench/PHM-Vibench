# Trainer Factory - CLAUDE.md

This module provides guidance for working with the trainer factory system in PHM-Vibench, which handles PyTorch Lightning trainer configuration, logging, callbacks, and training orchestration.

## Architecture Overview

The trainer factory system manages training infrastructure:
- **trainer_factory.py**: Core trainer instantiation and registry
- **Default_trainer.py**: Standard trainer configuration with logging and callbacks
- Integration with PyTorch Lightning for scalable training
- Support for multiple logging backends (WandB, SwanLab, CSV, TensorBoard)
- Comprehensive callback system for checkpointing, early stopping, and pruning

## Core Components

### Default Trainer Configuration
```python
# Standard trainer with comprehensive logging and callbacks
trainer:
  name: "Default_trainer"
  
  # Training parameters
  num_epochs: 100
  gpus: 1
  accelerator: "gpu"
  strategy: "auto"
  
  # Hardware optimization
  mixed_precision: true
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  
  # Validation and checkpointing
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  save_top_k: 3
```

### Trainer Factory Usage
```python
from src.trainer_factory.trainer_factory import trainer_factory

# Create trainer from configuration
trainer = trainer_factory(
    args_environment,
    args_trainer, 
    args_data,
    experiment_path
)

# The factory automatically:
# 1. Configures logging (CSV, WandB, SwanLab)
# 2. Sets up callbacks (checkpoints, early stopping)
# 3. Configures hardware acceleration
# 4. Returns configured PyTorch Lightning trainer
```

## Logging Systems

### CSV Logger (Always Enabled)
```python
# Basic logging to CSV files
csv_logger = CSVLogger(path, name="logs")
# Outputs: {path}/logs/version_X/metrics.csv
```

### WandB Integration
```python
# Configuration for Weights & Biases logging
environment:
  wandb: true
  project: "phm-vibench-experiments"

# Automatic setup:
# - Experiment tracking and visualization
# - Hyperparameter logging
# - Model artifact storage
# - Distributed training support
```

### SwanLab Integration
```python
# Configuration for SwanLab logging
environment:
  swanlab: true
  project: "phm-vibench-experiments"

# Features:
# - Experiment management
# - Real-time monitoring
# - Model versioning
```

### TensorBoard Support
```python
# TensorBoard logging (optional)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=path)
```

## Callback System

### Model Checkpointing
```python
# Automatic model checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_path,
    filename="epoch{epoch:02d}-val_loss{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
    verbose=True
)

# Configuration
trainer:
  args:
    save_top_k: 3              # Keep best 3 models
    monitor_metric: "val_acc"   # Metric to monitor
    mode: "max"                # Maximize accuracy
```

### Early Stopping
```python
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    verbose=True,
    strict=False
)

# Configuration
trainer:
  args:
    early_stopping: true
    patience: 10
    min_delta: 0.001
```

### Model Pruning
```python
# Neural network pruning for efficiency
pruning_callback = ModelPruning(
    "l1_unstructured",
    amount=0.2,
    use_global_unstructured=True,
    verbose=1
)

# Configuration  
trainer:
  args:
    pruning: true
    pruning_amount: 0.2
    pruning_method: "l1_unstructured"
```

## Training Configuration

### Basic Trainer Setup
```yaml
trainer:
  name: "Default_trainer"
  
  args:
    # Training duration
    num_epochs: 100
    max_steps: -1              # Use epochs instead
    
    # Hardware configuration
    gpus: 1
    accelerator: "gpu"         # "gpu", "cpu", "tpu"
    strategy: "auto"           # DDP strategy
    
    # Precision and performance
    mixed_precision: true      # 16-bit training
    gradient_clip_val: 1.0     # Gradient clipping
    accumulate_grad_batches: 1 # Gradient accumulation
    
    # Validation
    check_val_every_n_epoch: 1
    val_check_interval: 1.0    # Check validation every epoch
    
    # Checkpointing
    enable_checkpointing: true
    save_top_k: 3
    monitor_metric: "val_loss"
    
    # Early stopping
    early_stopping: true
    patience: 15
    
    # Logging
    log_every_n_steps: 50
    enable_progress_bar: true
```

### Distributed Training
```yaml
trainer:
  args:
    # Multi-GPU setup
    gpus: [0, 1, 2, 3]         # Use specific GPUs
    strategy: "ddp"            # Distributed Data Parallel
    sync_batchnorm: true       # Sync batch norm across GPUs
    
    # Multi-node setup
    num_nodes: 2
    accelerator: "gpu"
    
    # Advanced DDP settings
    replace_sampler_ddp: true
    find_unused_parameters: false
```

### Memory Optimization
```yaml
trainer:
  args:
    # Memory management
    gradient_checkpointing: true
    limit_train_batches: 1.0   # Use full training set
    limit_val_batches: 1.0     # Use full validation set
    
    # Precision settings
    precision: 16              # Mixed precision training
    amp_backend: "native"      # Use native AMP
    
    # Batch size optimization
    auto_scale_batch_size: true
    auto_lr_find: true         # Automatic learning rate finding
```

## Advanced Features

### Custom Callbacks
```python
# Create custom callback
class CustomCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")
        
    def on_epoch_end(self, trainer, pl_module):
        # Custom logic at epoch end
        pass

# Register callback
def custom_trainer(args_e, args_t, args_d, path):
    callbacks = [
        ModelCheckpoint(...),
        EarlyStopping(...),
        CustomCallback()
    ]
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        # ... other args
    )
    return trainer
```

### Profiling and Debugging
```python
# Performance profiling
trainer:
  args:
    profiler: "simple"         # "simple", "advanced", "pytorch"
    log_gpu_memory: true       # Log GPU memory usage
    
    # Debugging
    fast_dev_run: false        # Quick debug run
    overfit_batches: 0.01      # Overfit on small subset
    detect_anomaly: true       # Detect NaN/Inf
```

### Resume Training
```python
# Resume from checkpoint
trainer:
  args:
    resume_from_checkpoint: "/path/to/checkpoint.ckpt"
    
# Or automatically resume from last checkpoint
trainer:
  args:
    auto_resume: true
```

## Environment Integration

### Distributed Training Setup
```bash
# Single node, multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 main.py

# Multiple nodes
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=23456 \
    main.py
```

### Environment Variables
```python
# Important environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["LOCAL_RANK"] = "0"          # Set by launcher

# Logging control
is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
```

## Creating Custom Trainers

### 1. Implement Custom Trainer Function
```python
# In src/trainer_factory/CustomTrainer.py
import pytorch_lightning as pl
from src.trainer_factory import register_trainer

@register_trainer("CustomTrainer")
def trainer(args_e, args_t, args_d, path):
    """
    Custom trainer implementation.
    
    Args:
        args_e: Environment configuration
        args_t: Trainer configuration  
        args_d: Data configuration
        path: Experiment output path
        
    Returns:
        pl.Trainer: Configured PyTorch Lightning trainer
    """
    
    # Setup custom callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(path, "checkpoints"),
            # ... checkpoint config
        ),
        # Add custom callbacks
    ]
    
    # Setup loggers
    loggers = [CSVLogger(path)]
    if args_e.wandb:
        loggers.append(WandbLogger(project=args_e.project))
    
    # Create trainer with custom configuration
    trainer = pl.Trainer(
        max_epochs=args_t.num_epochs,
        callbacks=callbacks,
        logger=loggers,
        # ... other custom settings
    )
    
    return trainer
```

### 2. Use Custom Trainer
```yaml
# In configuration file
trainer:
  name: "CustomTrainer"
  
  # Custom trainer parameters
  args:
    custom_param: value
```

## Best Practices

### Performance Optimization
- Use mixed precision (16-bit) training for memory efficiency
- Enable gradient checkpointing for large models
- Use appropriate batch size and accumulation
- Profile training to identify bottlenecks

### Monitoring and Logging
- Always enable CSV logging for basic metrics
- Use WandB/SwanLab for advanced experiment tracking
- Log learning rate, loss, and key metrics
- Monitor GPU memory usage

### Checkpointing Strategy
- Save multiple checkpoints based on validation metrics
- Use meaningful checkpoint filenames
- Enable automatic resuming from checkpoints
- Save final model state for inference

### Distributed Training
- Use DDP for multi-GPU training
- Sync batch normalization across processes
- Ensure data loading is distributed properly
- Handle logging in distributed setting

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Distributed Hanging**: Check network configuration and port availability
3. **Slow Training**: Profile data loading and model forward pass
4. **Checkpoint Errors**: Verify checkpoint path and permissions

### Debug Commands
```python
# Test trainer creation
python -c "from src.trainer_factory.trainer_factory import trainer_factory; trainer = trainer_factory(...)"

# Check trainer configuration
print(trainer.logger)
print(trainer.callbacks)
```

## Integration with Other Modules

- **Task Factory**: Receives PyTorch Lightning modules for training
- **Data Factory**: Gets dataloaders from data factory
- **Model Factory**: Integrates model optimization with training
- **Utils**: Uses configuration utilities and environment setup
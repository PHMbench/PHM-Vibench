# Utils - CLAUDE.md

This module provides guidance for working with the utility system in PHM-Vibench, which contains helper functions, configuration management, registry patterns, and common utilities used throughout the framework.

## Architecture Overview

The utils module provides foundational utilities:
- **registry.py**: Generic registry pattern for component registration
- **config_utils.py**: Configuration loading, saving, and path management
- **utils.py**: General utilities for model loading, logging, and common operations
- **env_builders.py**: Environment setup and initialization
- **pipeline_config.py**: Pipeline configuration management
- **masking.py**: Signal masking utilities for self-supervised learning

## Core Components

### Registry System

The registry pattern enables dynamic component registration and instantiation:

#### Basic Registry Usage
```python
from src.utils.registry import Registry

# Create registry
MODEL_REGISTRY = Registry()

# Register components using decorator
@MODEL_REGISTRY.register("ResNet1D")
class ResNet1D(nn.Module):
    pass

# Or register manually  
MODEL_REGISTRY.register("CustomModel")(CustomModel)

# Retrieve registered components
model_class = MODEL_REGISTRY.get("ResNet1D")
available_models = MODEL_REGISTRY.available()
```

#### Factory Integration
```python
# Common pattern in factories
def register_model(model_type: str, name: str):
    """Decorator to register a model implementation."""
    return MODEL_REGISTRY.register(f"{model_type}.{name}")

@register_model("CNN", "ResNet1D")
class Model(nn.Module):
    # Implementation
```

### Configuration Management

#### Loading Configuration Files
```python
from src.utils.config_utils import load_config, save_config

# Load YAML configuration
config = load_config("configs/demo/Single_DG/CWRU.yaml")

# Handle encoding issues automatically
# Supports UTF-8 and GB18030 encodings

# Configuration structure
config = {
    'environment': {...},
    'data': {...},
    'model': {...}, 
    'task': {...},
    'trainer': {...}
}
```

#### Saving Configurations
```python
# Save configuration to file
save_config(config, "experiments/experiment_config.yaml")

# Automatic backup of experiment configurations
# Saved alongside results for reproducibility
```

#### Path Management
```python
from src.utils.config_utils import generate_experiment_path

# Generate timestamped experiment paths
experiment_path = generate_experiment_path(
    base_path="save",
    metadata_name="metadata_6_11.xlsx",
    model_name="M_01_ISFM",
    task_type="classification"
)
# Output: save/metadata_6_11.xlsx/M_01_ISFM/T_classification_13_20250825/
```

### Utility Functions

#### Model Checkpoint Loading
```python
from src.utils.utils import load_best_model_checkpoint

# Load best model from training
best_model = load_best_model_checkpoint(model, trainer)

# Automatically finds ModelCheckpoint callback
# Loads weights from best_model_path
```

#### Class Number Computation
```python
from src.utils.utils import get_num_classes

# Automatically compute number of classes from metadata
num_classes = get_num_classes(metadata, task_config)

# Handles different label formats:
# - Categorical labels
# - Multi-class scenarios
# - Cross-dataset mapping
```

## Configuration Utilities

### Environment Setup
```python
from src.utils.env_builders import setup_environment

# Setup experiment environment
setup_environment(args_environment)

# Handles:
# - Random seed setting
# - Device configuration
# - Logging initialization
# - Path creation
```

### Pipeline Configuration
```python
from src.utils.pipeline_config import PipelineConfig

# Manage complex pipeline configurations
pipeline_config = PipelineConfig(
    stage1_config="pretraining_config.yaml",
    stage2_config="finetuning_config.yaml"
)

# Support for multi-stage pipelines
# Configuration inheritance and overrides
```

## Masking Utilities

### Signal Masking for Self-Supervised Learning
```python
from src.utils.masking import create_mask, apply_mask

# Create random mask for signals
mask = create_mask(
    sequence_length=1024,
    mask_ratio=0.25,
    mask_type="random"  # "random", "block", "structured"
)

# Apply mask to input signals
masked_input, mask_indices = apply_mask(input_signal, mask)

# Support for different masking strategies:
# - Random masking: Randomly mask individual time steps
# - Block masking: Mask contiguous blocks
# - Structured masking: Pattern-based masking
```

### Masking Patterns
```python
# Block masking for temporal structure
block_mask = create_mask(
    sequence_length=1024,
    mask_ratio=0.25,
    mask_type="block",
    block_size=64
)

# Structured masking for frequency components
freq_mask = create_mask(
    sequence_length=1024,
    mask_ratio=0.3,
    mask_type="frequency",
    freq_bands=[(0, 100), (200, 300)]
)
```

## Common Utility Patterns

### Configuration Namespace Conversion
```python
from types import SimpleNamespace

def dict_to_namespace(config_dict):
    """Convert nested dictionary to namespace for dot notation access."""
    namespace = SimpleNamespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

# Usage
args_data = dict_to_namespace(config['data'])
print(args_data.batch_size)  # Dot notation access
```

### Path Generation
```python
def generate_timestamped_path(base_path, experiment_name):
    """Generate timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_path, f"{experiment_name}_{timestamp}")

# Usage
experiment_dir = generate_timestamped_path("save", "CWRU_experiment")
```

### Device Management
```python
def get_device(prefer_gpu=True):
    """Get appropriate device for computation."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
```

## Logging and Monitoring Utilities

### Experiment Logging
```python
def setup_logging(log_path, level="INFO"):
    """Setup comprehensive logging for experiments."""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_path, "experiment.log")),
            logging.StreamHandler()
        ]
    )
```

### Performance Monitoring
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{operation_name} took {end_time - start_time:.2f} seconds")

# Usage
with timer("Data loading"):
    data = load_dataset(...)
```

## Integration Utilities

### Factory Registration Helpers
```python
def auto_register_modules(registry, module_path, pattern="*.py"):
    """Automatically register all modules in a directory."""
    import glob
    import importlib
    
    module_files = glob.glob(os.path.join(module_path, pattern))
    for module_file in module_files:
        module_name = os.path.basename(module_file)[:-3]  # Remove .py
        if not module_name.startswith('_'):  # Skip __init__, etc.
            module = importlib.import_module(f"{module_path}.{module_name}")
            # Auto-registration logic here
```

### Configuration Validation
```python
def validate_config(config, schema):
    """Validate configuration against schema."""
    required_fields = ['data', 'model', 'task', 'trainer']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration section: {field}")
    
    # Type and value validation
    if config['data']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    return True
```

## Best Practices

### Registry Pattern
- Use descriptive names for registered components
- Implement hierarchical naming (e.g., "CNN.ResNet1D")
- Provide clear error messages for missing components
- Document available registry keys

### Configuration Management  
- Use meaningful default values
- Validate configuration early in pipeline
- Support configuration inheritance and overrides
- Save complete configurations with experiment results

### Path Management
- Use absolute paths when possible
- Create directories automatically
- Handle path separators correctly across platforms
- Include timestamps for unique experiment identification

### Error Handling
- Provide informative error messages
- Handle common failure cases gracefully
- Use logging instead of print statements
- Include context in error messages

## Advanced Utilities

### Reproducibility Utilities
```python
def set_seeds(seed=42):
    """Set seeds for reproducible experiments."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Memory Management
```python
def cleanup_cuda_memory():
    """Clean up CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CPU mode - no CUDA memory tracking"
```

### Configuration Merging
```python
def merge_configs(base_config, override_config):
    """Recursively merge two configuration dictionaries."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Check PYTHONPATH and module structure
2. **Configuration Errors**: Validate YAML syntax and required fields
3. **Path Issues**: Use absolute paths and check permissions
4. **Registry Errors**: Verify component registration and naming

### Debug Utilities
```python
def debug_config(config):
    """Print configuration for debugging."""
    import pprint
    print("Configuration:")
    pprint.pprint(config)

def check_registry_status(registry):
    """Check what's registered in a registry."""
    available = registry.available()
    print(f"Registry contains {len(available)} items:")
    for name in sorted(available.keys()):
        print(f"  - {name}")
```

## Integration with Other Modules

- **All Factories**: Provides registry patterns and configuration utilities
- **Data Factory**: Uses path management and configuration loading
- **Model Factory**: Uses registry for model instantiation
- **Task Factory**: Uses utility functions and configuration helpers
- **Trainer Factory**: Uses logging and checkpoint utilities
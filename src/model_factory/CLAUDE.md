# Model Factory - CLAUDE.md

This module provides guidance for working with the model factory system in PHM-Vibench, which handles neural network architectures, model instantiation, and the Industrial Signal Foundation Model (ISFM) framework.

## Architecture Overview

The model factory uses a modular, hierarchical design:
- **model_factory.py**: Core model instantiation and registry
- **ISFM/**: Industrial Signal Foundation Model implementations
  - **embedding/**: Signal embedding layers
  - **backbone/**: Core network architectures  
  - **task_head/**: Task-specific output layers
- **CNN/**: Convolutional neural networks for signal processing
- **RNN/**: Recurrent networks for temporal modeling
- **Transformer/**: Transformer-based architectures
- **MLP/**: Multi-layer perceptron models
- **NO/**: Neural operator models (FNO, DeepONet)

## Key Model Categories

### ISFM (Industrial Signal Foundation Model)

The core foundation model framework with modular components:

#### ISFM Models
```python
# Configuration for ISFM foundation model
model:
  name: "M_01_ISFM"           # Basic ISFM
  type: "ISFM"                # Model family
  embedding: "E_01_HSE"       # Hierarchical Signal Embedding
  backbone: "B_08_PatchTST"   # Patch-based Transformer
  task_head: "H_01_Linear_cla" # Classification head
```

**Available ISFM Models:**
- **M_01_ISFM**: Basic transformer-based foundation model
- **M_02_ISFM**: Advanced multi-modal foundation model with enhanced attention
- **M_03_ISFM**: Specialized temporal dynamics model for time-series

#### Embedding Layers (E_XX)
Signal preprocessing and embedding strategies:
- **E_01_HSE**: Hierarchical Signal Embedding for multi-scale features
- **E_02_HSE_v2**: Enhanced HSE with improved efficiency
- **E_03_Patch_DPOT**: Patch-based Discrete Precomputed Optimal Transport

#### Backbone Networks (B_XX)
Core processing architectures:
- **B_01_basic_transformer**: Standard transformer encoder
- **B_04_Dlinear**: Direct linear forecasting (efficient baseline)
- **B_06_TimesNet**: Time-series analysis with period detection
- **B_08_PatchTST**: Patch-based time series transformer
- **B_09_FNO**: Fourier Neural Operator for signal processing
- **B_05_Manba**: Mamba-based temporal modeling

#### Task Heads (H_XX)
Output layers for different tasks:
- **H_01_Linear_cla**: Linear classification head
- **H_02_distance_cla**: Distance-based classification
- **H_03_Linear_pred**: Linear prediction head
- **H_09_multiple_task**: Multi-task learning head

### Traditional Architectures

#### CNN Models
```python
model:
  name: "ResNet1D"
  type: "CNN"
  depth: 18
  in_channels: 1
  num_classes: 10
```

**Available CNN Models:**
- **ResNet1D**: 1D ResNet for signal classification
- **AttentionCNN**: CNN with attention mechanisms
- **MultiScaleCNN**: Multi-scale feature extraction
- **MobileNet1D**: Efficient mobile-friendly CNN
- **TCN**: Temporal Convolutional Network

#### RNN Models
```python
model:
  name: "AttentionLSTM"
  type: "RNN" 
  hidden_size: 128
  num_layers: 2
  bidirectional: true
```

**Available RNN Models:**
- **AttentionLSTM**: LSTM with attention mechanism
- **AttentionGRU**: GRU with attention
- **ConvLSTM**: Convolutional LSTM for spatial-temporal data
- **ResidualRNN**: RNN with residual connections

#### Transformer Models
```python
model:
  name: "PatchTST"
  type: "Transformer"
  patch_len: 16
  stride: 8
  d_model: 128
```

**Available Transformer Models:**
- **PatchTST**: Patch-based time series transformer
- **Autoformer**: Decomposition-based transformer
- **Informer**: Efficient long-sequence transformer
- **Linformer**: Linear complexity transformer

## Model Configuration Patterns

### Basic ISFM Configuration
```yaml
model:
  name: "M_01_ISFM"
  type: "ISFM"
  
  # Architecture components
  embedding: "E_01_HSE"
  backbone: "B_08_PatchTST" 
  task_head: "H_01_Linear_cla"
  
  # Model dimensions
  input_dim: 1                # Input channels
  d_model: 128               # Model dimension
  output_dim: 64             # Output embedding size
  
  # Transformer settings
  num_heads: 8               # Attention heads
  num_layers: 6              # Encoder layers
  d_ff: 256                  # Feed-forward dimension
  dropout: 0.1               # Dropout rate
  
  # Patch settings (for PatchTST)
  patch_size_L: 16           # Patch length
  num_patches: 64            # Number of patches
```

### Advanced ISFM with Multi-Task
```yaml
model:
  name: "M_02_ISFM"
  type: "ISFM"
  embedding: "E_02_HSE_v2"
  backbone: "B_06_TimesNet"
  task_head: "H_09_multiple_task"
  
  # Multi-task configuration
  task_configs:
    classification:
      num_classes: 10
      loss_weight: 1.0
    prediction:
      pred_len: 96
      loss_weight: 0.5
```

### CNN Configuration
```yaml
model:
  name: "ResNet1D"
  type: "CNN"
  depth: 18
  in_channels: 1
  num_classes: 10
  dropout: 0.2
  activation: "relu"
```

## Adding New Models

### 1. Create Model Implementation
```python
# In src/model_factory/YourFamily/YourModel.py
import torch.nn as nn

class Model(nn.Module):  # Must be named "Model"
    def __init__(self, args_m, metadata=None):
        super().__init__()
        # Extract parameters from args_m
        self.input_dim = args_m.input_dim
        self.num_classes = args_m.num_classes
        
        # Build your architecture
        self.layers = nn.Sequential(
            # Your layers here
        )
    
    def forward(self, x):
        return self.layers(x)
```

### 2. Register Model (if using registry)
```python
# In your model file
from ...utils.registry import Registry
from ..model_factory import register_model

@register_model("YourFamily", "YourModel")
class Model(nn.Module):
    # Implementation
```

### 3. Create Configuration
```yaml
# In configs/demo/
model:
  name: "YourModel"
  type: "YourFamily"
  # Your model parameters
```

### 4. Add to Factory Registry
```python
# In src/model_factory/YourFamily/__init__.py
from .YourModel import Model as YourModel
```

## ISFM Component Development

### Adding New Embedding Layer
```python
# In src/model_factory/ISFM/embedding/E_XX_YourEmbedding.py
class E_XX_YourEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Embedding implementation
        
    def forward(self, x):
        # Transform input to embedding space
        return embedded_x
```

### Adding New Backbone
```python  
# In src/model_factory/ISFM/backbone/B_XX_YourBackbone.py
class B_XX_YourBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Backbone architecture
        
    def forward(self, x):
        # Process embedded features
        return processed_x
```

### Adding New Task Head
```python
# In src/model_factory/ISFM/task_head/H_XX_YourHead.py
class H_XX_YourHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Task-specific output layers
        
    def forward(self, x):
        # Generate task outputs
        return output
```

## Model Factory Usage

### Instantiate Models
```python
from src.model_factory.model_factory import model_factory

# Create model from configuration
model = model_factory(args_model, metadata)

# The factory automatically:
# 1. Resolves the model module path
# 2. Imports the appropriate Model class
# 3. Passes configuration and metadata
# 4. Returns initialized model
```

### Model Input/Output Shapes
```python
# Standard input format
batch_size = 32
sequence_length = 1024
channels = 1
input_shape = (batch_size, channels, sequence_length)

# ISFM models typically expect:
# - Input: (batch_size, channels, sequence_length)  
# - Output: (batch_size, output_dim) or task-specific shape
```

## Best Practices

### Model Design
- Follow the modular ISFM pattern for new foundation models
- Use consistent parameter naming (d_model, num_heads, etc.)
- Implement proper initialization and dropout
- Support variable input lengths when possible

### Configuration Management
- Use hierarchical configs for complex models
- Provide sensible defaults for model parameters
- Document parameter ranges and interactions
- Include example configurations

### Memory and Performance
- Profile memory usage for large models
- Use gradient checkpointing for memory efficiency
- Implement mixed precision training support
- Consider model compression techniques

## Advanced Features

### Multi-Task Models
```python
# Using H_09_multiple_task head
model:
  task_head: "H_09_multiple_task"
  task_configs:
    classification:
      num_classes: 10
      loss_weight: 1.0
    regression:
      output_dim: 1
      loss_weight: 0.3
```

### Model Ensemble
```python
# Multiple backbone configuration
model:
  ensemble_backbones:
    - "B_08_PatchTST"
    - "B_06_TimesNet"
    - "B_04_Dlinear"
  fusion_strategy: "attention"
```

## Troubleshooting

### Common Issues
1. **Shape Mismatches**: Check input dimensions and model expectations
2. **Memory Errors**: Reduce model size or batch size
3. **Import Errors**: Verify model registration and __init__.py files
4. **Configuration Errors**: Check parameter names and types

### Debug Commands
```python
# Test model instantiation
python -c "from src.model_factory.model_factory import model_factory; model = model_factory(args, metadata); print(model)"

# Check model parameters
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

## Integration with Other Modules

- **Task Factory**: Receives models for training and evaluation
- **Data Factory**: Provides input shape information via metadata
- **Trainer Factory**: Handles model optimization and checkpointing
- **Utils**: Uses registry patterns and configuration utilities
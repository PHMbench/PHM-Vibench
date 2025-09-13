# PHM-Vibench Model Factory API Reference

This document provides a comprehensive reference for the PHM-Vibench Model Factory API, including all available models, configuration parameters, and usage patterns.

## 📋 Table of Contents

1. [Core API](#core-api)
2. [Model Categories](#model-categories)
3. [Configuration Parameters](#configuration-parameters)
4. [Model Interface](#model-interface)
5. [Factory Functions](#factory-functions)
6. [Utilities](#utilities)

## 🔧 Core API

### `build_model(args, metadata=None)`

Main factory function to build models.

**Parameters:**
- `args` (Namespace): Configuration object containing model parameters
- `metadata` (Any, optional): Dataset metadata for model initialization

**Returns:**
- `torch.nn.Module`: Initialized model instance

**Example:**
```python
from src.model_factory import build_model
from argparse import Namespace

args = Namespace(
    model_name='ResNetMLP',
    input_dim=3,
    hidden_dim=256,
    num_classes=4
)
model = build_model(args)
```

### `register_model(name)`

Decorator to register custom models.

**Parameters:**
- `name` (str): Model name for registration

**Example:**
```python
from src.model_factory import register_model

@register_model('CustomModel')
class CustomModel(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        # Model implementation
```

## 🏗️ Model Categories

### MLP Models

#### ResNetMLP
```python
args = Namespace(
    model_name='ResNetMLP',
    input_dim=3,           # Input feature dimension
    hidden_dim=256,        # Hidden layer dimension
    num_layers=6,          # Number of residual blocks
    activation='relu',     # Activation function
    dropout=0.1,           # Dropout probability
    num_classes=4          # Number of classes (classification)
)
```

#### MLPMixer
```python
args = Namespace(
    model_name='MLPMixer',
    input_dim=3,
    patch_size=16,         # Temporal patch size
    hidden_dim=256,        # Model dimension
    num_layers=8,          # Number of mixer blocks
    mlp_ratio=4.0,         # MLP expansion ratio
    dropout=0.1,
    num_classes=4
)
```

#### gMLP
```python
args = Namespace(
    model_name='gMLP',
    input_dim=6,
    hidden_dim=256,
    num_layers=6,
    seq_len=128,           # Required for spatial gating
    dropout=0.1,
    num_classes=3
)
```

### Neural Operators

#### FNO (Fourier Neural Operator)
```python
args = Namespace(
    model_name='FNO',
    input_dim=3,           # Input channels
    output_dim=3,          # Output channels
    modes=16,              # Number of Fourier modes
    width=64,              # Channel width
    num_layers=4,          # Number of FNO layers
    activation='gelu'      # Activation function
)
```

#### DeepONet
```python
args = Namespace(
    model_name='DeepONet',
    input_dim=3,
    branch_depth=6,        # Branch network depth
    trunk_depth=6,         # Trunk network depth
    width=128,             # Network width
    trunk_dim=1,           # Coordinate dimension
    output_dim=3
)
```

#### NeuralODE
```python
args = Namespace(
    model_name='NeuralODE',
    input_dim=3,
    hidden_dim=64,         # ODE function hidden dimension
    num_layers=3,          # ODE function depth
    solver='dopri5',       # ODE solver method
    rtol=1e-3,            # Relative tolerance
    atol=1e-4,            # Absolute tolerance
    num_classes=5
)
```

### Transformer Models

#### Informer
```python
args = Namespace(
    model_name='Informer',
    input_dim=3,
    d_model=512,           # Model dimension
    n_heads=8,             # Number of attention heads
    e_layers=2,            # Encoder layers
    d_layers=1,            # Decoder layers
    d_ff=2048,             # Feed-forward dimension
    factor=5,              # ProbSparse factor
    dropout=0.1,
    num_classes=4
)
```

#### PatchTST
```python
args = Namespace(
    model_name='PatchTST',
    input_dim=3,
    d_model=128,           # Model dimension
    n_heads=8,             # Number of attention heads
    e_layers=3,            # Encoder layers
    patch_len=16,          # Patch length
    stride=8,              # Patch stride
    seq_len=96,            # Input sequence length
    pred_len=24,           # Prediction length
    dropout=0.1
)
```

### RNN Models

#### AttentionLSTM
```python
args = Namespace(
    model_name='AttentionLSTM',
    input_dim=3,
    hidden_dim=128,        # LSTM hidden dimension
    num_layers=2,          # Number of LSTM layers
    dropout=0.1,
    bidirectional=True,    # Bidirectional LSTM
    num_classes=4
)
```

#### ConvLSTM
```python
args = Namespace(
    model_name='ConvLSTM',
    input_dim=3,
    hidden_dim=64,         # ConvLSTM hidden dimension
    kernel_size=3,         # Convolution kernel size
    num_layers=2,          # Number of ConvLSTM layers
    dropout=0.1,
    num_classes=4
)
```

### CNN Models

#### ResNet1D
```python
args = Namespace(
    model_name='ResNet1D',
    input_dim=3,
    block_type='basic',    # 'basic' or 'bottleneck'
    layers=[2, 2, 2, 2],   # Layers per block
    dropout=0.1,
    num_classes=4
)
```

#### TCN (Temporal Convolutional Network)
```python
args = Namespace(
    model_name='TCN',
    input_dim=3,
    num_channels=[25, 25, 25, 25],  # Channels per layer
    kernel_size=3,         # Convolution kernel size
    dropout=0.1,
    num_classes=4
)
```

### ISFM Models

#### ContrastiveSSL
```python
args = Namespace(
    model_name='ContrastiveSSL',
    input_dim=3,
    hidden_dim=256,
    num_layers=6,
    projection_dim=128,    # Projection head dimension
    temperature=0.1,       # Contrastive temperature
    dropout=0.1,
    num_classes=4
)
```

#### MultiModalFM
```python
args = Namespace(
    model_name='MultiModalFM',
    modality_dims={'vibration': 3, 'acoustic': 1, 'thermal': 2},
    hidden_dim=256,
    fusion_type='attention',  # 'attention', 'concat', 'add'
    num_classes=4
)
```

## 🔧 Model Interface

All models implement the following interface:

### `__init__(self, args, metadata=None)`

Initialize the model.

**Parameters:**
- `args` (Namespace): Configuration parameters
- `metadata` (Any, optional): Dataset metadata

### `forward(self, x, data_id=None, task_id=None, **kwargs)`

Forward pass through the model.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `data_id` (Any, optional): Data identifier
- `task_id` (Any, optional): Task identifier
- `**kwargs`: Additional model-specific arguments

**Returns:**
- `torch.Tensor` or `dict`: Model output

### Model-Specific Forward Arguments

#### ISFM Models
```python
# ContrastiveSSL
output = model(x, mode='contrastive')  # Returns dict with loss
output = model(x, mode='downstream')   # Returns predictions

# MaskedAutoencoder
output = model(x, mode='pretrain')     # Returns reconstruction
output = model(x, mode='downstream')   # Returns predictions

# MultiModalFM
x = {'vibration': x1, 'acoustic': x2, 'thermal': x3}
output = model(x)
```

## 🏭 Factory Functions

### Model Registration

```python
# Get all available models
from src.model_factory import get_available_models
models = get_available_models()

# Check if model exists
from src.model_factory import model_exists
exists = model_exists('ResNetMLP')

# Get model class
from src.model_factory import get_model_class
ModelClass = get_model_class('ResNetMLP')
```

### Configuration Validation

```python
# Validate configuration
from src.model_factory import validate_config
is_valid, errors = validate_config(args)

# Get default configuration
from src.model_factory import get_default_config
default_args = get_default_config('ResNetMLP')
```

## 🛠️ Utilities

### Data Preprocessing

```python
from src.utils.preprocessing import normalize_data, create_sequences

# Normalize data
normalized_data = normalize_data(data, method='standard')

# Create sequences for time series
sequences, targets = create_sequences(data, seq_len=100, pred_len=1)
```

### Model Utilities

```python
from src.utils.model_utils import count_parameters, get_model_size

# Count model parameters
num_params = count_parameters(model)

# Get model memory size
model_size_mb = get_model_size(model)
```

### Training Utilities

```python
from src.utils.training import EarlyStopping, ModelCheckpoint

# Early stopping
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

# Model checkpointing
checkpoint = ModelCheckpoint(filepath='best_model.pth', monitor='val_loss')
```

## 📊 Configuration Examples

### Complete Classification Example

```python
args = Namespace(
    # Model selection
    model_name='AttentionLSTM',
    
    # Data parameters
    input_dim=6,           # 6 sensor channels
    seq_len=1024,          # 1024 time steps
    
    # Model architecture
    hidden_dim=128,        # LSTM hidden size
    num_layers=3,          # Number of LSTM layers
    bidirectional=True,    # Bidirectional processing
    
    # Attention parameters
    attention_dim=64,      # Attention dimension
    
    # Regularization
    dropout=0.2,           # Dropout probability
    
    # Task parameters
    num_classes=4,         # 4 fault classes
    
    # Training parameters
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=100
)
```

### Complete Forecasting Example

```python
args = Namespace(
    # Model selection
    model_name='PatchTST',
    
    # Data parameters
    input_dim=8,           # 8 sensor channels
    seq_len=168,           # 1 week of hourly data
    pred_len=24,           # Predict next 24 hours
    
    # Model architecture
    d_model=256,           # Model dimension
    n_heads=8,             # Attention heads
    e_layers=4,            # Encoder layers
    
    # Patching parameters
    patch_len=24,          # 24-hour patches
    stride=12,             # 12-hour stride
    
    # Regularization
    dropout=0.1,
    
    # Training parameters
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=200
)
```

## 🔍 Error Handling

### Common Exceptions

```python
from src.model_factory.exceptions import (
    ModelNotFoundError,
    InvalidConfigurationError,
    IncompatibleDataError
)

try:
    model = build_model(args)
except ModelNotFoundError as e:
    print(f"Model '{args.model_name}' not found: {e}")
except InvalidConfigurationError as e:
    print(f"Invalid configuration: {e}")
except IncompatibleDataError as e:
    print(f"Data incompatibility: {e}")
```

### Debugging

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Model summary
from src.utils.debug import model_summary
summary = model_summary(model, input_shape=(32, 100, 3))
print(summary)
```

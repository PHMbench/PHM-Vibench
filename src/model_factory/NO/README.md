# Neural Operators (NO) - Operator Learning Family

Neural Operators represent a paradigm shift from learning functions to learning operators - mappings between function spaces. These models are particularly powerful for continuous-time dynamics, PDEs, and complex system modeling in industrial applications.

## 📋 Available Models

### 1. **FNO** - Fourier Neural Operator
**Paper**: Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" ICLR 2021

A neural operator that learns mappings between function spaces using spectral convolutions in Fourier domain.

**Key Features**:
- Spectral convolutions in Fourier space
- Resolution-invariant architecture
- Efficient for periodic and quasi-periodic signals
- Strong theoretical foundations

**Configuration**:
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

**Example Usage**:
```python
model = build_model(args)
x = torch.randn(32, 1024, 3)  # (batch, resolution, channels)
output = model(x)             # (32, 1024, 3) same resolution
```

### 2. **DeepONet** - Deep Operator Network
**Paper**: Lu et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators" Nature Machine Intelligence 2021

A neural operator with branch-trunk architecture for learning operators between infinite-dimensional spaces.

**Key Features**:
- Branch network for input functions
- Trunk network for evaluation coordinates
- Universal approximation for operators
- Flexible input/output handling

**Configuration**:
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

### 3. **NeuralODE** - Neural Ordinary Differential Equations
**Paper**: Chen et al. "Neural Ordinary Differential Equations" NeurIPS 2018

Continuous-time neural networks that model dynamics as ODEs, enabling adaptive computation and memory efficiency.

**Key Features**:
- Continuous-time dynamics
- Adaptive step size integration
- Memory-efficient backpropagation
- Natural handling of irregular sampling

**Configuration**:
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

### 4. **GraphNO** - Graph Neural Operator
**Paper**: Li et al. "Neural Operator: Graph Kernel Network for Partial Differential Equations" ICLR 2020 Workshop

Neural operator on graphs using spectral graph convolutions for structured data and irregular geometries.

**Key Features**:
- Spectral graph convolutions
- Handles irregular geometries
- Graph Laplacian eigendecomposition
- Flexible graph structures

**Configuration**:
```python
args = Namespace(
    model_name='GraphNO',
    input_dim=3,
    hidden_dim=64,
    num_layers=4,
    num_eigenvectors=32,   # Graph Laplacian eigenvectors
    graph_size=100,        # Number of graph nodes
    num_classes=4
)
```

### 5. **WaveletNO** - Wavelet Neural Operator
**Paper**: Gupta et al. "Multiwavelet-based Operator Learning for Differential Equations" NeurIPS 2021

Neural operator using wavelet transforms for multi-scale analysis and operator learning.

**Key Features**:
- Multi-scale wavelet decomposition
- Efficient for multi-resolution signals
- Captures both local and global patterns
- Adaptive frequency analysis

**Configuration**:
```python
args = Namespace(
    model_name='WaveletNO',
    input_dim=3,
    hidden_dim=64,
    num_layers=4,
    wavelet='db4',         # Wavelet type
    num_levels=3,          # Decomposition levels
    mode='symmetric',      # Boundary condition
    num_classes=5
)
```

## 🚀 Quick Start Examples

### PDE Solving Example
```python
# Configure FNO for solving Burgers' equation
args = Namespace(
    model_name='FNO',
    input_dim=1,           # Initial condition
    output_dim=1,          # Solution field
    modes=32,              # High-frequency modes
    width=128,
    num_layers=4
)

model = build_model(args)
# Initial condition: u(x, 0)
u0 = torch.randn(16, 256, 1)
# Solution: u(x, T)
solution = model(u0)
```

### Continuous Dynamics Modeling
```python
# Neural ODE for irregular time series
args = Namespace(
    model_name='NeuralODE',
    input_dim=6,           # Multi-sensor data
    hidden_dim=128,
    num_layers=4,
    solver='dopri5',       # Adaptive solver
    num_classes=3
)

model = build_model(args)
x = torch.randn(32, 100, 6)  # Irregular sampling supported
output = model(x)
```

### Operator Learning Example
```python
# DeepONet for learning Green's functions
args = Namespace(
    model_name='DeepONet',
    input_dim=1,           # Source function
    branch_depth=8,
    trunk_depth=8,
    width=256,
    trunk_dim=2,           # 2D coordinates
    output_dim=1           # Field value
)

model = build_model(args)
# Branch input: source function values
branch_input = torch.randn(16, 100, 1)
# Trunk input: evaluation coordinates
trunk_input = torch.randn(16, 50, 2)
field_values = model(branch_input, trunk_input)
```

## 📊 Performance Comparison

| Model | Parameters | Memory | Speed | Best Use Case |
|-------|------------|--------|-------|---------------|
| FNO | 3.4M | High | Fast | Periodic signals, PDEs |
| DeepONet | 2.8M | Medium | Medium | Operator learning |
| NeuralODE | 1.2M | Low* | Slow | Irregular sampling |
| GraphNO | 2.1M | Medium | Medium | Structured data |
| WaveletNO | 2.5M | Medium | Fast | Multi-scale signals |

*Memory-efficient due to adjoint method

## 🔧 Advanced Configuration

### FNO Spectral Settings
```python
args.modes = 32              # Number of Fourier modes to keep
args.padding = 8             # Padding for FFT
args.factor = 4              # Downsampling factor
```

### NeuralODE Solver Options
```python
args.solver = 'dopri5'       # Options: 'euler', 'rk4', 'dopri5', 'adams'
args.adjoint = True          # Use adjoint method for memory efficiency
args.adaptive = True         # Adaptive step size
```

### Graph Construction
```python
# For GraphNO - custom graph construction
def build_sensor_graph(positions):
    """Build graph from sensor positions"""
    # Compute pairwise distances
    distances = torch.cdist(positions, positions)
    # Create adjacency matrix (k-nearest neighbors)
    k = 8
    _, indices = torch.topk(-distances, k, dim=-1)
    # Build edge list
    edges = []
    for i in range(len(positions)):
        for j in indices[i]:
            edges.append([i, j.item()])
    return torch.tensor(edges).T
```

## 📈 Training Strategies

### 1. **Multi-Scale Training**
```python
# Train on multiple resolutions
resolutions = [64, 128, 256, 512]
for epoch in range(num_epochs):
    res = resolutions[epoch % len(resolutions)]
    # Interpolate data to current resolution
    x_res = F.interpolate(x, size=res, mode='linear')
    loss = criterion(model(x_res), y_res)
```

### 2. **Physics-Informed Loss**
```python
def physics_loss(model, x, t):
    """Add physics constraints to loss"""
    u = model(x)
    # Compute derivatives
    u_t = torch.autograd.grad(u, t, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, create_graph=True)[0]
    # PDE residual (example: heat equation)
    residual = u_t - 0.01 * torch.diff(u_x, dim=1)
    return torch.mean(residual**2)
```

### 3. **Transfer Learning**
```python
# Pre-train on synthetic data, fine-tune on real data
# 1. Pre-training phase
pretrain_model(synthetic_data)
# 2. Fine-tuning phase
finetune_model(real_data, freeze_backbone=True)
```

## 🔍 Model Selection Guide

- **FNO**: Best for periodic signals and PDE solving
- **DeepONet**: Ideal for learning input-output operators
- **NeuralODE**: Perfect for irregular time series and continuous dynamics
- **GraphNO**: Use for structured/spatial data with known topology
- **WaveletNO**: Excellent for multi-scale temporal analysis

## 🎯 Application Examples

### Bearing Fault Diagnosis with FNO
```python
# Vibration signal analysis in frequency domain
args = Namespace(
    model_name='FNO',
    input_dim=3,           # 3-axis accelerometer
    modes=64,              # Capture bearing frequencies
    width=128,
    num_classes=4          # Fault types
)
```

### Predictive Maintenance with NeuralODE
```python
# Continuous degradation modeling
args = Namespace(
    model_name='NeuralODE',
    input_dim=8,           # Multiple sensors
    hidden_dim=64,
    solver='dopri5',
    output_dim=1           # Remaining useful life
)
```

## 📚 References

1. Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" ICLR 2021
2. Lu et al. "Learning nonlinear operators via DeepONet" Nature Machine Intelligence 2021
3. Chen et al. "Neural Ordinary Differential Equations" NeurIPS 2018
4. Li et al. "Neural Operator: Graph Kernel Network for PDEs" ICLR 2020 Workshop
5. Gupta et al. "Multiwavelet-based Operator Learning" NeurIPS 2021

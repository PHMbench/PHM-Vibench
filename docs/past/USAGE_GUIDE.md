# PHM-Vibench Model Factory Usage Guide

This comprehensive guide covers everything you need to know about using the PHM-Vibench Model Factory for industrial signal analysis and prognostics health management applications.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Model Selection Guide](#model-selection-guide)
3. [Configuration Reference](#configuration-reference)
4. [Training Strategies](#training-strategies)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Basic Model Usage

```python
from src.model_factory import build_model
from argparse import Namespace
import torch

# Define configuration
args = Namespace(
    model_name='ResNetMLP',
    input_dim=3,
    hidden_dim=256,
    num_classes=4,
    dropout=0.1
)

# Build model
model = build_model(args)

# Forward pass
x = torch.randn(32, 1024, 3)  # (batch, seq_len, features)
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 4)
```

### Training Loop Template

```python
import torch.nn as nn
import torch.optim as optim

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 🎯 Model Selection Guide

### By Task Type

#### Classification Tasks
- **Best Overall**: `AttentionLSTM`, `ResNet1D`
- **Efficiency**: `MLPMixer`, `MobileNet1D`
- **Deep Learning**: `ResNetMLP`, `TransformerRNN`

#### Regression/Forecasting Tasks
- **Time Series**: `Informer`, `PatchTST`, `Autoformer`
- **Simple/Fast**: `Dlinear`, `MLPMixer`
- **Complex Dynamics**: `NeuralODE`, `FNO`

#### Self-Supervised Learning
- **Contrastive**: `ContrastiveSSL`
- **Reconstruction**: `MaskedAutoencoder`
- **Multi-Modal**: `MultiModalFM`

### By Data Characteristics

#### Short Sequences (< 100 steps)
- `ResNetMLP`, `DenseNetMLP`, `AttentionCNN`

#### Long Sequences (> 1000 steps)
- `Informer`, `Linformer`, `FNO`, `TCN`

#### Multi-Modal Data
- `MultiModalFM`, `SignalLanguageFM`

#### Irregular Sampling
- `NeuralODE`, `GraphNO`

## ⚙️ Configuration Reference

### Common Parameters

```python
# Data parameters
input_dim: int          # Input feature dimension
seq_len: int           # Sequence length (for some models)
output_dim: int        # Output dimension (regression)
num_classes: int       # Number of classes (classification)

# Model parameters
hidden_dim: int        # Hidden layer dimension
num_layers: int        # Number of layers
dropout: float         # Dropout probability
activation: str        # Activation function ('relu', 'gelu', 'swish')

# Training parameters
learning_rate: float   # Learning rate
batch_size: int       # Batch size
num_epochs: int       # Training epochs
```

### Model-Specific Parameters

#### Transformer Models
```python
d_model: int          # Model dimension
n_heads: int          # Number of attention heads
e_layers: int         # Encoder layers
d_layers: int         # Decoder layers
d_ff: int            # Feed-forward dimension
```

#### CNN Models
```python
kernel_size: int      # Convolution kernel size
stride: int          # Convolution stride
padding: int         # Convolution padding
dilation: int        # Dilation rate (for TCN)
```

#### RNN Models
```python
bidirectional: bool   # Bidirectional RNN
num_layers: int      # Number of RNN layers
hidden_size: int     # RNN hidden size
```

## 📈 Training Strategies

### 1. Learning Rate Scheduling

```python
# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
```

### 2. Data Augmentation

```python
# Time series augmentation
def augment_signal(x):
    # Add noise
    noise = torch.randn_like(x) * 0.01
    x_aug = x + noise
    
    # Time warping
    # Magnitude scaling
    scale = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
    x_aug = x_aug * scale
    
    return x_aug
```

### 3. Regularization Techniques

```python
# Dropout
model = build_model(args)  # args.dropout = 0.1

# Weight decay
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 4. Transfer Learning

```python
# Pre-train on large dataset
pretrain_model(large_unlabeled_data)

# Fine-tune on target task
for param in model.backbone.parameters():
    param.requires_grad = False  # Freeze backbone

# Only train classifier
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
```

## 📊 Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Basic metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

# MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### Time Series Specific Metrics

```python
# Directional accuracy (for forecasting)
def directional_accuracy(y_true, y_pred):
    true_direction = np.sign(np.diff(y_true, axis=1))
    pred_direction = np.sign(np.diff(y_pred, axis=1))
    return np.mean(true_direction == pred_direction)

# Symmetric MAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
```

## 🎯 Best Practices

### 1. Data Preprocessing

```python
# Normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
X_train_scaled = X_train_scaled.reshape(X_train.shape)
```

### 2. Model Validation

```python
# K-fold cross-validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kfold.split(X):
    # Train and evaluate model
    score = train_and_evaluate(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    scores.append(score)

print(f"CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

### 3. Hyperparameter Tuning

```python
# Grid search example
param_grid = {
    'hidden_dim': [128, 256, 512],
    'num_layers': [3, 6, 9],
    'dropout': [0.1, 0.2, 0.3]
}

best_score = 0
best_params = None

for hidden_dim in param_grid['hidden_dim']:
    for num_layers in param_grid['num_layers']:
        for dropout in param_grid['dropout']:
            args.hidden_dim = hidden_dim
            args.num_layers = num_layers
            args.dropout = dropout
            
            score = evaluate_model(args)
            if score > best_score:
                best_score = score
                best_params = (hidden_dim, num_layers, dropout)
```

### 4. Model Ensemble

```python
# Simple ensemble
models = [build_model(args1), build_model(args2), build_model(args3)]

def ensemble_predict(models, x):
    predictions = []
    for model in models:
        pred = model(x)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM)
```python
# Reduce batch size
args.batch_size = 16  # Instead of 32

# Use gradient accumulation
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. Slow Training
```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 3. Poor Convergence
```python
# Check learning rate
# Try different optimizers
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 4. Overfitting
```python
# Increase regularization
args.dropout = 0.3
args.weight_decay = 1e-3

# Early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = validate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

## 📚 Additional Resources

- [Model Architecture Details](../src/model_factory/README.md)
- [Example Scripts](../examples/)
- [API Reference](API_REFERENCE.md)
- [Performance Benchmarks](BENCHMARKS.md)
- [Contributing Guide](../CONTRIBUTING.md)

## 🤝 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review the example scripts
3. Open an issue on GitHub
4. Contact the development team

Remember to include:
- Model configuration
- Error messages
- Data shapes and types
- System specifications

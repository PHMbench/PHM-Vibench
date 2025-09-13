# Industrial Signal Foundation Models (ISFM)

The ISFM family represents the cutting-edge of foundation models specifically designed for industrial signal analysis. These models leverage self-supervised learning, contrastive learning, and multi-modal approaches to learn rich representations from industrial data.

## 🏗️ Model Architecture Overview

### Foundation Model Components

1. **Embedding Layer**: Converts raw signals into rich representations
2. **Backbone Network**: Core feature extraction and processing
3. **Task Head**: Specialized outputs for different downstream tasks

## 📋 Available Models

### 1. **ContrastiveSSL** - Self-Supervised Contrastive Learning
Learns representations through contrastive learning with temporal augmentations.

**Key Features**:
- Time-series specific augmentations (noise, jittering, masking)
- InfoNCE contrastive loss
- Projection head for representation learning
- Downstream task adaptation

### 2. **MaskedAutoencoder** - Masked Signal Reconstruction
Learns by reconstructing masked portions of industrial signals.

**Key Features**:
- Patch-based masking strategy
- Encoder-decoder architecture
- High masking ratios (75%+)
- Self-supervised pre-training

### 3. **MultiModalFM** - Multi-Modal Foundation Model
Processes multiple signal modalities (vibration, acoustic, thermal) jointly.

**Key Features**:
- Modality-specific encoders
- Cross-modal attention fusion
- Flexible modality combinations
- Joint representation learning

### 4. **SignalLanguageFM** - Signal-Language Foundation Model
Learns joint representations of signals and textual descriptions.

**Key Features**:
- Signal encoder for temporal data
- Text encoder for descriptions
- Contrastive signal-text alignment
- Zero-shot capabilities

### 5. **TemporalDynamicsSSL** - Temporal Dynamics Learning
Self-supervised learning through temporal prediction tasks.

**Key Features**:
- Next-step prediction
- Temporal permutation detection
- Masked reconstruction
- Multi-task self-supervision

## 🚀 Quick Start

### Contrastive Learning Example
```python
args = Namespace(
    model_name='ContrastiveSSL',
    input_dim=3,
    hidden_dim=256,
    projection_dim=128,
    temperature=0.1
)

model = build_model(args)
x = torch.randn(16, 64, 3)
output = model(x, mode='contrastive')
print(f"Contrastive loss: {output['loss']}")
```

### Multi-Modal Example
```python
args = Namespace(
    model_name='MultiModalFM',
    modality_dims={'vibration': 3, 'acoustic': 1, 'thermal': 2},
    hidden_dim=256,
    fusion_type='attention'
)

model = build_model(args)
x = {
    'vibration': torch.randn(16, 64, 3),
    'acoustic': torch.randn(16, 64, 1),
    'thermal': torch.randn(16, 2)
}
output = model(x)
```

## 📊 Pre-training Strategies

### 1. **Contrastive Pre-training**
- Generate augmented views of signals
- Learn representations that are invariant to augmentations
- Transfer to downstream classification/regression tasks

### 2. **Masked Reconstruction**
- Randomly mask signal patches
- Train to reconstruct original signal
- Learn robust temporal representations

### 3. **Multi-Modal Alignment**
- Align different signal modalities
- Learn shared representation space
- Enable cross-modal understanding

## 🔧 Advanced Configuration

### Self-Supervised Learning
```python
# Contrastive learning setup
args.temperature = 0.07      # Contrastive temperature
args.projection_dim = 128    # Projection head dimension
args.augmentation_strength = 0.5  # Augmentation intensity

# Masked autoencoder setup
args.mask_ratio = 0.75       # Masking ratio
args.patch_size = 16         # Patch size for masking
args.decoder_depth = 8       # Decoder layers
```

### Multi-Modal Configuration
```python
# Define modalities and their dimensions
args.modality_dims = {
    'vibration': 3,          # 3-axis accelerometer
    'acoustic': 1,           # Microphone
    'thermal': 2,            # Temperature sensors
    'current': 3             # Motor current (3-phase)
}
args.fusion_type = 'attention'  # Fusion strategy
```

## 📈 Training Pipeline

### Phase 1: Self-Supervised Pre-training
```python
# Large-scale pre-training on unlabeled data
for epoch in range(pretrain_epochs):
    for batch in unlabeled_dataloader:
        # Contrastive learning
        output = model(batch, mode='contrastive')
        loss = output['loss']
        loss.backward()
        optimizer.step()
```

### Phase 2: Downstream Fine-tuning
```python
# Fine-tune on labeled data for specific tasks
for epoch in range(finetune_epochs):
    for batch, labels in labeled_dataloader:
        output = model(batch, mode='downstream')
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
```

## 🎯 Applications

### Fault Diagnosis
- Pre-train on large unlabeled datasets
- Fine-tune for specific fault types
- Achieve better performance with limited labeled data

### Predictive Maintenance
- Learn temporal dynamics from historical data
- Predict remaining useful life
- Multi-modal sensor fusion for robust predictions

### Anomaly Detection
- Learn normal operation patterns
- Detect deviations from learned representations
- Zero-shot anomaly detection capabilities

## 📚 References

1. Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" ICML 2020
2. He et al. "Masked Autoencoders Are Scalable Vision Learners" CVPR 2022
3. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" ICML 2021
4. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" NAACL 2019
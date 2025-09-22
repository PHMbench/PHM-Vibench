# Pretrain Task Module

## Overview

The Pretrain task module implements self-supervised and unsupervised pretraining tasks for developing PHM (Prognostics and Health Management) foundation models. These tasks enable models to learn robust representations from large amounts of unlabeled industrial data before fine-tuning on specific downstream tasks. Pretraining is crucial for building foundation models that can transfer knowledge across different equipment types, fault conditions, and industrial domains.

## Architecture

Pretraining tasks follow the foundation model paradigm where models first learn general representations from large-scale data, then adapt to specific tasks through fine-tuning. This approach is particularly valuable for industrial applications where labeled data is scarce but raw sensor data is abundant.

## Available Tasks

### 1. masked_reconstruction.py ⭐ **Core Foundation Model Task**
**Masked Signal Reconstruction for Unsupervised Pretraining**

- **Purpose**: Learn robust signal representations through masked autoencoder training
- **Method**: Mask portions of input signals and train model to reconstruct masked regions
- **Innovation**: Adapted masked autoencoder paradigm for industrial time series data
- **Use Case**: Foundation model pretraining for vibration signal analysis

### 2. classification_prediction.py
**Multi-Task Pretraining with Classification and Prediction**

- **Purpose**: Joint pretraining on classification and signal prediction tasks
- **Method**: Simultaneous optimization of classification accuracy and future signal prediction
- **Strengths**: Learns both discriminative and generative representations
- **Use Case**: Comprehensive foundation model training with multiple objectives

### 3. classification.py
**Classification-Based Pretraining**

- **Purpose**: Standard supervised pretraining on classification tasks
- **Method**: Pre-train backbone networks on classification objectives
- **Strengths**: Simple, effective when labeled data is available
- **Use Case**: Warm-starting models before specialized task adaptation

### 4. prediction.py
**Signal Prediction Pretraining**

- **Purpose**: Learn temporal dynamics through future signal prediction
- **Method**: Predict future signal values from past observations
- **Strengths**: Captures temporal dependencies and signal evolution
- **Use Case**: Time series foundation models, predictive maintenance applications

## Configuration Examples

### Masked Reconstruction Pretraining
```yaml
task:
  type: "pretrain"
  name: "masked_reconstruction"

  # Masking configuration
  mask_ratio: 0.25             # Fraction of signal to mask
  mask_strategy: "random"      # "random", "block", "temporal"
  patch_size: 16               # Size of signal patches

  # Reconstruction loss
  reconstruction_loss: "MSE"   # "MSE", "L1", "Huber"
  normalize_targets: true      # Normalize reconstruction targets

  # Contrastive learning (optional)
  enable_contrastive: true
  contrastive_weight: 0.1      # Weight for contrastive loss
  temperature: 0.07            # Temperature for contrastive learning

  # Training parameters
  lr: 1e-4
  epochs: 200
  warmup_epochs: 20
```

### Multi-Task Classification + Prediction
```yaml
task:
  type: "pretrain"
  name: "classification_prediction"

  # Task weights
  classification_weight: 1.0   # Weight for classification loss
  prediction_weight: 0.5       # Weight for prediction loss

  # Classification configuration
  classification:
    loss: "CE"                 # Cross-entropy loss
    num_classes: 10            # Number of fault classes

  # Prediction configuration
  prediction:
    loss: "MSE"                # Mean squared error
    prediction_horizon: 64     # Future steps to predict
    input_horizon: 256         # Past steps for prediction

  # Training parameters
  lr: 5e-4
  epochs: 100
```

### Signal Prediction Pretraining
```yaml
task:
  type: "pretrain"
  name: "prediction"

  # Prediction configuration
  prediction_horizon: 128      # Number of future steps
  input_horizon: 512           # Number of past steps
  prediction_loss: "MSE"       # Loss function

  # Temporal modeling
  use_temporal_encoding: true  # Add temporal position encoding
  causal_attention: true       # Use causal attention masks

  # Data augmentation
  noise_augmentation: true     # Add noise during training
  noise_std: 0.01             # Standard deviation of noise

  # Training parameters
  lr: 1e-3
  epochs: 150
```

### Foundation Model Pipeline
```yaml
# Two-stage training: pretrain -> finetune
pipeline: "Pipeline_02_pretrain_fewshot"

# Stage 1: Pretraining
pretrain_config:
  task:
    type: "pretrain"
    name: "masked_reconstruction"
    mask_ratio: 0.15
    epochs: 200
    lr: 1e-4

# Stage 2: Few-shot fine-tuning
finetune_config:
  task:
    type: "FS"
    name: "prototypical_network"
    num_support: 5
    num_query: 15
    epochs: 50
    lr: 1e-5
```

## Key Parameters

### Masking Configuration
- `mask_ratio`: Fraction of input signal to mask (0.15-0.75)
- `mask_strategy`: Masking pattern ("random", "block", "temporal", "frequency")
- `patch_size`: Size of signal patches for masking
- `mask_value`: Value used for masked regions (0.0, "noise", "learned")

### Loss Functions
- `reconstruction_loss`: Loss for masked reconstruction ("MSE", "L1", "Huber")
- `contrastive_weight`: Weight for contrastive learning component
- `prediction_loss`: Loss for temporal prediction tasks
- `regularization_weight`: Weight for regularization terms

### Training Strategy
- `warmup_epochs`: Epochs for learning rate warmup
- `gradient_clipping`: Maximum gradient norm
- `weight_decay`: L2 regularization strength
- `dropout_rate`: Dropout probability during training

## Masking Strategies

### 1. Random Masking
Randomly mask individual time steps:
```yaml
mask_strategy: "random"
mask_ratio: 0.25
mask_probability: 0.15      # Probability per time step
```

### 2. Block Masking
Mask contiguous blocks of signal:
```yaml
mask_strategy: "block"
block_size_range: [8, 64]   # Range of block sizes
num_blocks: 3               # Number of blocks to mask
```

### 3. Temporal Masking
Mask specific temporal patterns:
```yaml
mask_strategy: "temporal"
temporal_pattern: "periodic"  # "periodic", "transient", "startup"
pattern_duration: 32         # Duration of masked patterns
```

### 4. Frequency Masking
Mask specific frequency components:
```yaml
mask_strategy: "frequency"
frequency_bands: [[0, 100], [500, 1000]]  # Frequency ranges to mask
mask_in_frequency_domain: true
```

## Usage Examples

### Foundation Model Pretraining
```bash
# Large-scale masked reconstruction pretraining
python main.py --config configs/demo/Pretraining/foundation_pretrain.yaml
```

### Multi-Dataset Pretraining
```bash
# Pretrain on multiple datasets for robust representations
python main.py --config configs/demo/Pretraining/multi_dataset_pretrain.yaml
```

### Two-Stage Pipeline
```bash
# Complete pretrain + few-shot pipeline
python main.py --pipeline Pipeline_02_pretrain_fewshot \
    --config_path configs/demo/Pretraining/pretrain_stage.yaml \
    --fs_config_path configs/demo/GFS/fewshot_stage.yaml
```

### Ablation Studies
```bash
# Compare different masking strategies
python main.py --config configs/demo/Pretraining/ablation_masking.yaml

# Test different mask ratios
python main.py --config configs/demo/Pretraining/ablation_mask_ratio.yaml
```

## Integration with Framework

### Task Registration
Pretraining tasks are registered with the `@register_task` decorator and integrate seamlessly with the training pipeline.

### Model Compatibility
- **ISFM Models**: Primary target for foundation model pretraining
- **Transformer Architectures**: Excellent for masked reconstruction
- **CNN Backbones**: Support for convolutional pretraining
- **Hybrid Models**: Multi-modal pretraining capabilities

### Data Pipeline
- Support for all 30+ datasets in PHM-Vibench
- Automatic batching and masking during training
- Efficient data loading for large-scale pretraining

## Advanced Features

### 1. Contrastive Learning Integration
Combine masked reconstruction with contrastive learning:
```yaml
enable_contrastive: true
contrastive_weight: 0.1
contrastive_type: "SimCLR"    # "SimCLR", "MoCo", "SwAV"
```

### 2. Multi-Scale Pretraining
Learn representations at multiple time scales:
```yaml
multi_scale_training: true
time_scales: [1, 2, 4, 8]     # Different downsampling factors
scale_weights: [1.0, 0.8, 0.6, 0.4]  # Weights for each scale
```

### 3. Domain-Adaptive Pretraining
Pretrain with domain awareness:
```yaml
domain_adaptive: true
domain_embedding_dim: 64      # Dimension of domain embeddings
num_domains: 10               # Number of source domains
```

### 4. Progressive Training
Gradually increase task difficulty:
```yaml
progressive_training: true
mask_ratio_schedule: "linear"  # "linear", "cosine", "step"
start_mask_ratio: 0.05
end_mask_ratio: 0.25
```

## Evaluation Metrics

### Pretraining Metrics
- **Reconstruction Loss**: Quality of signal reconstruction
- **Masked Accuracy**: Accuracy on masked regions
- **Perplexity**: Model uncertainty on predictions
- **Feature Quality**: Downstream task performance with frozen features

### Transfer Learning Metrics
- **Linear Probe Accuracy**: Performance with linear classifier on frozen features
- **Fine-tuning Performance**: Performance after task-specific fine-tuning
- **Few-shot Transfer**: Performance on few-shot downstream tasks
- **Zero-shot Transfer**: Performance without any task-specific training

## Research Applications

### Foundation Model Development
- Large-scale pretraining for industrial signal analysis
- Transfer learning across different equipment types
- Universal representations for PHM applications

### Self-Supervised Learning
- Learning from unlabeled industrial data
- Discovering temporal and spectral patterns
- Robust feature learning for noisy environments

### Cross-Domain Transfer
- Knowledge transfer across industrial domains
- Adaptation to new equipment with minimal data
- Universal fault diagnosis capabilities

## Best Practices

### 1. Data Preparation
- Use large, diverse datasets for pretraining
- Ensure data quality and preprocessing consistency
- Balance different equipment types and conditions

### 2. Hyperparameter Selection
- Start with established mask ratios (15-25%)
- Use learning rate warmup for stable training
- Monitor reconstruction quality throughout training

### 3. Transfer Strategy
- Freeze backbone for initial evaluation
- Gradually unfreeze layers during fine-tuning
- Use lower learning rates for fine-tuning

## Troubleshooting

### Common Issues
- **Poor reconstruction**: Reduce mask ratio or improve model capacity
- **Overfitting**: Increase dropout, weight decay, or data augmentation
- **Training instability**: Use gradient clipping and learning rate scheduling

### Debug Tips
- Visualize reconstructed signals to assess quality
- Monitor gradient norms during training
- Check feature representations through visualization

## Industrial Deployment

### 1. Foundation Model Serving
```yaml
# Deploy pretrained model for inference
deployment:
  model_type: "foundation"
  checkpoint_path: "pretrained_model.ckpt"
  inference_mode: "feature_extraction"
```

### 2. Incremental Learning
```yaml
# Continuous learning from new data
incremental_learning:
  update_frequency: "daily"
  replay_buffer_size: 10000
  adaptation_lr: 1e-5
```

### 3. Multi-Task Deployment
```yaml
# Deploy for multiple downstream tasks
multi_task_deployment:
  shared_backbone: true
  task_heads: ["classification", "prediction", "anomaly_detection"]
```

## Performance Benchmarks

### Pretraining Scale
- **Small**: 1M samples, 50 epochs, single dataset
- **Medium**: 10M samples, 100 epochs, 3-5 datasets
- **Large**: 100M samples, 200 epochs, all datasets

### Transfer Performance
- **Linear Probe**: 70-85% accuracy with frozen features
- **Fine-tuning**: 85-95% accuracy with end-to-end training
- **Few-shot**: 60-80% accuracy with 5 shots per class

## References

- [Task Factory Documentation](../CLAUDE.md)
- [Masked Autoencoders Paper](https://arxiv.org/abs/2111.06377)
- [Foundation Models Survey](https://arxiv.org/abs/2108.07258)
- [Configuration System](../../../configs/CLAUDE.md)
- [Model Factory](../../../model_factory/CLAUDE.md)
- [Pipeline Documentation](../../../trainer_factory/CLAUDE.md)
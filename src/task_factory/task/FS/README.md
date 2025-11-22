# FS Task Module

## 🚧 实现状态 (Implementation Status)

### ✅ 已实现 (Fully Implemented)
- **原型网络**: `prototypical_network.py` - 完整的原型网络实现
- **匹配网络**: `matching_network.py` - 注意力机制匹配网络
- **KNN特征学习**: `knn_feature.py` - 基于学习特征空间的KNN分类
- **微调方法**: `finetuning.py` - 基于梯度微调的少样本学习
- **基础框架**: `FS.py` - 通用少样本学习框架
- **工具函数**: `utils.py` - 完整的episode生成和评估工具

### 🚧 部分实现 (Partially Implemented)
- **距离度量**: euclidean, cosine等基础度量已实现，高级度量可能部分实现
- **注意力机制**: 基础注意力已实现，复杂变体可能有限制

### ❌ TODO: 待实现 (Not Yet Implemented)
- **多尺度episode**: 动态shot/way数量的复杂episode生成
- **跨域episode**: 多数据集混合episode生成
- **分层少样本**: 基于故障层次的结构化少样本学习
- **高级embedding**: 学习型距离度量等高级特征表示

> **注意**: FS模块实现较为完整，但部分高级功能可能存在限制。

## Overview

The FS (Few-Shot Learning) task module implements meta-learning algorithms designed to learn from very few examples. This is particularly valuable in industrial fault diagnosis where collecting large amounts of labeled fault data is expensive, dangerous, or time-consuming. Few-shot learning enables models to quickly adapt to new fault types or equipment with minimal training data.

## Architecture

FS tasks follow the episodic training paradigm where the model learns from multiple episodes, each containing support samples (for learning) and query samples (for testing). This meta-learning approach enables rapid adaptation to new tasks with minimal examples.

## Available Tasks

### 1. prototypical_network.py
**Prototypical Networks for Few-Shot Classification**

- **Purpose**: Learn class prototypes from support samples and classify query samples based on distance to prototypes
- **Method**: Compute class centroids in embedding space and classify based on nearest prototype
- **Strengths**: Simple, effective, works well with limited data
- **Use Case**: Standard few-shot classification, fault type recognition with few examples

### 2. matching_network.py
**Matching Networks with Attention Mechanisms**

- **Purpose**: Learn to match query samples to support samples using attention-based comparison
- **Method**: Uses attention mechanisms to compare query embeddings with support set
- **Strengths**: Explicit comparison learning, handles variable support set sizes
- **Use Case**: Complex fault pattern matching, cross-equipment adaptation

### 3. knn_feature.py
**K-Nearest Neighbors in Learned Feature Space**

- **Purpose**: Perform KNN classification in learned embedding space
- **Method**: Learn embeddings through episodic training, classify using KNN
- **Strengths**: Non-parametric, interpretable, works with any embedding
- **Use Case**: Baseline method, interpretable fault diagnosis

### 4. finetuning.py
**Fine-tuning Based Few-Shot Learning**

- **Purpose**: Fine-tune pre-trained models on support samples for few-shot adaptation
- **Method**: Rapid adaptation through gradient-based fine-tuning
- **Strengths**: Leverages pre-trained knowledge, fast adaptation
- **Use Case**: Transfer learning from large datasets to specific equipment

### 5. FS.py
**Generic Few-Shot Learning Framework**

- **Purpose**: Base framework for implementing custom few-shot methods
- **Features**: Common utilities, episode generation, evaluation metrics
- **Use Case**: Foundation for implementing new few-shot algorithms

### 6. utils.py
**Utility Functions for Few-Shot Learning**

- **Purpose**: Helper functions for episodic training and evaluation
- **Features**: Episode sampling, distance computations, evaluation utilities
- **Use Case**: Supporting infrastructure for all few-shot methods

## Configuration Examples

### Prototypical Networks
```yaml
task:
  type: "FS"
  name: "prototypical_network"

  # Few-shot configuration
  num_support: 5               # Support samples per class
  num_query: 15                # Query samples per class
  num_way: 5                   # Number of classes per episode
  num_episodes: 1000           # Training episodes

  # Distance metric
  distance_metric: "euclidean"  # "euclidean", "cosine", "manhattan"

  # Training parameters
  lr: 1e-3
  epochs: 100
  episode_batch_size: 4        # Episodes per batch
```

### Matching Networks
```yaml
task:
  type: "FS"
  name: "matching_network"

  # Few-shot configuration
  num_support: 5
  num_query: 15
  num_way: 5
  num_episodes: 1000

  # Attention mechanisms
  use_attention: true
  attention_type: "cosine"     # "cosine", "dot", "concat"

  # LSTM processing
  use_lstm: true
  lstm_layers: 2
  lstm_hidden: 256

  # Training parameters
  lr: 1e-3
  epochs: 100
```

### KNN Feature Learning
```yaml
task:
  type: "FS"
  name: "knn_feature"

  # Few-shot configuration
  num_support: 5
  num_query: 15
  num_way: 5
  num_episodes: 1000

  # KNN parameters
  k_neighbors: 3               # Number of neighbors
  distance_metric: "euclidean"
  weight_function: "uniform"   # "uniform", "distance"

  # Feature learning
  embedding_dim: 512
  feature_backbone: "resnet"   # Feature extraction backbone

  # Training parameters
  lr: 1e-3
  epochs: 100
```

### Fine-tuning Approach
```yaml
task:
  type: "FS"
  name: "finetuning"

  # Few-shot configuration
  num_support: 5
  num_query: 15
  num_way: 5
  num_episodes: 1000

  # Fine-tuning parameters
  inner_lr: 0.01               # Learning rate for inner optimization
  inner_steps: 5               # Number of gradient steps on support set
  meta_lr: 1e-3                # Meta-learning rate

  # Model configuration
  freeze_backbone: false       # Whether to freeze feature extractor
  finetune_layers: ["head"]    # Layers to fine-tune

  # Training parameters
  epochs: 100
```

## Key Parameters

### Episode Configuration
- `num_support`: Number of support samples per class in each episode
- `num_query`: Number of query samples per class in each episode
- `num_way`: Number of classes per episode (N-way classification)
- `num_episodes`: Total number of training episodes

### Meta-Learning
- `episode_batch_size`: Number of episodes processed in parallel
- `meta_lr`: Meta-learning rate for outer optimization
- `inner_lr`: Learning rate for inner optimization (MAML-style methods)
- `inner_steps`: Number of gradient steps for inner optimization

### Distance Metrics
- `distance_metric`: Distance function ("euclidean", "cosine", "manhattan", "learned")
- `temperature`: Temperature scaling for distance-based similarities
- `normalize_features`: Whether to normalize feature vectors

## Meta-Learning Paradigms

### 1. Metric Learning (Prototypical Networks, Matching Networks)
Learn embeddings where similar samples are close and dissimilar samples are far:
```yaml
# Focus on learning good embeddings
embedding_dim: 512
distance_metric: "euclidean"
normalize_features: true
```

### 2. Optimization-Based (Fine-tuning, MAML)
Learn initial parameters that can be quickly adapted:
```yaml
# Focus on rapid adaptation
inner_lr: 0.01
inner_steps: 5
adapt_all_layers: false
```

### 3. Memory-Based (KNN, Attention)
Explicitly compare with stored examples:
```yaml
# Focus on comparison mechanisms
k_neighbors: 3
attention_type: "cosine"
memory_size: 1000
```

## Usage Examples

### Basic Few-Shot Experiment
```bash
# 5-way 5-shot prototypical networks
python main.py --config configs/demo/GFS/proto_5way_5shot.yaml
```

### Cross-Dataset Few-Shot
```bash
# Train on CWRU, test few-shot adaptation to new datasets
python main.py --config configs/demo/GFS/cross_dataset_fewshot.yaml
```

### Pretraining + Few-Shot Pipeline
```bash
# Two-stage: pretrain backbone, then few-shot adaptation
python main.py --pipeline Pipeline_02_pretrain_fewshot \
    --config_path configs/demo/Pretraining/backbone_pretrain.yaml \
    --fs_config_path configs/demo/GFS/fewshot_adapt.yaml
```

### Ablation Studies
```bash
# Compare different shot numbers
python main.py --config configs/demo/GFS/ablation_shots.yaml

# Compare different methods
python main.py --config configs/demo/GFS/ablation_methods.yaml
```

## Integration with Framework

### Task Registration
All FS tasks inherit from `Default_task` and implement episodic training logic.

### Model Compatibility
- **ISFM Models**: Full support with rapid adaptation capabilities
- **CNN Backbones**: ResNet, ConvNet architectures for feature extraction
- **Transformer Models**: Attention-based few-shot learning
- **Custom Embeddings**: HSE and other specialized embedding layers

### Data Pipeline
- Automatic episode generation from datasets
- Support for all 30+ datasets in PHM-Vibench
- Cross-dataset few-shot evaluation protocols

## Episode Generation

### Training Episodes
Each training episode contains:
- N classes randomly sampled from training set
- K support samples per class
- Q query samples per class
- Goal: Classify query samples using only support samples

### Evaluation Episodes
Similar structure but from held-out test classes:
- Ensures genuine few-shot learning (unseen classes)
- Multiple episodes for statistical significance
- Confidence intervals on performance metrics

## Advanced Features

### 1. TODO: Multi-Scale Episodes - NOT IMPLEMENTED
Episodes with varying difficulty:
```yaml
# TODO: multi_scale_episodes: true - NOT IMPLEMENTED
# shot_range: [1, 5, 10]       # Variable number of shots
# way_range: [3, 5, 10]        # Variable number of ways
```

### 2. TODO: Cross-Domain Episodes - NOT IMPLEMENTED
Episodes spanning multiple datasets:
```yaml
# TODO: cross_domain_episodes: true - NOT IMPLEMENTED
# source_datasets: ["CWRU", "XJTU"]
# target_datasets: ["THU", "MFPT"]
```

### 3. TODO: Hierarchical Few-Shot - NOT IMPLEMENTED
Few-shot learning with fault hierarchy:
```yaml
# TODO: hierarchical_episodes: true - NOT IMPLEMENTED
# fault_hierarchy:
#   bearing: ["inner", "outer", "ball"]
#   gear: ["tooth", "shaft", "wear"]
```

## Evaluation Metrics

### Standard Few-Shot Metrics
- **N-way K-shot Accuracy**: Classification accuracy on query set
- **Confidence Intervals**: Statistical significance testing
- **Per-Class Performance**: Individual class accuracies
- **Learning Curves**: Performance vs. number of support samples

### Industrial-Specific Metrics
- **Cross-Equipment Transfer**: Adaptation to new equipment types
- **Cross-Condition Transfer**: Adaptation to new operating conditions
- **Safety-Critical Performance**: Performance on critical fault types

## Research Applications

### Industrial Fault Diagnosis
- Rapid adaptation to new equipment types
- Few-shot fault detection for rare failure modes
- Cross-manufacturer equipment adaptation

### Predictive Maintenance
- Quick deployment to new industrial sites
- Adaptation to seasonal or operational changes
- Learning from expert annotations with minimal data

### Quality Control
- Few-shot defect detection in manufacturing
- Adaptation to new product lines
- Learning from inspector feedback

## Best Practices

### 1. Episode Design
- Use realistic episode configurations matching deployment scenarios
- Balance episode difficulty for stable training
- Include diverse examples in support sets

### 2. Backbone Selection
- Pre-train backbones on large datasets when possible
- Choose architectures appropriate for signal characteristics
- Consider computational constraints for real-time deployment

### 3. Hyperparameter Tuning
- Start with established baselines from computer vision
- Adapt learning rates for signal data characteristics
- Use validation episodes from held-out classes

## Troubleshooting

### Common Issues
- **Episode imbalance**: Ensure balanced class sampling
- **Overfitting to support**: Use regularization and data augmentation
- **Poor generalization**: Increase episode diversity

### Debug Tips
- Visualize episode distributions
- Monitor support vs. query performance gap
- Check feature embedding quality through visualization

## Experimental Protocols

### Standard Benchmarks
- **5-way 1-shot**: Standard few-shot benchmark
- **5-way 5-shot**: Higher data regime
- **10-way 1-shot**: More challenging classification

### Industrial Protocols
- **Cross-Equipment**: Train on equipment A, test on equipment B
- **Cross-Condition**: Train on condition X, test on condition Y
- **Progressive Learning**: Accumulate knowledge across tasks

## References

- [Task Factory Documentation](../CLAUDE.md)
- [Few-Shot Learning Survey](https://arxiv.org/abs/1904.05046)
- [Meta-Learning Literature](https://meta-learning.ml/)
- [Configuration System](../../../configs/CLAUDE.md)
- [Model Factory](../../../model_factory/CLAUDE.md)
# GFS Task Module

## 🚧 实现状态 (Implementation Status)

### ✅ 已实现 (Fully Implemented)
- **基础分类**: `classification.py` - 21行Default_task基础包装器
- **匹配网络**: `matching.py` - 91行基础匹配网络实现
- **基本少样本支持**: 基础的support/query结构

### 🚧 部分实现 (Partially Implemented)
- **基础参数**: base_classes, novel_classes, num_support, num_query等基本配置
- **简单权重**: 基础的base_class_weight, novel_class_weight支持

### ❌ TODO: 待实现 (Not Yet Implemented)
- **知识蒸馏**: teacher-student知识传递机制
- **特征对齐**: MMD等特征分布对齐技术
- **自适应加权**: 基于不确定性的动态权重调整
- **持续学习**: 内存回放和防遗忘机制
- **元学习**: MAML等元学习算法
- **渐进式训练**: 多阶段训练策略
- **高级匹配网络**: 注意力机制和记忆增强
- **多任务GFS**: 跨任务共享学习

> **注意**: 当前实现为基础版本，以下文档描述的大部分高级功能为设计目标。

## Overview

The GFS (Generalized Few-Shot Learning) task module implements generalized few-shot learning algorithms that handle both base classes (with many examples) and novel classes (with few examples) simultaneously. Unlike standard few-shot learning which only considers novel classes, GFS maintains performance on base classes while adapting to new classes. This is crucial for industrial applications where systems must continue recognizing known faults while learning new fault patterns.

## Architecture

GFS extends traditional few-shot learning by jointly optimizing for:
1. **Base class performance**: Maintaining accuracy on well-represented fault types
2. **Novel class adaptation**: Learning new fault types from few examples
3. **Knowledge transfer**: Leveraging base class knowledge to improve novel class learning

This dual-objective approach is essential for incremental learning in industrial fault diagnosis systems.

## Available Tasks

### 1. classification.py
**Standard Generalized Few-Shot Classification**

- **Purpose**: Joint optimization for base and novel classes in generalized few-shot setting
- **Method**: Balanced training on both base classes (many samples) and novel classes (few samples)
- **Strengths**: Maintains base class performance while enabling novel class learning
- **Use Case**: Incremental fault diagnosis, adding new fault types to existing systems

### 2. matching.py
**Matching Networks for Generalized Few-Shot Learning**

- **Purpose**: Attention-based matching extended for generalized few-shot scenarios
- **Method**: Uses matching networks with special handling for base/novel class distinction
- **Strengths**: Explicit comparison learning with base class memory
- **Use Case**: Complex pattern matching across base and novel fault types

## Configuration Examples

### Basic GFS Classification
```yaml
task:
  type: "GFS"
  name: "classification"

  # Class configuration
  base_classes: 8              # Number of base classes (well-represented)
  novel_classes: 2             # Number of novel classes (few-shot)

  # Few-shot configuration
  num_support: 5               # Support samples per novel class
  num_query: 15                # Query samples per novel class
  num_episodes: 1000           # Training episodes

  # GFS-specific parameters
  base_class_weight: 1.0       # Weight for base class loss
  novel_class_weight: 1.0      # Weight for novel class loss
  balance_sampling: true       # Balance base/novel sampling

  # Training parameters
  lr: 1e-3
  epochs: 100
```

### Advanced GFS with Knowledge Distillation ❌ TODO: NOT IMPLEMENTED
```yaml
task:
  type: "GFS"
  name: "classification"

  # Class configuration (BASIC SUPPORT)
  base_classes: 8               # ✅ WORKS
  novel_classes: 2              # ✅ WORKS
  num_support: 5                # ✅ WORKS
  num_query: 15                 # ✅ WORKS

  # TODO: Knowledge transfer - NOT IMPLEMENTED
  # use_knowledge_distillation: true
  # distillation_weight: 0.5     # Weight for distillation loss
  # teacher_temperature: 4.0     # Temperature for knowledge distillation

  # TODO: Feature alignment - NOT IMPLEMENTED
  # feature_alignment: true
  # alignment_weight: 0.1        # Weight for feature alignment loss

  # TODO: Dynamic weighting - NOT IMPLEMENTED
  # adaptive_weighting: true     # Dynamically balance base/novel losses
  # weighting_strategy: "uncertainty"  # "uncertainty", "gradient", "performance"

  # Training parameters (WORKING)
  lr: 1e-3
  epochs: 100
```

### GFS Matching Networks 🚧 PARTIALLY IMPLEMENTED
```yaml
task:
  type: "GFS"
  name: "matching"

  # Class configuration (BASIC SUPPORT)
  base_classes: 8               # ✅ WORKS
  novel_classes: 2              # ✅ WORKS
  num_support: 5                # ✅ WORKS
  num_query: 15                 # ✅ WORKS

  # TODO: Matching configuration - PARTIALLY IMPLEMENTED
  # use_attention: true          # Basic attention may work
  # attention_type: "cosine"     # Cosine similarity should work

  # TODO: Base class memory - NOT IMPLEMENTED
  # base_memory_size: 1000       # Size of base class memory bank
  # memory_update_rate: 0.1      # Rate for updating memory

  # TODO: Matching strategies - BASIC IMPLEMENTATION
  base_matching_weight: 0.7     # ✅ MAY WORK
  novel_matching_weight: 0.3    # ✅ MAY WORK

  # Training parameters (WORKING)
  lr: 1e-3
  epochs: 100
```

## Key Parameters

### Class Configuration
- `base_classes`: Number of base classes with abundant training data
- `novel_classes`: Number of novel classes with few-shot examples
- `class_split_strategy`: How to split classes ("random", "semantic", "temporal")

### Episode Structure
- `num_support`: Support samples per novel class
- `num_query`: Query samples per novel class (and base classes)
- `base_samples_per_episode`: Number of base class samples per episode
- `balance_sampling`: Whether to balance base/novel sampling

### Loss Weighting
- `base_class_weight`: Weight for base class classification loss
- `novel_class_weight`: Weight for novel class classification loss
- `adaptive_weighting`: Enable dynamic loss weighting
- `weighting_strategy`: Strategy for adaptive weighting

### Knowledge Transfer
- `use_knowledge_distillation`: Enable teacher-student knowledge transfer
- `distillation_weight`: Weight for knowledge distillation loss
- `teacher_temperature`: Temperature for softmax in distillation

## Training Strategies

### 1. Joint Training
Train on both base and novel classes simultaneously:
```yaml
training_strategy: "joint"
base_class_weight: 1.0
novel_class_weight: 1.0
balance_sampling: true
```

### 2. Progressive Training
First train on base classes, then adapt to novel classes:
```yaml
training_strategy: "progressive"
base_training_epochs: 50     # Pre-train on base classes
novel_adaptation_epochs: 50  # Adapt to novel classes
freeze_backbone: false       # Whether to freeze during adaptation
```

### 3. Meta-Learning Approach
Use meta-learning for generalized few-shot:
```yaml
training_strategy: "meta"
meta_lr: 1e-3               # Meta-learning rate
inner_lr: 0.01              # Inner optimization learning rate
inner_steps: 5              # Gradient steps per episode
```

## Loss Functions

### Standard GFS Loss
Combination of base and novel class losses:
```python
total_loss = base_weight * base_loss + novel_weight * novel_loss
```

### Knowledge Distillation Loss
Transfer knowledge from base classes to novel classes:
```python
distill_loss = KL_divergence(student_logits / T, teacher_logits / T)
```

### Feature Alignment Loss
Align feature distributions between base and novel classes:
```python
alignment_loss = MMD(base_features, novel_features)
```

## Usage Examples

### Basic GFS Experiment
```bash
# 8 base classes + 2 novel classes with 5-shot learning
python main.py --config configs/demo/GFS/gfs_8base_2novel.yaml
```

### Incremental Learning Simulation
```bash
# Simulate adding new fault types to existing system
python main.py --config configs/demo/GFS/incremental_learning.yaml
```

### Cross-Dataset GFS
```bash
# Base classes from CWRU, novel classes from THU
python main.py --config configs/demo/GFS/cross_dataset_gfs.yaml
```

### Ablation Studies
```bash
# Compare different weighting strategies
python main.py --config configs/demo/GFS/ablation_weighting.yaml

# Test different numbers of novel classes
python main.py --config configs/demo/GFS/ablation_novel_classes.yaml
```

## Integration with Framework

### Task Registration
GFS tasks extend the episodic training framework to handle both base and novel classes.

### Model Compatibility
- **ISFM Models**: Excellent for leveraging pre-trained knowledge
- **Meta-Learning Architectures**: MAML, Prototypical Networks with GFS extensions
- **Attention Mechanisms**: For matching-based approaches
- **Memory Networks**: For maintaining base class representations

### Data Pipeline
- Automatic base/novel class splitting
- Balanced episode generation
- Support for temporal class emergence scenarios

## Evaluation Protocols

### GFS Benchmarks
- **Base Class Accuracy**: Performance on well-represented classes
- **Novel Class Accuracy**: Performance on few-shot classes
- **Harmonic Mean**: Balanced performance metric
- **Forgetting Rate**: Performance degradation on base classes

### Evaluation Metrics
```yaml
metrics:
  - "base_accuracy"          # Accuracy on base classes
  - "novel_accuracy"         # Accuracy on novel classes
  - "harmonic_mean"          # 2 * (base * novel) / (base + novel)
  - "overall_accuracy"       # Combined accuracy
  - "forgetting_rate"        # Degradation on base classes
```

## Advanced Features

### 1. TODO: Adaptive Class Balancing - NOT IMPLEMENTED
Dynamically adjust sampling based on performance:
```yaml
# TODO: adaptive_sampling: true - NOT IMPLEMENTED
# sampling_strategy: "performance"  # "performance", "uncertainty", "gradient"
# rebalance_frequency: 10           # Episodes between rebalancing
```

### 2. TODO: Continual Learning - NOT IMPLEMENTED
Handle sequential arrival of novel classes:
```yaml
# TODO: continual_learning: true - NOT IMPLEMENTED
# memory_replay: true              # Replay base class examples
# memory_size: 1000               # Size of replay memory
# anti_forgetting_weight: 0.1     # Weight for anti-forgetting loss
```

### 3. TODO: Multi-Task GFS - NOT IMPLEMENTED
Handle multiple tasks with shared base classes:
```yaml
# TODO: multi_task_gfs: true - NOT IMPLEMENTED
# shared_base_classes: [0, 1, 2, 3]    # Classes shared across tasks
# task_specific_classes: {
#   task_1: [4, 5],                     # Novel classes for task 1
#   task_2: [6, 7]                      # Novel classes for task 2
# }
```

## Industrial Applications

### Fault Diagnosis Evolution
- **Initial System**: Deploy with known fault types (base classes)
- **System Updates**: Add new fault patterns as they emerge (novel classes)
- **Knowledge Retention**: Maintain performance on original faults

### Equipment Lifecycle Management
- **Base Classes**: Common faults across equipment lifecycle
- **Novel Classes**: Equipment-specific or age-related faults
- **Transfer Learning**: Apply knowledge from similar equipment

### Multi-Site Deployment
- **Base Classes**: Universal fault types across all sites
- **Novel Classes**: Site-specific operational conditions or faults
- **Scalability**: Easy addition of new sites with minimal data

## Best Practices

### 1. Class Selection Strategy
- Choose base classes that represent fundamental fault physics
- Ensure novel classes are meaningfully different from base classes
- Consider temporal emergence patterns in industrial settings

### 2. Balance Optimization
- Monitor both base and novel class performance
- Use harmonic mean as primary optimization target
- Implement early stopping based on combined performance

### 3. Knowledge Transfer
- Pre-train on large datasets when possible
- Use feature alignment to improve transfer
- Consider domain-specific knowledge distillation

## Troubleshooting

### Common Issues
- **Base class forgetting**: Increase base class weight or use replay
- **Poor novel adaptation**: Increase novel class episodes or adjust learning rate
- **Imbalanced performance**: Use adaptive weighting strategies

### Debug Tips
- Visualize base vs. novel class feature distributions
- Monitor individual class performance trends
- Check gradient flows for both class types

## Research Context

### Key Innovations
- **Balanced Learning**: Joint optimization for base and novel classes
- **Knowledge Transfer**: Leveraging base class knowledge for novel adaptation
- **Industrial Relevance**: Practical deployment scenarios for fault diagnosis

### Comparison with Standard Few-Shot
- **Standard FS**: Only considers novel classes
- **GFS**: Maintains base class performance while learning novel classes
- **Industrial Value**: More realistic for deployed systems

## References

- [Task Factory Documentation](../CLAUDE.md)
- [Generalized Few-Shot Learning Paper](https://arxiv.org/abs/1707.09482)
- [Few-Shot Learning Module](../FS/README.md)
- [Configuration System](../../../configs/CLAUDE.md)
- [Model Factory](../../../model_factory/CLAUDE.md)
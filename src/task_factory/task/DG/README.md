# DG Task Module

## 🚧 实现状态 (Implementation Status)

### ✅ 已实现 (Fully Implemented)
- **基础分类包装器**: `classification.py` - 基于`Default_task`的21行基础实现
- **标准分类功能**: 基本交叉熵损失分类

### ❌ TODO: 待实现 (Not Yet Implemented)
- **领域自适应损失**: MMD, CORAL, DANN等所有领域对齐技术
- **对抗训练**: 梯度反转层和对抗域适应
- **正则化技术**: domain_penalty, adversarial, coral等正则化方法
- **域感知批归一化**: 分离的域特定批归一化
- **渐进式域适应**: 动态权重调整机制
- **域拆分策略**: 基于条件、负载、速度的自动域分割

> **注意**: 当前实现为最小包装器，以下文档描述的功能大部分为设计目标，需要逐步实现。

## Overview

The DG (Domain Generalization) task module implements domain generalization techniques for single-dataset experiments. Unlike CDDG which handles cross-dataset scenarios, DG focuses on improving model robustness within a single dataset by learning domain-invariant representations that generalize across different conditions, equipment states, or operational environments.

## Architecture

DG tasks focus on learning representations that are invariant to domain-specific variations while maintaining discriminative power for the target classification task. This is essential for industrial fault diagnosis where models must work across different operating conditions, loads, or equipment variations.

## Available Tasks

### 1. classification.py
**Standard domain generalization classification task**

- **Purpose**: Single-dataset classification with domain regularization
- **Use Case**: When you need to improve robustness within one dataset across different operational conditions
- **Features**:
  - Domain-invariant feature learning
  - Regularization techniques (domain penalty, adversarial training)
  - Cross-validation across different data splits
  - Support for domain adaptation losses

## Configuration Examples

### Basic DG Classification
```yaml
task:
  type: "DG"
  name: "classification"
  loss: "CE"                     # Cross-entropy loss
  # TODO: regularization: ["domain_penalty"]  # Domain regularization techniques - NOT IMPLEMENTED
  # TODO: domain_weight: 0.1             # Weight for domain regularization - NOT IMPLEMENTED

  # TODO: Domain configuration (optional) - NOT IMPLEMENTED
  # source_domains: [1, 2, 3]      # Source domain IDs within dataset
  # target_domain: 4               # Target domain ID for evaluation
```

### DG with Advanced Regularization
```yaml
task:
  type: "DG"
  name: "classification"
  loss: "CE"

  # TODO: Multiple regularization techniques - NOT IMPLEMENTED
  # regularization:
  #   - "domain_penalty"           # L2 penalty on domain-specific features
  #   - "adversarial"              # Adversarial domain adaptation
  #   - "coral"                    # CORAL domain alignment

  # TODO: Regularization weights - NOT IMPLEMENTED
  # domain_weight: 0.1
  # adversarial_weight: 0.05
  # coral_weight: 0.02

  # Training parameters (BASIC ONES WORK)
  lr: 1e-3
  epochs: 100
  batch_size: 64
```

## Key Parameters

### Domain Configuration
- `source_domains`: List of source domain IDs for training (optional)
- `target_domain`: Target domain ID for evaluation (optional)
- `domain_split_method`: Method for splitting domains ("condition", "load", "speed")

### Regularization
- `regularization`: List of regularization techniques to apply
- `domain_weight`: Weight for domain penalty regularization
- `adversarial_weight`: Weight for adversarial domain adaptation
- `coral_weight`: Weight for CORAL domain alignment

### Loss Functions
- `loss`: Primary classification loss ("CE", "focal", "label_smooth")
- `domain_loss`: Domain adaptation loss ("MMD", "CORAL", "adversarial")

## Regularization Techniques

### 1. TODO: Domain Penalty - NOT IMPLEMENTED
Applies L2 penalty to encourage domain-invariant features:
```yaml
# TODO: regularization: ["domain_penalty"] - NOT IMPLEMENTED
# domain_weight: 0.1
```

### 2. TODO: Adversarial Training - NOT IMPLEMENTED
Uses adversarial loss to confuse domain classifier:
```yaml
# TODO: regularization: ["adversarial"] - NOT IMPLEMENTED
# adversarial_weight: 0.05
# gradient_reversal_lambda: 1.0
```

### 3. TODO: CORAL (Correlation Alignment) - NOT IMPLEMENTED
Aligns feature distributions across domains:
```yaml
# TODO: regularization: ["coral"] - NOT IMPLEMENTED
# coral_weight: 0.02
```

### 4. TODO: MMD (Maximum Mean Discrepancy) - NOT IMPLEMENTED
Minimizes distribution difference between domains:
```yaml
# TODO: regularization: ["mmd"] - NOT IMPLEMENTED
# mmd_weight: 0.03
# mmd_kernel: "rbf"              # Kernel type for MMD
```

## Usage Examples

### Basic Single-Dataset DG
```bash
# Train with domain generalization on CWRU dataset
python main.py --config configs/demo/Single_DG/CWRU_DG.yaml
```

### Cross-Validation DG
```bash
# Domain generalization with cross-validation
python main.py --config configs/demo/Single_DG/CWRU_DG_CV.yaml
```

### Advanced Regularization
```bash
# Multiple regularization techniques
python main.py --config configs/demo/Single_DG/CWRU_DG_advanced.yaml
```

## Integration with Framework

### Task Registration
The DG classification task inherits from `Default_task` and adds domain-specific regularization capabilities.

### Model Compatibility
- **All Backbone Networks**: Compatible with CNN, Transformer, and hybrid architectures
- **Task Heads**: Works with standard classification heads
- **ISFM Models**: Full support for foundation model fine-tuning

### Data Pipeline
- Supports automatic domain splitting based on metadata
- Compatible with all 30+ datasets in PHM-Vibench
- Handles different domain definitions (by condition, load, speed, etc.)

## Domain Splitting Strategies

### 1. TODO: Condition-Based - NOT IMPLEMENTED
Split by operating conditions (normal, fault types):
```yaml
# TODO: domain_split_method: "condition" - NOT IMPLEMENTED
# source_conditions: ["normal", "inner_race", "outer_race"]
# target_condition: "ball_fault"
```

### 2. TODO: Load-Based - NOT IMPLEMENTED
Split by mechanical load levels:
```yaml
# TODO: domain_split_method: "load" - NOT IMPLEMENTED
# source_loads: [0, 1, 2]        # 0HP, 1HP, 2HP
# target_load: 3                 # 3HP
```

### 3. TODO: Speed-Based - NOT IMPLEMENTED
Split by rotational speeds:
```yaml
# TODO: domain_split_method: "speed" - NOT IMPLEMENTED
# source_speeds: [1797, 1772, 1750]  # RPM values
# target_speed: 1730
```

### 4. TODO: Custom Domain Definition - NOT IMPLEMENTED
Define custom domain splits:
```yaml
# TODO: domain_split_method: "custom" - NOT IMPLEMENTED
# domain_mapping:
#   domain_0: [0, 1, 2]          # File indices for domain 0
#   domain_1: [3, 4, 5]          # File indices for domain 1
#   domain_2: [6, 7, 8]          # File indices for domain 2
```

## Evaluation Metrics

### Standard Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- Per-class performance metrics
- Confusion matrices

### Domain-Specific Metrics
- Source domain accuracy
- Target domain accuracy
- Domain gap (performance difference)
- Domain adaptation effectiveness

### Robustness Metrics
- Cross-domain consistency
- Worst-case domain performance
- Average domain performance

## Advanced Features

### 1. TODO: Gradient Reversal Layer - NOT IMPLEMENTED
For adversarial domain adaptation:
```python
# TODO: Automatically included when using adversarial regularization - NOT IMPLEMENTED
# regularization: ["adversarial"]
# gradient_reversal_lambda: 1.0
```

### 2. TODO: Domain-Adaptive Batch Normalization - NOT IMPLEMENTED
Separate batch normalization for different domains:
```yaml
# TODO: use_domain_bn: true - NOT IMPLEMENTED
# num_domains: 4
```

### 3. TODO: Progressive Domain Adaptation - NOT IMPLEMENTED
Gradually increase domain adaptation weight:
```yaml
# TODO: progressive_adaptation: true - NOT IMPLEMENTED
# adaptation_schedule: "linear"    # or "exponential"
# max_adaptation_weight: 0.1
```

## Research Applications

### Industrial Fault Diagnosis
- Robust fault detection across different operating conditions
- Generalization to unseen load conditions
- Cross-equipment type adaptation within same manufacturer

### Predictive Maintenance
- Condition monitoring across different operational phases
- Degradation pattern generalization
- Cross-seasonal performance maintenance

## Best Practices

### 1. Domain Definition
- Choose meaningful domain splits based on industrial knowledge
- Ensure sufficient samples per domain
- Validate domain relevance through expert knowledge

### 2. Regularization Selection
- Start with domain penalty for simple cases
- Add adversarial training for complex domain gaps
- Use CORAL for distribution alignment

### 3. Hyperparameter Tuning
- Balance classification and domain losses carefully
- Use validation set from target domain for early stopping
- Monitor both source and target domain performance

## Troubleshooting

### Common Issues
- **Domain collapse**: Reduce domain adaptation weight
- **Poor target performance**: Increase domain regularization
- **Training instability**: Lower learning rate or use gradient clipping

### Debug Tips
- Visualize feature distributions across domains
- Monitor domain classification accuracy
- Check gradient flows in adversarial training

## References

- [Task Factory Documentation](../CLAUDE.md)
- [Configuration System](../../../configs/CLAUDE.md)
- [Model Factory](../../../model_factory/CLAUDE.md)
- [Domain Adaptation Literature](https://domain-adaptation.org/)
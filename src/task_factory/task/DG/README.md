# DG Task Module

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
  regularization: ["domain_penalty"]  # Domain regularization techniques
  domain_weight: 0.1             # Weight for domain regularization

  # Domain configuration (optional)
  source_domains: [1, 2, 3]      # Source domain IDs within dataset
  target_domain: 4               # Target domain ID for evaluation
```

### DG with Advanced Regularization
```yaml
task:
  type: "DG"
  name: "classification"
  loss: "CE"

  # Multiple regularization techniques
  regularization:
    - "domain_penalty"           # L2 penalty on domain-specific features
    - "adversarial"              # Adversarial domain adaptation
    - "coral"                    # CORAL domain alignment

  # Regularization weights
  domain_weight: 0.1
  adversarial_weight: 0.05
  coral_weight: 0.02

  # Training parameters
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

### 1. Domain Penalty
Applies L2 penalty to encourage domain-invariant features:
```yaml
regularization: ["domain_penalty"]
domain_weight: 0.1
```

### 2. Adversarial Training
Uses adversarial loss to confuse domain classifier:
```yaml
regularization: ["adversarial"]
adversarial_weight: 0.05
gradient_reversal_lambda: 1.0
```

### 3. CORAL (Correlation Alignment)
Aligns feature distributions across domains:
```yaml
regularization: ["coral"]
coral_weight: 0.02
```

### 4. MMD (Maximum Mean Discrepancy)
Minimizes distribution difference between domains:
```yaml
regularization: ["mmd"]
mmd_weight: 0.03
mmd_kernel: "rbf"              # Kernel type for MMD
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

### 1. Condition-Based
Split by operating conditions (normal, fault types):
```yaml
domain_split_method: "condition"
source_conditions: ["normal", "inner_race", "outer_race"]
target_condition: "ball_fault"
```

### 2. Load-Based
Split by mechanical load levels:
```yaml
domain_split_method: "load"
source_loads: [0, 1, 2]        # 0HP, 1HP, 2HP
target_load: 3                 # 3HP
```

### 3. Speed-Based
Split by rotational speeds:
```yaml
domain_split_method: "speed"
source_speeds: [1797, 1772, 1750]  # RPM values
target_speed: 1730
```

### 4. Custom Domain Definition
Define custom domain splits:
```yaml
domain_split_method: "custom"
domain_mapping:
  domain_0: [0, 1, 2]          # File indices for domain 0
  domain_1: [3, 4, 5]          # File indices for domain 1
  domain_2: [6, 7, 8]          # File indices for domain 2
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

### 1. Gradient Reversal Layer
For adversarial domain adaptation:
```python
# Automatically included when using adversarial regularization
regularization: ["adversarial"]
gradient_reversal_lambda: 1.0
```

### 2. Domain-Adaptive Batch Normalization
Separate batch normalization for different domains:
```yaml
use_domain_bn: true
num_domains: 4
```

### 3. Progressive Domain Adaptation
Gradually increase domain adaptation weight:
```yaml
progressive_adaptation: true
adaptation_schedule: "linear"    # or "exponential"
max_adaptation_weight: 0.1
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
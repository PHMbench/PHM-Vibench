# CDDG Task Module

## Overview

The CDDG (Cross-Dataset Domain Generalization) task module implements tasks designed for training models that can generalize across different datasets and domains. This is critical for industrial fault diagnosis where models need to work reliably across different equipment types, operating conditions, and measurement systems.

## Architecture

The CDDG module focuses on learning domain-invariant representations that maintain performance when transferring from source domains (training datasets) to target domains (testing datasets).

## Available Tasks

### 1. classification.py
**Standard cross-dataset classification task**

- **Purpose**: Basic classification with cross-dataset domain adaptation
- **Use Case**: When you need simple domain transfer without specialized techniques
- **Features**:
  - Multi-source domain training
  - Domain adaptation losses (MMD, CORAL, etc.)
  - Cross-entropy loss with domain penalty

### 2. hse_contrastive.py ⭐ **Innovation Task**
**HSE Prompt-guided Contrastive Learning for Cross-Dataset Domain Generalization**

- **Purpose**: Novel contrastive learning approach with system metadata prompts
- **Innovation**: First work combining system prompts with contrastive learning for industrial fault diagnosis
- **Target**: ICML/NeurIPS 2025 submission
- **Features**:
  - Prompt-guided contrastive learning with system-aware sampling
  - Two-stage training support (pretrain/finetune)
  - Integration with 6 SOTA contrastive losses
  - System-invariant representation learning
  - Cross-system domain generalization

## Configuration Examples

### Standard CDDG Classification
```yaml
task:
  type: "CDDG"
  name: "classification"
  source_domain_id: [1, 5, 6]    # Training domains
  target_domain_id: 19           # Test domain
  loss: "CE"                     # Cross-entropy loss
  domain_adaptation_loss: "MMD"  # Domain adaptation
  domain_weight: 0.1             # Weight for domain loss
```

### HSE Contrastive Learning
```yaml
task:
  type: "CDDG"
  name: "hse_contrastive"
  source_domain_id: [1, 5, 6]
  target_domain_id: 19

  # Contrastive learning settings
  contrastive_loss: "SimCLR"     # SimCLR, SwAV, MoCo, etc.
  temperature: 0.07
  contrastive_weight: 1.0

  # Prompt settings
  use_system_prompts: true
  prompt_dim: 256
  prompt_fusion: "attention"

  # Two-stage training
  training_stage: "pretrain"     # "pretrain" or "finetune"
  freeze_prompts: false          # Set to true in finetune stage
```

## Key Parameters

### Domain Configuration
- `source_domain_id`: List of source domain IDs for training
- `target_domain_id`: Target domain ID for evaluation
- `domain_adaptation_loss`: Type of domain adaptation loss ("MMD", "CORAL", "DANN")
- `domain_weight`: Weight for domain adaptation loss

### Contrastive Learning (HSE)
- `contrastive_loss`: Contrastive loss type ("SimCLR", "SwAV", "MoCo", "VICReg", "BarlowTwins", "BYOL")
- `temperature`: Temperature parameter for contrastive loss
- `contrastive_weight`: Weight for contrastive loss term
- `num_negatives`: Number of negative samples per positive

### Prompt System (HSE)
- `use_system_prompts`: Enable system metadata prompts
- `prompt_dim`: Dimension of prompt embeddings
- `prompt_fusion`: Method for fusing prompts ("concat", "attention", "add")
- `freeze_prompts`: Whether to freeze prompt parameters

## Usage Examples

### Basic CDDG Experiment
```bash
# Train on CWRU, test on THU
python main.py --config configs/demo/Multiple_DG/CWRU_THU_basic.yaml
```

### HSE Contrastive Learning Pipeline
```bash
# Stage 1: Pretraining with contrastive learning
python main.py --config configs/hse/pretrain_contrastive.yaml

# Stage 2: Fine-tuning for classification
python main.py --config configs/hse/finetune_classification.yaml
```

### Multi-Domain Training
```bash
# Train on multiple sources, test on single target
python main.py --config configs/demo/Multiple_DG/all_to_THU.yaml
```

## Integration with Framework

### Task Registration
Tasks are automatically registered when imported. The HSE contrastive task extends the `Default_task` class with specialized contrastive learning capabilities.

### Model Compatibility
- **ISFM Models**: Full support with prompt integration
- **Backbone Networks**: Compatible with all backbone architectures
- **Task Heads**: Works with classification and multi-task heads

### Data Pipeline
- Supports all 30+ datasets in PHM-Vibench
- Automatic domain splitting based on metadata
- System-aware sampling for contrastive learning

## Advanced Features

### System-Aware Sampling (HSE)
The HSE contrastive task implements intelligent sampling strategies:
- **Positive pairs**: Same fault type, different systems (cross-system invariance)
- **Negative pairs**: Different fault types with system awareness
- **Hard negatives**: Similar faults from different systems

### Two-Stage Training
1. **Pretraining Stage**: Learn system-invariant representations via contrastive learning
2. **Fine-tuning Stage**: Adapt to specific classification tasks with frozen prompts

### Metrics and Evaluation
- Standard classification metrics (accuracy, F1-score, precision, recall)
- Domain-specific metrics (per-domain accuracy, domain gap)
- Contrastive learning metrics (alignment, uniformity)

## Research Context

The HSE contrastive learning task implements our core research contribution:
- **Innovation**: Prompt-guided contrastive learning for industrial domains
- **Novelty**: First to combine system metadata with contrastive learning
- **Impact**: Addresses critical cross-system generalization challenge
- **Validation**: Comprehensive evaluation across 30+ industrial datasets

## References

- [Task Factory Documentation](../CLAUDE.md)
- [HSE Innovation Specification](.claude/specs/hse-complete-publication-pipeline/)
- [Configuration System](../../../configs/CLAUDE.md)
- [Model Factory](../../../model_factory/CLAUDE.md)
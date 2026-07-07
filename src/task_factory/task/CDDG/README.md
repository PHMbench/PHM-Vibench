# CDDG Task Module

## 🚧 实现状态 (Implementation Status)

### ✅ 已实现 (Fully Implemented)
- **HSE对比学习**: `hse_contrastive.py` - 1037行复杂实现，支持阶段感知训练
- **对比策略系统**: `contrastive_strategies.py` - 1225行完整的对比学习框架
- **多阶段训练**: 两阶段预训练+微调，真正的流分离架构
- **系统提示融合**: attention/gate/concat等融合机制
- **双视图对比学习**: SimCSE风格的数据增强策略

### 🚧 部分实现 (Partially Implemented)
- **标准分类包装器**: `classification.py` - 21行基础Default_task包装器
- **跨域基本功能**: 基础的source_domain_id/target_domain_id支持

### ❌ TODO: 待实现 (Not Yet Implemented)
- **其他对比损失**: SimCLR, SwAV, MoCo, VICReg, BarlowTwins, BYOL等6种SOTA方法
- **域自适应损失**: MMD, CORAL, DANN等域对齐技术
- **系统感知采样**: 跨系统正负样本对的智能采样策略
- **多源域自适应**: 真正的多源域训练和域间对齐

> **注意**: HSE对比学习是核心实现，其他功能为设计目标或基础包装。

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
  source_domain_id: [1, 5, 6]    # Training domains (BASIC SUPPORT)
  target_domain_id: 19           # Test domain (BASIC SUPPORT)
  loss: "CE"                     # Cross-entropy loss (WORKS)
  # TODO: domain_adaptation_loss: "MMD"  # Domain adaptation - NOT IMPLEMENTED
  # TODO: domain_weight: 0.1             # Weight for domain loss - NOT IMPLEMENTED
```

### HSE Contrastive Learning ✅ IMPLEMENTED
```yaml
task:
  type: "CDDG"
  name: "hse_contrastive"
  source_domain_id: [1, 5, 6]    # WORKS
  target_domain_id: 19           # WORKS

  # Contrastive learning settings (WORKING)
  contrast_weight: 1.0           # ✅ WORKS (renamed from contrastive_weight)
  classification_weight: 0.1    # ✅ WORKS
  temperature: 0.07              # ✅ WORKS

  # Prompt settings (WORKING)
  prompt_fusion: "attention"     # ✅ WORKS (attention/gate/add/none)

  # Two-stage training (WORKING)
  training_stage: "pretrain"     # ✅ WORKS ("pretrain" or "finetune")
  # freeze_prompts: false          # Controlled by stage settings
```

## Key Parameters

### Domain Configuration
- `source_domain_id`: ✅ List of source domain IDs for training (BASIC SUPPORT)
- `target_domain_id`: ✅ Target domain ID for evaluation (BASIC SUPPORT)
- `domain_adaptation_loss`: ❌ TODO: Type of domain adaptation loss ("MMD", "CORAL", "DANN") - NOT IMPLEMENTED
- `domain_weight`: ❌ TODO: Weight for domain adaptation loss - NOT IMPLEMENTED

### Contrastive Learning (HSE) ✅ IMPLEMENTED
- `contrast_weight`: ✅ Weight for contrastive loss term (WORKS)
- `classification_weight`: ✅ Weight for classification loss term (WORKS)
- `temperature`: ✅ Temperature parameter for contrastive loss (WORKS)
- `num_negatives`: ❌ TODO: Number of negative samples per positive - NOT NEEDED (auto-handled)
- `contrast_loss`: ❌ TODO: Other loss types ("SimCLR", "SwAV", "MoCo", "VICReg", "BarlowTwins", "BYOL") - NOT IMPLEMENTED

### Prompt System (HSE) ✅ IMPLEMENTED
- `prompt_fusion`: ✅ Method for fusing prompts ("attention", "gate", "add", "none") (WORKS)
- `training_stage`: ✅ Stage for training behavior ("pretrain", "finetune") (WORKS)
- `prompt_dim`: ❌ TODO: Dimension of prompt embeddings - AUTO-CONFIGURED
- `freeze_prompts`: ❌ TODO: Whether to freeze prompt parameters - CONTROLLED BY STAGE

## Usage Examples (v0.1.0 demos)

### Basic CDDG Classification (demo)
```bash
# Multi-system CDDG classification demo
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

### HSE Contrastive Learning (demo, single-stage view)
```bash
# HSE-style pretraining demo (single-stage view)
python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

### Two-Stage HSE Pretrain + CDDG (paper submodule, TODO)
Multi-stage research configs/scripts are planned to live in the paper submodule to keep the main repo workflow stable.

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

### System-Aware Sampling (HSE) ❌ TODO: NOT IMPLEMENTED
The HSE contrastive task plans to implement intelligent sampling strategies:
- **Positive pairs**: Same fault type, different systems (cross-system invariance)
- **Negative pairs**: Different fault types with system awareness
- **Hard negatives**: Similar faults from different systems

> **当前实现**: 使用基础的双视图对比学习，未实现高级系统感知采样策略。

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

- [Task Factory Documentation](../../CLAUDE.md)
- HSE innovation notes: summarized above; historical agent specs are intentionally not tracked.
- [Configuration System](../../../configs/CLAUDE.md)
- [Model Factory](../../../model_factory/CLAUDE.md)

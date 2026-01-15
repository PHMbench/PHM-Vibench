# ISFM_Prompt - CLAUDE.md

This module provides architecture guidance for the ISFM Prompt-tuning framework in PHM-Vibench. For configuration details, see [@README.md].

## Architecture Overview

ISFM_Prompt extends the ISFM framework with prompt-based adaptation for efficient fine-tuning:

```
Pre-trained ISFM
     ↓
┌─────────────────────────────────────┐
│  Prompt Injection                     │
│  - Learnable prompt vectors           │
│  - Domain-specific prompts            │
│  - Task-specific prompts              │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Feature Extraction (frozen backbone) │
│  - Optional: freeze ISFM backbone     │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Lightweight Adaptation Head         │
│  - Prompt-conditioned prediction      │
└─────────────────────────────────────┘
     ↓
Output
```

## Design Philosophy

### Efficient Adaptation
Instead of fine-tuning entire model:
- **Prompt tuning**: Learn small prompt vectors
- **Frozen backbone**: Keep pre-trained features
- **Parameter efficient**: Only train prompts + head

### Few-Shot Learning
Prompts enable rapid adaptation with limited data:
- Domain prompts: Adapt to new operating conditions
- Task prompts: Switch between classification/prediction

## Configuration Pattern

```yaml
model:
  type: "ISFM_Prompt"
  name: "ISFM_Prompt"

  # Base ISFM (pre-trained)
  base_model: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_08_PatchTST"
  task_head: "H_01_Linear_cla"

  # Prompt configuration
  prompt_length: 10       # Number of prompt tokens
  prompt_dropout: 0.1

  # Training
  freeze_backbone: true   # Keep ISFM frozen
  learning_rate: 1e-4     # Lower LR for prompt tuning
```

## Key Benefits

1. **Parameter Efficient**: Train <1% of parameters
2. **Fast Adaptation**: Few epochs needed
3. **Multi-Domain**: Separate prompts per domain
4. **Knowledge Retention**: Preserve pre-trained features

## Related Documentation

- [@README.md] - Configuration and Usage Guide
- [@README_Simplified.md](README_Simplified.md) - Simplified Guide
- [@../ISFM/README.md] - Base ISFM Framework

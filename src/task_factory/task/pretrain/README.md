# Pretrain Tasks

`pretrain` contains task modules used before downstream classification or
few-shot stages. The current maintained public demos use `hse_contrastive`.

## Current Surface

| Task type | Task name | Module | Status |
|---|---|---|---|
| `pretrain` | `classification` | `classification.py` | Registered implementation |
| `pretrain` | `hse_contrastive` | `hse_contrastive.py` | Maintained demo path |
| `pretrain` | `masked_reconstruction` | `masked_reconstruction.py` | Registered implementation |
| `pretrain` | `prediction` | `prediction.py` | Registered implementation |
| `pretrain` | `classification_prediction` | `classification_prediction.py` | Registered implementation |

Maintained demo paths:

- `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`

The registry source of truth is `src/task_factory/task_registry.csv`.

## Configuration Notes

Inspect a maintained pretrain demo before copying it:

```bash
python -m scripts.config_inspect \
  --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1
```

Base fields live in `configs/base/task/pretrain.yaml`; demo files may override
`task.name`, contrastive weights, augmentation settings, and stage-specific
fields.

## Boundaries

- This README does not promote every registered pretrain module to
  release-supported status.
- It does not claim validated foundation-model transfer, advanced masking
  strategies, multi-scale pretraining, domain-adaptive pretraining, or
  progressive training.
- Add new pretrain behavior by updating code, `src/task_factory/task_registry.csv`,
  config validation, docs, and a focused smoke path together.

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
| `pretrain` | `ppt_time_order` | `ppt_time_order.py` | Experimental, time-order-only |
| `pretrain` | `ppt_order` | `ppt_order.py` | Experimental time/channel SSL or supervised |

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

`ppt_time_order` is a clean-room implementation of the univariate time-order
subset described in the ICLR 2025 PPT paper. It requires `M_01_ISFM`,
`E_03_Patch`, `B_08_PatchTST`, one input channel, non-overlapping patches, and a
model-level `window_size` matching `data.window_size`. It does not implement the
paper's multivariate channel-order losses and is not release-supported. See
`configs/experiments/pretraining/ppt_time_order_univariate.yaml`.

`ppt_order` adds a channel-independent `[B,C,P,D]` contract. It supports
`task.ppt.mode: ssl|supervised`, time/channel axes, fixed or homoscedastic
uncertainty weighting, and omits channel-order terms unless at least three fixed
channels are configured. The existing `ppt_time_order` behavior is unchanged.
See `configs/experiments/pretraining/ppt_order_multichannel.yaml`.

## Boundaries

- This README does not promote every registered pretrain module to
  release-supported status.
- It does not claim validated foundation-model transfer, advanced masking
  strategies, multi-scale pretraining, domain-adaptive pretraining, or
  progressive training.
- Add new pretrain behavior by updating code, `src/task_factory/task_registry.csv`,
  config validation, docs, and a focused smoke path together.

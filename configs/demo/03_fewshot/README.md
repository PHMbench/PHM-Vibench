# Demo: Few-shot (`demo_03_fewshot`)

## Purpose

Single-system few-shot-style configuration using the FS task base.

## Minimal Run

```bash
python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Common Pitfalls

1) Confusing FS and GFS (different samplers/episodes semantics).
2) Output directory may inherit from base environment if not overridden.


# Demo: Cross-system Few-shot (`demo_04_cross_system_fewshot`)

## Purpose

Cross-system few-shot-style demo (task is still `FS` in v0.1.0).

## Minimal Run

```bash
python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Common Pitfalls

1) Expecting full GFS semantics while the task type is `FS`.
2) Using `task.target_system_id` values not present in metadata.


# Demo: CDDG (`demo_02_cross_system`)

## Purpose

Cross-system CDDG classification (multi-system requires editing `task.target_system_id`).

## Minimal Run

```bash
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Common Pitfalls

1) Expecting “multi-system” behavior while `task.target_system_id` only contains one id.
2) Using system ids not present in metadata.


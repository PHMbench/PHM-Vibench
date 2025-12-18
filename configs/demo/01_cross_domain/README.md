# Demo: DG (Domain Split) (`demo_01_cross_domain`)

## Purpose

Single-dataset DG via domain split (`source_domain_id` → `target_domain_id`).

## Minimal Run

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Key Fields

- `task.target_system_id`: system selection (metadata `Dataset_id`)
- `task.source_domain_id` / `task.target_domain_id`: domain split ids

## Common Pitfalls

1) Hard-coding dataset name (“CWRU→Ottawa”) without verifying metadata `Dataset_id` mapping.
2) Domain ids not present in your metadata.


# Demo: Pretrain for CDDG View (`demo_06_pretrain_cddg`)

## Purpose

Single-stage view of HSE contrastive pretraining under cross-system data settings.

## Minimal Run

```bash
python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Common Pitfalls

1) Expecting multi-system behavior while `task.target_system_id` is a single id.
2) Running on CPU without overriding `trainer.device=cpu`.


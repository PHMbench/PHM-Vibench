# Demo: Pretrain (HSE) (`demo_05_pretrain_fewshot`)

## Purpose

Pipeline_02 pretrain/few-shot pipeline: current demo runs in single-stage mode (no `stages:`).

## Minimal Run

```bash
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Common Pitfalls

1) Assuming it is a true two-stage run without adding `stages`.
2) Confusing paper-only pipeline03 scripts with this repo demo.


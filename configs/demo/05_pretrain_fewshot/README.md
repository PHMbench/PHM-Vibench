# Demo: HSE Pretraining Smoke (`demo_05_pretrain_fewshot`)

## Purpose

Run the maintained single-stage `pretrain/hse_contrastive` path through Pipeline 02.
The current YAML has no `stages:` block, so it does not perform downstream few-shot
adaptation or a second-stage evaluation.

The filename `pretrain_hse_then_fewshot.yaml` is retained temporarily as a compatibility
path; the effective experiment is single-stage HSE contrastive pretraining.

## Minimal Run

```bash
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Boundary

```text
current demo = one pretraining stage
future two-stage path = pretraining checkpoint + explicit adaptation task + independent evaluation
```

No two-stage or few-shot claim should be derived from this smoke run.

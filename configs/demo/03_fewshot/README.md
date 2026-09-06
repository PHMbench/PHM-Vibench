# Demo: Supervised Window Holdout (`demo_03_fewshot`)

## Purpose

Run ordinary supervised CWRU classification through the existing `FS/classification`
compatibility path. The data path uses held-out windows, while the task optimizes
cross-entropy on an ordinary `x/y/file_id` batch.

The compatibility filename `cwru_protonet.yaml` does not describe the current
algorithm. This run has no support/query batch, prototype computation, episode-local
label mapping, or query-only loss.

## Minimal Run

```bash
python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## What This Run Does Not Establish

```text
few-shot adaptation
ProtoNet
N-way K-shot episodes
episodic evaluation
```

A true ProtoNet path requires a separate dataset/sampler/task vertical slice.

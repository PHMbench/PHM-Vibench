# Demo: Single-system Held-out-domain ERM (`demo_02_cross_system`)

## Purpose

Run ordinary cross-entropy classification for one known `Dataset_id`, using source
and target domain selection from the existing CDDG compatibility path. The model uses
a system-specific classification head, so this demo does not evaluate an unknown new
system.

The filename `multi_system_cddg.yaml` is retained temporarily as a compatibility path;
its current configuration contains `task.target_system_id: [1]`.

## Minimal Run

```bash
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## What This Run Establishes

```text
known system identity
+ held-out domains
+ shared ISFM/HSE representation
+ ordinary CE classification
```

It does not establish unknown-system generalization or a multi-system benchmark.

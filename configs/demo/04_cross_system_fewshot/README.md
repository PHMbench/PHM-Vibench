# Demo: Hierarchically Sampled Classification (`demo_04_cross_system_fewshot`)

## Purpose

Exercise the existing GFS sampler across selected systems, domains and labels, then
train the standard `GFS/classification` Lightning task with cross-entropy.

The sampler emits a single ordinary batch. The current path does not mark support and
query samples, define base and novel classes, perform episode-specific adaptation, or
report generalized few-shot metrics.

The filename `gfs_dlinear.yaml` is retained temporarily as a compatibility path.

## Minimal Run

```bash
python main.py --config configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Resolved Contract

```text
Pipeline: Pipeline_01_Fault_Diagnosis
Task:     GFS/classification (ordinary CE)
Sampler:  hierarchical system/domain/label sampling
Model:    ISFM/M_01_ISFM
Embedding:E_01_HSE
Backbone: B_04_Dlinear
Trainer:  Default_trainer
```

This is execution-smoke evidence for a sampled classification path, not a generalized
few-shot benchmark.

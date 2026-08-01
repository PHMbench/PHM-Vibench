# Demo: Cross-system Generalized Few-shot (`demo_04_cross_system_fewshot`)

## Purpose

Exercise the maintained `GFS/classification` path across two systems with the standard
`ISFM/M_01_ISFM` model, `E_01_HSE` embedding, and `B_04_Dlinear` backbone. The filename
and documentation intentionally name the resolved model rather than implying TSPN.

## Minimal Run

```bash
python main.py --config configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Resolved Contract

```text
Pipeline: Pipeline_01_Fault_Diagnosis
Task:     GFS/classification
Model:    ISFM/M_01_ISFM
Embedding:E_01_HSE
Backbone: B_04_Dlinear
Trainer:  Default_trainer
```

## Common Pitfalls

1. Treating the filename or directory as evidence for a different model family.
2. Using `task.target_system_id` values absent from the supplied metadata.
3. Treating one-epoch `sanity_ok` smoke evidence as a benchmark result.

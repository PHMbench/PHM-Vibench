# UXFD Demos (`configs/demo/uxfd/`)

Maintained runnable examples for the UXFD merge **core model contract**:
- one core model: `model.type: X_model` + `model.name: TSPN_UXFD`
- paper differences expressed via `model.uxfd.*` and `trainer.extensions.*`

## Demos

1) Minimal (no UXFD modules):
```bash
python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml --override trainer.num_epochs=1
```

2) UXFD-enabled (SP2D + fusion + predictions):
```bash
python main.py --config configs/demo/uxfd/10_smoke_tspn_uxfd_sp2d.yaml --override trainer.num_epochs=1
```

3) UXFD-enabled (SP2D + fusion + fuzzy + operator-attention + logic):
```bash
python main.py --config configs/demo/uxfd/20_smoke_tspn_uxfd_full.yaml --override trainer.num_epochs=1
```

## Notes

- These demos use repo-shipped dummy data (`data/metadata_dummy.csv`) and run on CPU.
- Expected contract: each run emits `<run_dir>/config_snapshot.yaml` and `<run_dir>/artifacts/manifest.json`.
- Confusion matrix plotting requires predictions (`trainer.extensions.predictions.enable=true`).

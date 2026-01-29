# NSN Demos (`configs/demo/nsn/`)

Maintained runnable examples for the **NSN (no-presets)** configuration surface.

Notes:
- Model instantiation stays config-first: `model.type: X_model` + `model.name: NSN`
- NSN is a thin wrapper over `TSPN_UXFD` that maps optional flat-ish knobs into existing `model.uxfd.*`

## Demos

1) Minimal (no extra modules):
```bash
python main.py --config configs/demo/nsn/00_smoke_nsn_min.yaml --override trainer.num_epochs=1
```

2) SP2D-enabled (STFT branch + fusion + predictions):
```bash
python main.py --config configs/demo/nsn/10_smoke_nsn_sp2d.yaml --override trainer.num_epochs=1
```

3) Full (SP2D + fusion + fuzzy + operator-attention + logic):
```bash
python main.py --config configs/demo/nsn/20_smoke_nsn_full.yaml --override trainer.num_epochs=1
```


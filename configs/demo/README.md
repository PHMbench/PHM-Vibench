# Demo Configs (`configs/demo/`)

Demo configs are runnable examples built from `configs/base/*` + small overrides.

The demo list is indexed in `configs/config_registry.csv` and rendered in `docs/CONFIG_ATLAS.md`.

## Fastest Start (No External Data)

- Registry id: `demo_00_smoke_dummy_dg`
- Config: `configs/demo/00_smoke/dummy_dg.yaml`
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## Demo Index

- `configs/demo/00_smoke/README.md`
- `configs/demo/01_cross_domain/README.md`
- `configs/demo/02_cross_system/README.md`
- `configs/demo/03_fewshot/README.md`
- `configs/demo/04_cross_system_fewshot/README.md`
- `configs/demo/05_pretrain_fewshot/README.md`
- `configs/demo/06_pretrain_cddg/README.md`


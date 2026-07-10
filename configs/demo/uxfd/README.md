# UXFD Demo (`configs/demo/uxfd/`)

This directory contains maintained, repository-local smoke coverage for the `TSPN_UXFD` core contract.

## Minimal offline smoke

```bash
python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml \
  --override trainer.num_epochs=1
```

The demo uses the repository-shipped dummy metadata/data path, runs on CPU, and keeps all optional UXFD modules disabled. Its purpose is to verify the stable model entrypoint and configuration wiring before promoting more complex compositions.

## Acceptance gates

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml \
  --dump targets --format yaml
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml \
  --override trainer.num_epochs=1
```

## Deferred variants

SP2D/fusion, fuzzy, operator-attention, and logic composition demos are intentionally deferred to a separate follow-up PR. Unit-level assembly coverage does not by itself prove that every end-to-end demo configuration is ready for the maintained surface.

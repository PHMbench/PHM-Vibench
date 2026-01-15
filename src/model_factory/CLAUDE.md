# Model Factory - CLAUDE.md

This file captures the design intent and change strategy for `src/model_factory/`.
The canonical user-facing guide (how to configure models, registry SSOT, directory layout) is [@README.md].

## Invariants (don’t break)

- Model construction is config-driven via `model.type` + `model.name`.
- The supported set is documented in `src/model_factory/model_registry.csv` (treat it as SSOT).
- ISFM-style models use registry-addressed subcomponents (e.g. `embedding` / `backbone` / `task_head`) with stable IDs
  like `E_01_*`, `B_04_*`, `H_01_*`.
- Avoid hard-coding model imports in pipelines; keep wiring inside the factory and registries.

## Extension Points

- New model family/type: add a subdir under `src/model_factory/` and update factory resolution.
- New model implementation: add under `src/model_factory/<TYPE>/` and expose `Model(args_model, metadata)`.
- Registry: add a row to `src/model_factory/model_registry.csv`.
- Type-specific docs: update the relevant `src/model_factory/<TYPE>/README.md` (if present).

## Safe Change Rules

- Prefer additive changes (new IDs / new registry rows) over renaming existing IDs.
- If you change config keys consumed by a model, update at least one demo in `configs/demo/` and validate with:
  `python -m scripts.config_inspect --config <yaml>`.

## Validation Gate (minimal)

```bash
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --dump targets
python -m scripts.validate_configs
python -m pytest test/
```

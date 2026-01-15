# Data Factory - CLAUDE.md

This file captures the design intent and change strategy for `src/data_factory/`.
The canonical user-facing guide (layout, config keys, smoke demo behavior) is [@README.md].

## Invariants (don’t break)

- Entry point: `src/data_factory.build_data(args_data, args_task)`.
- Factory selection: `data.factory_name` (default is `"default"`).
- Readers + dataset wrappers should keep the “sample identity” joinable through metadata (`Id` / `file_id`).
- Batches should be dict-like for task modules (avoid tuple-unpacking assumptions).

## Extension Points (where to add things)

- New dataset reader: `src/data_factory/reader/` (follow `RM_*.py` patterns).
- New task-specific dataset behavior: `src/data_factory/dataset_task/`.
- New sampling strategy: `src/data_factory/samplers/`.
- New factory type (advanced): register in `src/data_factory/data_factory.py` and set `data.factory_name`.

## ID-based Pipeline (why it exists)

`factory_name: "id"` is intended for large-scale or on-demand processing workflows where you want to defer heavy work
(e.g., windowing/normalization) and keep batches ID-addressable for downstream task logic.

When changing this path, validate that tasks consuming ID batches still work with the expected keys
(e.g. `x`, `y`, and `file_id` where applicable).

## Validation Gate (minimal)

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml
python -m pytest test/
```

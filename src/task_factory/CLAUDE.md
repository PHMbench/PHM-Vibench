# Task Factory - CLAUDE.md

This file captures the design intent and change strategy for `src/task_factory/`.
The canonical user-facing guide (layout, config keys, registry SSOT) is [@README.md].

## Invariants (don’t break)

- Task selection is config-driven via `task.type` + `task.name`.
- The supported set is documented in `src/task_factory/task_registry.csv` (treat it as SSOT).
- Tasks should accept dict-style batches (e.g. `batch["x"]`, `batch["y"]`, and optional keys like `file_id`).
- Keep task logic independent from data reading and model architecture (data/model changes belong to their factories).

## Extension Points

- New task implementation: `src/task_factory/task/<TYPE>/<name>.py`
- Shared task utilities: `src/task_factory/Components/`
- Registry row: add/update in `src/task_factory/task_registry.csv`

## Safe Change Rules

- Avoid breaking the `task.type` / `task.name` namespace; add new rows rather than renaming existing ones.
- If you change batch format expectations, update the registry notes and at least one demo config.
- Prefer smoke-friendly defaults in demos (`--override trainer.num_epochs=1` for validation).

## Validation Gate (minimal)

```bash
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --dump targets
python -m scripts.validate_configs
python -m pytest test/
```

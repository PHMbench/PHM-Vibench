# config_schema

Pydantic-backed schema used by `python -m scripts.validate_configs`.

Notes for UXFD merge:
- `trainer.extensions.*` is treated as an optional, forward-compatible namespace for orchestration features
  (explain/report/collect/agent). Fields under it must be safe to ignore when not implemented.


# trainer_factory/extensions

Optional trainer extensions wired via `trainer.extensions.*` (no extra top-level YAML blocks).

Currently implemented:
- `report.manifest`: write `<run_dir>/artifacts/manifest.json`

Planned (see UXFD final plan):
- `explain`: for now the default pipeline writes `artifacts/data_metadata_snapshot.json` and (when enabled) `artifacts/explain/eligibility.json`; full explainer execution will be moved into an extension callback later
- `collect`: batch manifest collection (prefer scripts for cross-run)
- `agent`: TODO-only distillation (LLM off by default)

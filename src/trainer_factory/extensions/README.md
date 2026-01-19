# trainer_factory/extensions

Optional trainer extensions wired via `trainer.extensions.*` (no extra top-level YAML blocks).

Currently implemented:
- `manifest`: write `<run_dir>/artifacts/manifest.json` (currently invoked by `src/Pipeline_01_default.py`, best-effort)
- `agent`: LLM-free distillation → `<run_dir>/artifacts/distilled/summary.json` (when `trainer.extensions.agent.enable=true`)

Related (implemented in `src/task_factory/Default_task.py`, but configured under `trainer.extensions.*` for consistency):
- `predictions`: write `<run_dir>/artifacts/predictions.npz` (when `trainer.extensions.predictions.enable=true`)

Planned (see UXFD final plan):
- `explain`: for now the default pipeline writes `artifacts/data_metadata_snapshot.json` and (when enabled) `artifacts/explain/eligibility.json`; full explainer execution will be moved into an extension callback later
- `collect`: batch manifest collection (prefer scripts for cross-run)

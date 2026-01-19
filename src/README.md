# src/

Runtime code for PHM‑Vibench pipelines and factories.

Entry point:
- `python main.py --config <yaml> [--override key=value ...]`

Pipelines:
- `src/Pipeline_01_default.py` (maintained default)

Factories:
- `src/data_factory/`
- `src/model_factory/`
- `src/task_factory/`
- `src/trainer_factory/`

UXFD merge:
- Entry index: `paper/LQ_vibench_fix/merge_uxfd/README.md`
- SSOT status/TODO: `paper/LQ_vibench_fix/merge_uxfd/1_15/codex/TODO.md`

Run artifacts (UXFD merge, best-effort):
- `<run_dir>/config_snapshot.yaml` (fully resolved config snapshot)
- `<run_dir>/artifacts/manifest.json` (run evidence index)
- `<run_dir>/artifacts/data_metadata_snapshot.json` (best-effort data/batch metadata snapshot)
- `<run_dir>/artifacts/explain/eligibility.json` (only when `trainer.extensions.explain.enable=true`)
- `<run_dir>/artifacts/predictions.npz` (only when `trainer.extensions.predictions.enable=true`)
- `<run_dir>/artifacts/distilled/summary.json` (only when `trainer.extensions.agent.enable=true`, LLM-free)

# explain_factory

Explainability utilities for PHM‑Vibench (UXFD merge).

Principles:
- Reads **data metadata** (dataset/batch), not run logs.
- Never hard-crashes when optional explain dependencies are missing; writes auditable `eligibility.json` instead.
- Produces structured artifacts under `<run_dir>/artifacts/explain/`.

Entry points (planned):
- Trainer extension: `trainer.extensions.explain`
- Explainers: `src/explain_factory/explainers/`


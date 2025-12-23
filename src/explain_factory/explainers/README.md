# explainers

Explainer implementations used by `trainer.extensions.explain.explainer`.

This directory is intended to **wrap and inherit existing explainability methods** (UXFD merge), e.g.:
- `GradCAM_XFD.py` from `Unified_X_fault_diagnosis/model_collection/`

Each explainer must:
- Be optional-dependency safe (no hard ImportError).
- Accept `x` and `meta` (data metadata) and write artifacts under `<run_dir>/artifacts/explain/`.


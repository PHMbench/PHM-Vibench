# X_model (Explainability / Auxiliary Models)

This directory hosts models used for explainability, feature extraction, and auxiliary processing.

Typical modules:
- `Feature_extract.py`
- `MWA_CNN.py`
- `Signal_processing.py`
- `TSPN.py`
- `TSPN_UXFD.py` (UXFD-aligned stable wrapper)
- `BASE_ExplainableCNN.py` (baseline entry)
- `UXFD/` (organized common UXFD modules)
- `baselines/` (comparison baselines used by UXFD papers)

Usage patterns may vary and are often task-specific. When a model here is intended to be instantiated by `model_factory`, it should follow the same pattern:

```yaml
model:
  type: "X_model"
  name: "Feature_extract"  # or another class exposed in this directory
  # additional hyperparameters...
```

For such models:
- `model.embedding`, `model.backbone`, and `model.task_head` are not used and should be recorded as `not_applicable` in the CSV registry.
- Please document any common configuration fields here as these components stabilize.

## UXFD merge notes

- Paper-specific configs and mapping docs live in each paper submodule under `paper/UXFD_paper/<paper_id>/`.
- This directory only keeps reusable code and stable model entry modules that the vibench `model_factory` can import.

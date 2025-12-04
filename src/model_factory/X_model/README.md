# X_model (Explainability / Auxiliary Models)

This directory hosts models used for explainability, feature extraction, and auxiliary processing.

Typical modules:
- `Feature_extract.py`
- `MWA_CNN.py`
- `Signal_processing.py`
- `TSPN.py`

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


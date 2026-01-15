# UXFD Signal Processing (1D)

Reusable 1D signal processing operators aligned with the upstream UXFD `Signal_processing.py`.

Conventions:
- Default tensor layout: `(B, L, C)` (batch, length, channels).
- Operator implementations should be pure `torch.nn.Module` and avoid side effects (no filesystem writes).

Current status:
- This directory currently re-exports selected base classes from `src/model_factory/X_model/Signal_processing.py`.


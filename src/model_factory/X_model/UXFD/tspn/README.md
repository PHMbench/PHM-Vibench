# UXFD TSPN

TSPN-compatible models and wrappers, aligned with upstream UXFD `TSPN.py`.

Conventions:
- Input tensor layout: `(B, L, C)`
- Output tensor layout: `(B, num_classes)`
- Keep the core layer structure:
  - `SignalProcessingLayer → FeatureExtractorlayer → Classifier`

The vibench-loadable models are exposed as modules under `src/model_factory/X_model/`.


# UXFD Signal Processing (2D)

Reusable 2D time-frequency operators aligned with the upstream UXFD `Signal_processing_2D.py`.

Conventions:
- Input layout: `(B, L, C)` time-domain signal.
- Output layout (recommended): `(B, T, F, C)` where:
  - `T`: time frames
  - `F`: frequency bins
  - `C`: channels

Implementations here are **pure PyTorch** and must not require extra dependencies.


# Generative Components

Reusable generative losses, samplers, metrics, and manifest helpers live here.
V0 uses Conditional Flow Matching with `[N, C, L]` tensors. FFT/envelope/TSTR
logic is eval-only and must not be added to training loss.


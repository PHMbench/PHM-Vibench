# Future Generative Model Blueprints

This document tracks research directions that are not benchmark-valid by
default. Promotion requires the V0/V1 protocol, manifest, leakage checks, and
eval metrics to be mature for the method.

- Rectified Flow: runtime loss added as a straight-path velocity baseline; it
  reuses the V0 Euler ODE sampler and remains exploratory by default.
- DDPM: epsilon loss, scheduler, and sampler are available for future baseline
  wiring; outputs remain exploratory until manifest and leakage checks pass.
- Score SDE: research skeleton only; no predictor-corrector sampler in core V0.
- Mamba1D: stateless adapter placeholder only; no mandatory CUDA Mamba
  dependency.
- MeanFlow: research-only notes.
- Drifting Models: research-only notes.

# Generative Components

Reusable generative losses, samplers, metrics, schedulers, and manifest helpers
live here. This README is the canonical component-level guide for PHM generative
work.

## Placement

```text
Losses:
  src/task_factory/Components/generative/losses/

Samplers:
  src/task_factory/Components/generative/samplers/

Schedulers:
  src/task_factory/Components/generative/schedulers/

Metrics:
  src/task_factory/Components/generative/metrics/

Manifests:
  src/task_factory/Components/generative/manifests/
```

Do not create `src/phm_factory/`.

## V0 Boundary

V0 uses Conditional Flow Matching with `[N, C, L]` tensors. FFT, STFT, Hilbert
envelope, distributional metrics, TSTR, and TRTS logic are eval-only and must
not be added to the training loss.

Future spectral guidance must be handled by a separate research goal and should
prefer multi-scale STFT spectral convergence over direct full-FFT loss.

## Evidence Contracts

Generative components must expose explicit failures rather than silent fallback:

- samplers guard finite values, shape, dtype, and device after updates
- metrics report missing value status and reasons
- manifests record config, protocol, split, condition, normalization, leakage,
  and metric evidence before benchmark-valid claims
- benchmark-valid output requires source split `train` and leakage checks

# UXFD Fact Table (SSOT)

This document is **fact-driven**: it only describes what exists in code/config today.

## Core Model Contract

All UXFD demos/papers should converge on:

- `model.type: X_model`
- `model.name: TSPN_UXFD`

Implementation entry:
- `src/model_factory/X_model/TSPN_UXFD.py`

## Data Flow (high level)

Input:
- tensor `x` with shape `(B, L, C)` (time-series windows; `C=in_channels`)

Forward (default):
- `TSPN` path in `src/model_factory/X_model/TSPN.py`:
  - Signal processing layers (operator graph)
  - Feature extraction (statistical features)
  - Classifier → logits `(B, num_classes)`

Optional UXFD modules (assembled by `TSPN_UXFD`):
- `model.uxfd.operator_attention.enable=true`: pre-processing of `x` via operator-attention
- `model.uxfd.enable_sp2d=true`: 2D STFT branch + fusion into 1D feature vector
- `model.uxfd.fuzzy.enable=true`: fuzzy logits residual added to base logits
- `model.uxfd.logic.enable=true`: logic logits residual added to base logits

## NSN Wrapper

NSN is planned for U3 and is out of scope for the U1 runtime contract. This repository state only exposes the
`TSPN_UXFD` entrypoint listed above.

Semantic rule for future NSN work:
- `STFT` is not an `ALL_SP` operator key. If a YAML uses an `STFT` token, it must be mapped to the SP2D branch
  (`model.uxfd.enable_sp2d=true`) rather than treated as a 1D operator.

## Configuration Surface (selected knobs)

TSPN composition:
- `model.signal_processing_configs.layer*`: operator keys (see `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md`)
- `model.feature_extractor_configs`: feature keys (see `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md`)

UXFD assembly toggles:
- `model.uxfd.enable_sp2d` (bool)
- `model.uxfd.sp2d.n_fft`, `model.uxfd.sp2d.hop_length`
- `model.uxfd.fusion.type`: `concat|sum|gated`
- `model.uxfd.fuzzy.enable`, `model.uxfd.fuzzy.logit_scale`
- `model.uxfd.operator_attention.enable`, `model.uxfd.operator_attention.operators`
- `model.uxfd.logic.enable`, `model.uxfd.logic.logit_scale`

Run artifacts:
- `<run_dir>/config_snapshot.yaml`
- `<run_dir>/artifacts/manifest.json`
- optional: `<run_dir>/artifacts/predictions.npz` when `trainer.extensions.predictions.enable=true`

## Runnable Demos

Maintained UXFD and NSN demos are planned for U2/U3 and are not part of this U1 contract.

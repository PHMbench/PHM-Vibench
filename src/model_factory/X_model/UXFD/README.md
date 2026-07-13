# UXFD (Common Modules)

This package hosts **reusable UXFD building blocks** that are shared across the UXFD paper submodules.

## Quick Validate

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m pytest test/test_tspn_uxfd_assembly.py -q
```

## Core Contract (SSOT)

One core model contract for UXFD-enabled runs:
- `model.type: X_model`
- `model.name: TSPN_UXFD` (implementation: `src/model_factory/X_model/TSPN_UXFD.py`)

The composition surface is config-driven:
- base operator graph: `model.signal_processing_configs.layer*` (keys from `ALL_SP`)
- base feature extraction: `model.feature_extractor_configs` (keys from `ALL_FE`)
- UXFD assembly toggles: `model.uxfd.*` (SP2D / fusion / fuzzy / operator-attention / logic)

SSOT docs:
- `src/model_factory/X_model/UXFD/FACT_TABLE.md` (what exists today + artifact contract)
- `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md` (operator/feature key catalog)

## Package Layout

- `src/model_factory/X_model/UXFD/signal_processing_1d/`: 1D signal operators used by `TSPN` (see `ALL_SP`).
- `src/model_factory/X_model/UXFD/signal_processing_2d/`: 2D time-frequency branch (STFT) used by `TSPN_UXFD`.
- `src/model_factory/X_model/UXFD/fusion/`: fusion modules for combining SP2D branch with 1D features.
- `src/model_factory/X_model/UXFD/fuzzy/`: fuzzy residual logits (best-effort, config-gated).
- `src/model_factory/X_model/UXFD/operator_attention/`: operator-attention pre-processing of raw input (config-gated).
- `src/model_factory/X_model/UXFD/neurosymbolic/`: neurosymbolic / logic residual logits (best-effort, config-gated).

## Repo Discipline

- Keep paper-specific configs inside each paper submodule (`paper/UXFD_paper/<paper_id>/`).
- Keep shared code here. Maintained runnable entrypoints under `configs/demo/uxfd/` are planned for U2.

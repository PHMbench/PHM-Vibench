# Model, Loss, And Baseline Registry

This document summarizes source-derived support status. The source of truth remains:

- `src/model_factory/model_registry.csv`
- `src/model_factory/ISFM/isfm_components.csv`
- `src/task_factory/Components/README.md`
- `scripts/model_support_matrix.py`

## Current Support Rule

- `smoke-tested`: covered by a focused test or maintained smoke config.
- `dependency-blocked`: registry entry is valid, but the current environment lacks
  an optional dependency.
- `unverified`: registry-backed, but no passing smoke evidence is recorded.
- `failed`: validation found a concrete missing path or contract failure.
- `unsupported`: absent from source-of-truth registries or intentionally out of scope.

## Current Evidence Highlights

- `ISFM.M_01_ISFM`: `smoke-tested` through
  `configs/hydra/experiments/00_smoke/dummy_dg.yaml`.
- `X_model.*`: covered by `test/test_x_model_smoke.py`; `CI_GNN` is
  `dependency-blocked` when `torch_geometric` is unavailable.
- ISFM components used by the smoke config are `smoke-tested`:
  `E_01_HSE`, `B_04_Dlinear`, `H_01_Linear_cla`.
- Loss keys are validated through `test/test_loss_component_contract.py`,
  `test/test_infonce_pairing.py`, and `test/test_hse_contrastive_failfast.py`.
- 2025+ PHM literature methods are tracked in `docs/literature/README.md` and
  validated by `python -m scripts.phm_literature_matrix --min-count 50`. These
  entries are literature mappings, not runtime support claims.

## Validation Commands

```bash
python -m scripts.model_support_matrix
python -m scripts.phm_literature_matrix --min-count 50
python -m pytest -q test/test_model_registry_contract.py
python -m pytest -q test/test_loss_component_contract.py
python -m pytest -q test/test_x_model_smoke.py test/test_tspn_uxfd_assembly.py
```

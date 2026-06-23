# Baseline Mapping

Baseline mapping is derived by `scripts/baseline_mapping.py` from model support
status and the Slice 2 task matrix. It is not a frozen paper claim list.

| Task family | Role | Model | Evidence | Blocker / note |
|---|---|---|---|---|
| `DG.classification` | `mandatory` | `ISFM.M_01_ISFM` | `smoke-tested` | `bash scripts/run_demo_matrix.sh --mode smoke` |
| `CDDG.classification` | `optional` | `ISFM.M_01_ISFM` | `real-data-ready` | `PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full` |
| `FS.classification` | `optional` | `ISFM.M_01_ISFM` | `real-data-ready` | `PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full` |
| `GFS.classification` | `optional` | `ISFM.M_01_ISFM` | `real-data-ready` | `PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full` |
| `pretrain.hse_contrastive` | `mandatory` | `ISFM.M_01_ISFM` | `smoke-tested` | focused HSE contrastive test |
| `DG.classification` | `blocked` | `X_model.CI_GNN` | `dependency-blocked` | requires `torch_geometric` in the current environment |

## Validation Command

```bash
python -m scripts.baseline_mapping
python -m pytest -q test/test_baseline_mapping_contract.py
```

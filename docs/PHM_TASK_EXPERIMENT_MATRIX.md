# PHM Task Experiment Matrix

This matrix is derived from `src/task_factory/task_registry.csv` and
`configs/config_registry.csv` through `python -m scripts.task_experiment_matrix`.
It is not a second source of truth.

Recent 2025+ PHM papers mapped to these task families are tracked separately in
`docs/literature/README.md` and validated by
`python -m scripts.phm_literature_matrix --min-count 50`. Literature mappings do
not change a task status unless a registry/config/test gate exists.

| Task type | Task name | Status | Evidence / reason |
|---|---|---|---|
| `CDDG` | `classification` | `real-data-ready` | full matrix config requires `PHM_VIBENCH_DATA`: `configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml` |
| `DG` | `classification` | `smoke-tested` | offline smoke config: `configs/hydra/experiments/00_smoke/dummy_dg.yaml` |
| `Default_task` | `Default_task` | `unverified` | registry-backed, but no maintained config is recorded |
| `Default_task` | `ID_task` | `unverified` | registry-backed, but no maintained config is recorded |
| `FS` | `classification` | `real-data-ready` | full matrix config requires `PHM_VIBENCH_DATA`: `configs/hydra/experiments/03_fewshot/cwru_protonet.yaml` |
| `FS` | `finetuning` | `unverified` | registry-backed, but no maintained config is recorded |
| `FS` | `knn_feature` | `unverified` | registry-backed, but no maintained config is recorded |
| `FS` | `matching_network` | `unverified` | registry-backed, but no maintained config is recorded |
| `FS` | `prototypical_network` | `unverified` | registry-backed, but no maintained config is recorded |
| `GFS` | `classification` | `real-data-ready` | full matrix config requires `PHM_VIBENCH_DATA`: `configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml` |
| `GFS` | `matching` | `unverified` | registry-backed, but no maintained config is recorded |
| `pretrain` | `classification` | `unverified` | registry-backed, but no maintained config is recorded |
| `pretrain` | `classification_prediction` | `unverified` | registry-backed, but no maintained config is recorded |
| `pretrain` | `hse_contrastive` | `smoke-tested` | focused offline test: `test/test_hse_contrastive_failfast.py::test_hse_contrastive_flow_has_nonzero_signal` |
| `pretrain` | `masked_reconstruction` | `unverified` | registry-backed, but no maintained config is recorded |
| `pretrain` | `prediction` | `unverified` | registry-backed, but no maintained config is recorded |

## Absent Capabilities

- `multi-task`: unsupported. Multi-task code exists experimentally but has no
  task-registry row.
- `regression`: unsupported. No registry-backed regression task family is currently
  exposed.

## Validation Commands

```bash
python -m scripts.task_experiment_matrix
python -m scripts.phm_literature_matrix --min-count 50
python -m pytest -q test/test_task_experiment_matrix.py
bash scripts/run_demo_matrix.sh --mode smoke
env -u PHM_VIBENCH_DATA bash scripts/run_demo_matrix.sh --mode full
PHM_VIBENCH_DATA=/home/user/data/PHMbenchdata/PHM-Vibench bash scripts/run_demo_matrix.sh --mode full
```

## Full Benchmark Evidence

Recorded on 2026-05-11 in the `LQ_signal` conda environment:

```bash
PHM_VIBENCH_DATA=/home/user/data/PHMbenchdata/PHM-Vibench bash scripts/run_demo_matrix.sh --mode full
```

Result: passed. The command ran the offline smoke config plus six real-data Hydra
configs and each run wrote both `artifacts/manifest.json` and `test_result_0.csv`.

Latest evidence paths:

- `smoke`: `results/demo/dummy_dg_smoke/metadata_dummy.csv/M_M_01_ISFM/T_DGclassification_11_155830/iter_0/artifacts/manifest.json`
- `DG.classification`: `results/demo/cwru_dg/metadata.xlsx/M_M_01_ISFM/T_DGclassification_11_155838/iter_0/artifacts/manifest.json`
- `CDDG.classification`: `results/demo/multi_system_cddg/metadata.xlsx/M_M_01_ISFM/T_CDDGclassification_11_155909/iter_0/artifacts/manifest.json`
- `FS.classification`: `results/demo/cwru_protonet/metadata.xlsx/M_M_01_ISFM/T_FSclassification_11_155941/iter_0/artifacts/manifest.json`
- `GFS.classification`: `results/demo/cross_system_fewshot_tspn/metadata.xlsx/M_M_01_ISFM/T_GFSclassification_11_160006/iter_0/artifacts/manifest.json`
- `Pipeline_02 pretrain.hse_contrastive`: `results/demo/pretrain_hse_then_fewshot/metadata.xlsx/M_M_01_ISFM/T_pretrainhse_contrastive_11_160056/iter_0/artifacts/manifest.json`
- `pretrain.hse_contrastive CDDG`: `results/demo/pretrain_hse_cddg/metadata.xlsx/M_M_01_ISFM/T_pretrainhse_contrastive_11_160246/iter_0/artifacts/manifest.json`

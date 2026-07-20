# P02 - XFD Benchmark Toolkit - additive experiment configs

These configs are ADDITIVE: they live under
`configs/experiments/p02_xfd_benchmark_toolkit/` and do NOT modify any
existing config under `configs/base/`, `configs/demo/`, or `configs/reference/`.

## Purpose

Provide the same-protocol config scaffolding for the planned P02-G050 / G060
runs that the normalized manuscript is waiting on.

Two engines are involved (see `paper/experiments/experiment_plan.md`):

- **E1 (legacy toolkit)** - drives the 5-model explainability cube from its own
  scripts under `paper/UXFD_paper/Explainable_FD_Toolkit/scripts/`. The 5
  toolkit models (TSPN, Fusion1D2D, MoE, OperatorAttention, FuzzyLogic) are NOT
  registered in `src/model_factory/model_registry.csv`, so this engine cannot be
  launched through `main.py`. The two YAMLs below are traceability bindings for
  the run ledger.
- **E2 (`main.py`)** - drives the real-data, registry-style baselines that the
  cube is missing. The three `p02_resnet1d_*.yaml` configs below are runnable.

### Traceability configs (Engine E1, not launched via `main.py`)

- `p02_toolkit_benchmark.yaml` - six-plus-baseline matrix entrypoint
  (TSPN, Fusion1D2D, MoE, OperatorAttention, FuzzyLogic, +1 candidate)
  driven by `scripts/run_unified_explain_eval.py`.
- `p02_toolkit_ablation.yaml` - Toolkit ablation runner binding
  (`scripts/run_toolkit_ablations.py` - EXISTS, 216 LOC; in the legacy snapshot
  it consumed synthetic fixtures, real-data binding requires the protocol lock).

### Runnable configs (Engine E2, via `python main.py --config <yaml>`)

6th same-protocol baseline candidate (CNN.ResNet1D, registered), one per dataset.
Smoke-friendly defaults (1 epoch, CPU); for decisive runs apply the
`grouped_metadata` leakage-safe overrides documented in each file's header:

- `p02_resnet1d_cwru.yaml`    - Dataset_id=1 (CWRU)
- `p02_resnet1d_xjtu.yaml`    - Dataset_id=2 (XJTU)
- `p02_resnet1d_thu018.yaml`  - Dataset_id=14 (RM_018_THU24 = manuscript "THU_018_basic")

The E1 traceability configs are **planned, not runnable as-is** through `main.py`.
The E2 configs are runnable today (status: ready).

## Source provenance

- Legacy toolkit scripts: `paper/UXFD_paper/Explainable_FD_Toolkit/scripts/`
  (READ-ONLY; do not edit).
- Legacy parent smoke config: `paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml`.
- Engine: PHM-Vibench_fix (`main.py --config <yaml>`).

## Non-destructive attestation

- No file under `configs/base/`, `configs/demo/`, or `configs/reference/` was modified.
- No git commit / push was performed.
- The legacy `paper/UXFD_paper/Explainable_FD_Toolkit/` tree remains read-only.

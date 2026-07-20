# T040 Evidence README

Status on 2026-05-11: partial. The placeholder figure path, first benchmark
evidence binding, and generic manuscript placeholder body are resolved from
existing submodule artifacts and an evidence-bound IEEEtran checkpoint, but the
paper is not IEEE Transactions submission-ready.

## Resolved In This Patch

- Replaced the missing TeX figure path `../../figures/example.pdf` with the
  existing benchmark figure
  `../../results/autoresearch/20260319_090111/benchmark/overall_scores_comparison.png`.
- Replaced the placeholder comparison table with values from
  `results/autoresearch/20260319_090111/benchmark/explainability_benchmark_table.csv`.
- Replaced the generic manuscript title, abstract, method, discussion, and
  conclusion placeholders with a conservative evidence checkpoint in
  `manuscript/final_tex/main.tex`.
- Updated `VIBENCH.md` command examples to record `CUDA_VISIBLE_DEVICES=0` and
  point standalone output at the submodule-local `results/autoresearch/...`
  directory.

## Local Evidence That Exists

| Gate fragment | Current artifact | Status |
|---|---|---|
| Benchmark bootstrap | `outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111/{run_meta.yaml,metrics.json}` | Schema-valid. |
| Benchmark table and figure | `results/autoresearch/20260319_090111/benchmark/explainability_benchmark_table.csv`, `results/autoresearch/20260319_090111/benchmark/overall_scores_comparison.png` | Bound into `manuscript/final_tex/main.tex`. |
| Manuscript checkpoint | `manuscript/final_tex/main.tex` | Compiles with `pdflatex` to `/tmp/uxfd_paper01_tex/main.pdf`; not final submission text. |
| Five-model matrix | `results/autoresearch/20260319_162507/unified_model_matrix/benchmark_results_table.csv` | Present, but only five diagnostic models. |
| Captum comparison | `outputs/RM_COMPETITOR_SYNTH/ToolkitVsCaptum/seed_0/20260319_162715/{run_meta.yaml,metrics.json}` | Schema-valid synthetic comparison. |
| SHAP/LIME comparison | `outputs/RM_COMPETITOR_SYNTH/ToolkitVsShapLime/seed_0/20260319_163123/{run_meta.yaml,metrics.json}` | Schema-valid synthetic comparison. |
| THU018 matrix | `outputs/RM_THU018_UNIFIED/UnifiedExplainEval/seed_0/20260320_104118/{run_meta.yaml,metrics.json}` | Schema-valid additional dataset artifact. |

Validation command pattern:

```bash
python scripts/validate_schema.py --run_dir outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111
```

## Remaining IEEE Submission Blockers

### 1. Six-plus same-protocol baselines

Current exact artifact:
`results/autoresearch/20260319_162507/unified_model_matrix/benchmark_results_table.csv`
contains only `TSPN`, `Fusion1D2D`, `MoE`, `OperatorAttention`, and
`FuzzyLogic`.

Missing artifact:

```text
results/autoresearch/<run_id>/six_baseline_matrix/benchmark_results_table.csv
outputs/RM_MULTI_CWRU_XJTU/<six_baseline_model_matrix>/seed_<seed>/<run_id>/run_meta.yaml
outputs/RM_MULTI_CWRU_XJTU/<six_baseline_model_matrix>/seed_<seed>/<run_id>/metrics.json
```

The current `scripts/run_unified_explain_eval.py` model registry defines only
five models, and `configs/` contains no same-protocol Toolkit configs for the
additional required baseline candidates such as `ResNet1D`, `SincNet`, `TFN`,
or `PatchTST`.

Current command-bound checkpoint:

```text
submission_prep/baseline_ablation_matrix.yaml
```

This matrix records six PHM-Vibench model baseline commands with dummy-smoke
validation in `LQ_signal`. It is not accepted same-protocol evidence because it
uses dummy data and CPU fallback in the current sandbox.

### 2. Ablation suite

No submodule-local artifact currently covers the required Toolkit ablations:
schema removal, faithfulness/stability/efficiency metric-family removal,
standardized manifest on/off, and fixed-seed/config-snapshot on/off.

The matrix binds only one smoke-level Toolkit ablation: disabling the
PHM-Vibench explain extension. The remaining Toolkit ablation hooks still need
an implementation runner before reviewer-facing claims are possible.

Missing artifact:

```text
results/autoresearch/<run_id>/toolkit_ablation_matrix/ablation_results_table.csv
outputs/RM_MULTI_CWRU_XJTU/ToolkitAblation/seed_<seed>/<run_id>/run_meta.yaml
outputs/RM_MULTI_CWRU_XJTU/ToolkitAblation/seed_<seed>/<run_id>/metrics.json
```

Candidate command after an ablation runner exists:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_toolkit_ablations.py --datasets CWRU,XJTU --seeds 0,1,2 --output results/autoresearch/<run_id>/toolkit_ablation_matrix
python scripts/validate_schema.py --run_dir outputs/RM_MULTI_CWRU_XJTU/ToolkitAblation/seed_0/<run_id>
```

### 3. TOP-source runnable baseline binding

The goal package requires Toolkit linkage to `RWTOP2024-TIMEXPP`,
`RWTOP2024-MOMENT`, `RWTOP2025-DADA`, and `RWTOP2025-CFCBM`. No
submodule-local result currently names those IDs or records exact/representative
commands for them.

Missing artifact:

```text
results/autoresearch/<run_id>/top_recent_work_mapping/top_recent_work_status.json
results/autoresearch/<run_id>/top_recent_work_mapping/top_recent_work_status.md
```

Candidate command after proxy support exists:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_top_proxy_eval.py --methods timexpp_proxy,moment_proxy,dada_proxy --datasets CWRU,XJTU --output results/autoresearch/<run_id>/top_recent_work_mapping
```

`RWTOP2025-CFCBM` remains literature-only until fault-diagnosis concept labels
and a local 2x4090-feasible protocol exist.

### 4. Compute metadata

Existing accepted `run_meta.yaml` files record `env.device: cuda`, but they do
not record all new global gate fields: `CUDA_VISIBLE_DEVICES`, GPU model, GPU
count, device IDs, batch size, precision, runtime, and OOM/failure reason.

Required next runner behavior:

```text
run_meta.yaml -> env.cuda_visible_devices, env.gpu_model, env.gpu_count
run_meta.yaml -> run.batch_size, run.precision, run.runtime_sec
run_meta.yaml -> failures.oom, failures.resource_blocked_reason
```

### 5. SOTA wording

SOTA infrastructure claims remain blocked until the six-plus baseline matrix,
TOP-source runnable mapping, ablations, and compute metadata all pass under the
same dataset split, seed protocol, preprocessing, and metrics.

### 6. Manuscript package

The canonical entrypoint `manuscript/final_tex/main.tex` now compiles as a
conservative IEEEtran evidence checkpoint and no longer contains the generic
title, abstract, method, discussion, and conclusion placeholders. It is still
not an IEEE Transactions submission manuscript because accepted six-baseline
evidence, same-protocol ablations, TOP representative artifacts, strict local
GPU metadata, and final SOTA-safe wording are missing.

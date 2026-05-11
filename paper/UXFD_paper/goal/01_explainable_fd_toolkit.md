# Paper Goal: Explainable FD Toolkit

## Target

- Submodule: `paper/UXFD_paper/Explainable_FD_Toolkit`
- Default journal: IEEE Transactions on Industrial Informatics
- Alternate journal: IEEE Transactions on Artificial Intelligence
- Contribution: a unified explainability interface, metric protocol, and benchmark/reporting toolkit for fault diagnosis.

## Canonical Package

- Manuscript entrypoint: `manuscript/final_tex/main.tex`
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current manuscript status: `manuscript/final_tex/main.tex` compiles as an
  evidence-bound IEEEtran checkpoint and no longer contains generic title,
  abstract, method, discussion, or conclusion placeholders. Final
  evidence-bearing IEEE text remains blocked until accepted six-baseline,
  Toolkit-ablation, TOP representative, GPU metadata, and SOTA-safe wording
  gates pass.

## Required Evidence

- Multi-model and multi-method benchmark across at least CWRU and XJTU.
- Captum/SHAP/LIME or documented alternatives for post-hoc comparison.
- Unified schema examples for `run_meta.yaml`, `metrics.json`, and generated tables.
- Demo evidence with latency/failure-rate reporting.

## Baseline Suite

- `ISFM.M_01_ISFM` with Toolkit explanations.
- `X_model.NSN` or `TSPN_UXFD` with Toolkit explanations.
- `CNN.ResNet1D` with Toolkit explanations.
- `X_model.Sincnet` with Toolkit explanations.
- `X_model.TFN` with Toolkit explanations.
- Captum Integrated Gradients or Saliency as post-hoc explanation baseline.
- SHAP or LIME as perturbation-based explanation baseline.

## Ablation Suite

- Remove the unified schema and compare against ad hoc per-method outputs.
- Disable each metric family: faithfulness, stability, and efficiency.
- Compare report generation with and without standardized artifact manifests.
- Compare benchmark reproducibility with and without fixed seeds and config snapshots.

## SOTA Optimization Gate

- The Toolkit paper may claim SOTA infrastructure only if it beats the declared explanation baselines on reproducibility, metric coverage, latency/reporting overhead, and benchmark completeness under the same protocol.
- If diagnosis accuracy is reported, it must be tied to the diagnostic model baseline and cannot be attributed to the Toolkit unless the Toolkit changes the model or decision path.

## TOP Recent-Work Quota

- RWTOP2024-TIMEXPP: `representative-runnable` time-series explanation baseline for faithfulness and stability.
- RWTOP2024-MOMENT: `representative-runnable` foundation-model representation baseline for explanation coverage.
- RWTOP2025-DADA: `representative-runnable` bottleneck/anomaly baseline for general diagnostic explanation.
- RWTOP2025-CFCBM: `literature-only` concept/counterfactual comparator until FD concepts are defined.
- RWTOP2026-TIMESEG: `representative-runnable` segment-wise explanation comparator for temporal faithfulness.
- RWTOP2026-TIMESLIVER: `representative-runnable` symbolic-linear attribution comparator for Toolkit reporting.
- RWTOP2026-TSPULSE: `representative-runnable` compact pretrained representation comparator under the 2x4090 budget.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` by default.
- Default execution: one GPU per toolkit benchmark job; at most two concurrent jobs across explanation methods.
- Runtime tier: smoke and benchmark runs must fit a single RTX 4090; large TOP methods that exceed this budget are `resource-blocked` for exact reproduction.
- Required metadata: device ID, GPU model, seed, batch size, precision, runtime, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- Toolkit papers fail if they are only a wrapper around demos.
- Synthetic-only benchmark evidence cannot support claims about industrial fault diagnosis coverage.
- Interface claims must be backed by working examples and schema validation.

## Acceptance Gates

- Canonical TeX compiles without fatal errors.
- Benchmark table links to accepted CSV/JSON artifacts.
- At least six baselines and the Toolkit ablations above are present in accepted artifacts.
- Any SOTA infrastructure claim satisfies the SOTA optimization gate.
- All API, metric, and report-generation claims map to runnable examples or blockers.
- Submodule-local commit records the accepted paper package milestone.

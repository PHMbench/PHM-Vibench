# Paper Goal: 1D-2D Fusion Explainable Fault Diagnosis

## Target

- Submodule: `paper/UXFD_paper/1D-2D_fusion_explainable`
- Default journal: IEEE Transactions on Industrial Informatics
- Alternate journal: IEEE Transactions on Industrial Electronics or Information Fusion
- Contribution: physically and semantically aligned 1D time-series plus 2D time-frequency fusion with measurable explanations.

## Canonical Package

- Manuscript entrypoint: `paper_draft/NMI_Paper1_Fusion1D2D.tex` is currently named as canonical by the placeholder TeX note.
- Non-canonical placeholder: `manuscript/final_tex/main.tex`
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current compile blocker: placeholder/final entrypoint mismatch and TeX path text with underscores.

## Required Evidence

- CWRU and XJTU multi-seed performance with confidence intervals.
- Alignment ablations for 1D-only, 2D-only, and fusion variants.
- Explainability metrics: faithfulness, stability, and efficiency.
- Success and failure cases showing modality contribution and alignment consistency.

## Baseline Suite

- `ISFM.M_01_ISFM`.
- `X_model.NSN` or `TSPN_UXFD` without 2D fusion.
- `CNN.ResNet1D`.
- `X_model.Sincnet`.
- `X_model.TFN`.
- `X_model.WKN`.
- `Transformer.PatchTST` or `Transformer.ConvTransformer`.

## Ablation Suite

- 1D-only branch.
- 2D-only branch.
- Fusion without physical alignment.
- Fusion without semantic/geometric alignment.
- Late fusion versus progressive fusion.
- Explainability loss or attribution module removed.

## SOTA Optimization Gate

- The optimized fusion model must beat all declared baselines on the primary diagnostic metric and not regress on required explainability metrics under the same CWRU/XJTU split and seed protocol before any SOTA claim is allowed.
- If it only wins on explanation quality or robustness, the manuscript must scope the contribution to that axis rather than broad SOTA accuracy.

## TOP Recent-Work Quota

- RWTOP2024-TIMEMIXER: `representative-runnable` multiscale temporal baseline for the 1D branch.
- RWTOP2024-MOMENT: `representative-runnable` foundation-model representation baseline for fused features.
- RWTOP2025-CATCH: `representative-runnable` channel/frequency baseline for 2D time-frequency fusion.
- RWTOP2025-DADA: `representative-runnable` bottleneck/anomaly baseline for robust fusion.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` by default.
- Default execution: one GPU per seed/config for 1D, 2D, and fusion variants; at most two concurrent jobs.
- Runtime tier: six-baseline and fusion-ablation matrix must be scheduled as single-GPU jobs; two-GPU runs require `CUDA_VISIBLE_DEVICES=0,1` and a recorded justification.
- Resource policy: exact TOP fusion/foundation baselines that exceed 2x4090 are `resource-blocked` and must use a labelled representative run instead.
- Required metadata: device ID, GPU model, seed, batch size, precision, runtime, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- High accuracy without alignment ablation will be treated as black-box fusion.
- Historical placeholder manuscript content must not be submitted.
- THU or additional dataset claims remain blocked until accepted artifacts exist.

## Acceptance Gates

- One canonical IEEE-style TeX entrypoint is declared and compiles.
- All fusion/alignment claims map to accepted artifacts.
- Tables report mean, standard deviation, and confidence interval for required seeds.
- At least six baselines and the fusion/alignment ablations above are present in accepted artifacts.
- Any SOTA claim satisfies the SOTA optimization gate.
- Submodule-local commit records the accepted paper package milestone.

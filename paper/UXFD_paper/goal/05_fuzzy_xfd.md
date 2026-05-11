# Paper Goal: Fuzzy-XFD

## Target

- Submodule: `paper/UXFD_paper/Paper_fuzzy_XFD`
- Default journal: IEEE Transactions on Fuzzy Systems
- Alternate journal: IEEE Transactions on Industrial Informatics
- Contribution: auditable fuzzy rules and safety-oriented rule-level explanations for fault diagnosis.

## Canonical Package

- Manuscript entrypoint: `manuscript/final_tex/main.tex`
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current compile blocker: placeholder TeX content and missing `../../figures/example.pdf`.

## Required Evidence

- CWRU and XJTU multi-seed results with confidence intervals.
- Rule-level faithfulness, stability, sparsity, and efficiency metrics.
- Safety-critical failure cases with triggered rules, membership values, and decision paths.
- Clear distinction between verified final results and current-state legacy results.

## Baseline Suite

- `ISFM.M_01_ISFM`.
- `X_model.NSN` or `TSPN_UXFD` without fuzzy rules.
- `CNN.ResNet1D`.
- `X_model.Sincnet`.
- `X_model.TFN`.
- `X_model.WKN`.
- Classical fuzzy inference or rule-based classifier baseline.

## Ablation Suite

- Remove fuzzy rule layer.
- Remove membership calibration.
- Replace fuzzy inference with hard thresholds.
- Vary number of rules and membership functions.
- Remove safety fallback path.
- Remove rule-level explanation output.

## SOTA Optimization Gate

- The Fuzzy-XFD paper may claim SOTA safe/transparent diagnosis only if it beats all declared baselines on diagnostic performance and rule-level explanation metrics under the same protocol.
- If it trades small accuracy loss for safety or auditability, the paper must state that tradeoff explicitly instead of claiming accuracy SOTA.

## TOP Recent-Work Quota

- RWTOP2024-TIMEXPP: `representative-runnable` time-series explanation baseline for rule faithfulness.
- RWTOP2025-CFCBM: `literature-only` concept/counterfactual comparator until fuzzy concepts are bound.
- RWTOP2025-CBAE: `literature-only` post-hoc concept baseline until concept supervision exists.
- RWTOP2025-IFCBM: `literature-only` top-journal concept-bottleneck comparator until task mapping is defined.
- RWTOP2026-TIMESEG: `representative-runnable` segment-level explanation comparator for fuzzy-rule faithfulness.
- RWTOP2026-TIMESLIVER: `representative-runnable` symbolic-linear comparator for rule attribution.
- RWTOP2026-PROTOTS: `literature-only` hierarchical prototype comparator until diagnosis mapping exists.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` when GPU is used.
- Default execution: fuzzy/rule runs should fit one GPU or CPU; at most two GPU-backed jobs may run concurrently.
- Runtime tier: concept-bottleneck TOP comparators are `resource-blocked` for exact reproduction if concept supervision or model scale exceeds the 2x4090 budget.
- Required metadata: device ID when used, GPU model, seed, batch size, precision, runtime, rule count, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- Fuzzy rules must be interpretable and sparse enough for an engineer to audit.
- Placeholder table values cannot remain in a submission manuscript.
- Safety fallback claims require concrete failure examples.

## Acceptance Gates

- Canonical TeX compiles without placeholder title, abstract, figure, or table values.
- Rule evidence maps to accepted artifacts.
- Safety-case examples are present or explicitly blocked.
- At least six baselines and the fuzzy-rule ablations above are present in accepted artifacts.
- Any SOTA fuzzy/safety claim satisfies the SOTA optimization gate.
- Submodule-local commit records the accepted paper package milestone.

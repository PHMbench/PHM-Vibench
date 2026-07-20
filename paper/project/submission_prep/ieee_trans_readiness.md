# Paper 05 Fuzzy-XFD IEEE Transactions Readiness

Date: 2026-05-11

This checkpoint converts the Fuzzy-XFD comparison plan from prose blockers into
a command-bound matrix. It is not submission readiness.

## Current Evidence

- Matrix: `submission_prep/baseline_ablation_matrix.yaml`
- Base config: `configs/vibench/min.yaml`
- Evidence level: config-target, dummy-data smoke validation, and non-accepted
  reviewer-ablation smoke artifacts only
- Manuscript checkpoint: `manuscript/final_tex/main.tex` compiles from the
  submodule root after binding local `FuzzyLogic_explainable/results/*.pdf`
  figures; this is an evidence snapshot, not final IEEE TFS text
- Compute policy: local RTX 4090 GPUs `0,1`; every runnable command binds
  `CUDA_VISIBLE_DEVICES=0`

## Dummy-Smoke Summary

The following commands completed in `LQ_signal` on dummy data with CPU fallback
because the current environment reported `GPU available: False` and
`Can't initialize NVML`.

| ID | Role | Status |
|---|---|---|
| P00 | Fuzzy-XFD / NSN fuzzy residual head | pass; dummy only |
| B01/A01 | NSN/TSPN_UXFD without fuzzy head | pass; dummy only |
| B02 | ResNet | pass; dummy only |
| B03 | SincNet | pass; dummy only |
| B04 | TFN | pass; dummy only |
| B05 | WKN | pass; dummy only |
| B06 | ConvTransformer | pass; dummy only |
| B07 | Classical fuzzy script | pass with script-generated demo data only |
| A02-A06 | fuzzy scale/rule/membership/feature sensitivity | pass; dummy only |
| R01-R03 | hard-threshold, no-safety-fallback, no-rule-output reviewer ablations | pass; non-accepted smoke only |

## Reviewer-Ablation Smoke Summary

`python scripts/run_reviewer_ablation_smoke.py --condition all` now emits
non-accepted `run_meta.yaml` and `metrics.json` artifacts for:

- hard-threshold inference replacement;
- removing the safety fallback path;
- removing rule-level explanation output.

These artifacts prove the reviewer surfaces are command-bound. They do not
replace accepted same-protocol rule-metric or safety-case artifacts.

## Still Blocked

- CWRU/XJTU or industrial multi-seed runs with mean/std/95% CI.
- Full GPU metadata from local GPUs `0,1`.
- Rule-level faithfulness, stability, sparsity, and efficiency artifacts.
- Safety cases with sample IDs, membership values, and decision paths.
- TOP recent-work representative artifact, especially `RWTOP2024-TIMEXPP`.
- Accepted hard-threshold, safety-fallback, and no-rule-output reviewer
  ablation artifacts under the same dataset/seed/metric protocol.
- Any SOTA wording.

## Allowed Manuscript Wording

The manuscript may say that the current repository now exposes runnable
comparison, sensitivity, and reviewer-ablation smoke entrypoints for Fuzzy-XFD.
It must not claim accepted performance, real-data superiority, safety
validation, TOP-method reproduction, or SOTA from this checkpoint.

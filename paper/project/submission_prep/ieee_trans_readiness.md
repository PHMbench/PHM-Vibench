# Paper 01 Toolkit IEEE Transactions Readiness

Date: 2026-05-11

This checkpoint adds a command-bound comparison matrix for the Explainable FD
Toolkit paper. It does not make the paper submission-ready.

## Current Evidence

- Matrix: `submission_prep/baseline_ablation_matrix.yaml`
- Base config: `configs/vibench/min.yaml`
- Existing gate: `manuscript/T040_EVIDENCE_README.md`
- Manuscript checkpoint: `manuscript/final_tex/main.tex` now compiles as a
  conservative IEEEtran evidence checkpoint; it is not final submission text
- Evidence level: six PHM-Vibench baseline dummy smokes plus partial existing
  Toolkit benchmark/schema evidence plus non-accepted Toolkit ablation smoke
  artifacts and a manuscript checkpoint
- Compute policy: local RTX 4090 GPUs `0,1`; runnable commands bind
  `CUDA_VISIBLE_DEVICES=0`

## Dummy-Smoke Summary

The proposed Toolkit smoke and six model baselines completed in `LQ_signal` on
dummy data with CPU fallback because the current environment reported
`GPU available: False` and `Can't initialize NVML`.

The Toolkit ablation smoke runner also exists:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_toolkit_ablations.py --condition all --output /tmp/uxfd_paper01_toolkit_ablation_smoke --seed 0
```

It writes per-condition `run_meta.yaml` and `metrics.json` for schema removal,
metric-family removal, manifest-off, snapshot-off, and post-hoc-only comparator
conditions. These outputs are marked `accepted_evidence: false` and are not
reviewer-grade ablation evidence.

## Remaining Gaps

- Final evidence-bearing IEEE Transactions manuscript after accepted artifacts
  exist.
- Full CWRU/XJTU or industrial multi-seed six-baseline matrix.
- Accepted Toolkit-specific ablation artifacts for schema removal,
  metric-family removal, standardized manifest off, fixed-seed/config-snapshot
  off, and post-hoc-only comparator mode.
- TOP representative artifacts for TimeX++/MOMENT/DADA.
- Complete strict local GPU metadata from devices `0,1`.
- SOTA/submission-ready infrastructure gate.

## Allowed Manuscript Wording

The manuscript may state that partial Toolkit benchmark/schema evidence exists,
the repository records six runnable baseline commands, and the canonical TeX
entrypoint compiles as an evidence checkpoint. It must not claim final
same-protocol superiority, TOP-method reproduction, GPU feasibility, SOTA, or
submission readiness.

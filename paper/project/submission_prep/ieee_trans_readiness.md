# Paper 04 MoE IEEE Transactions Readiness

Date: 2026-05-11

This checkpoint adds a machine-readable comparison matrix for the MoE paper. It
does not make the paper submission-ready.

## Current Evidence

- Matrix: `submission_prep/baseline_ablation_matrix.yaml`
- Base config: `configs/vibench/min.yaml`
- Existing gate: `T043_SUBMISSION_READINESS_EVIDENCE.md`
- Evidence level: six PHM-Vibench baseline dummy smokes, partial existing
  route/expert evidence, and non-accepted MoE ablation smoke artifacts
- Compute policy: local RTX 4090 GPUs `0,1`; runnable commands bind
  `CUDA_VISIBLE_DEVICES=0`

## Dummy-Smoke Summary

The PHM-Vibench proposed proxy and B01-B06 baseline commands completed in
`LQ_signal` on dummy data with CPU fallback because the current environment
reported `GPU available: False` and `Can't initialize NVML`.

## Remaining Gaps

- Full CWRU/XJTU or industrial multi-seed baseline matrix.
- Accepted MoE-specific ablation artifacts: no load-balance, no sparsity,
  router temperature sweep, expert-family removal, and uniform/equal-weight
  router currently have smoke metadata only.
- TOP representative artifacts for Time-MoE/Moirai-MoE/MOMENT/TimeX++.
- Strict local GPU metadata from devices `0,1`.
- SOTA gate.

## Allowed Manuscript Wording

The manuscript may state that bounded route/expert evidence exists and that the
repository now records runnable baseline commands plus non-accepted MoE
ablation smoke surfaces. It must not claim final same-protocol superiority,
TOP-method reproduction, or SOTA.

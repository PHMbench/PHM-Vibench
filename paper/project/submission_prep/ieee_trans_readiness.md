# Paper 02 1D-2D Fusion IEEE Transactions Readiness

Date: 2026-05-11

This checkpoint converts the 1D-2D Fusion evidence plan into a command-bound
baseline and ablation matrix. It does not make the paper submission-ready.

## Current Evidence

- Matrix: `submission_prep/baseline_ablation_matrix.yaml`
- Base config: `configs/vibench/min.yaml`
- Existing gate: `README_T041_SUBMISSION_READINESS.md`
- Evidence level: six PHM-Vibench baseline dummy smokes, a paper-local
  Fusion1D2D dummy demo, and non-accepted fusion-ablation smoke artifacts
- Compute policy: local RTX 4090 GPUs `0,1`; runnable PHM-Vibench commands bind
  `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`

## Dummy-Smoke Summary

The PHM-Vibench proposed proxy, the no-2D proxy, and six model baselines
completed in `LQ_signal` on dummy data with CPU fallback because the current
environment reported `GPU available: False` and `Can't initialize NVML`.

The paper-local `scripts/run_minimal_demo.py` also completed with
`--use_dummy --num_classes=10`, producing
`test_accuracy=0.39` and `test_f1_macro=0.23883535636476813` in `/tmp`.
The same demo failed with `--num_classes=4` because dummy labels include class
IDs outside that range; this is recorded as an entrypoint sanity blocker.

The paper-local demo now also has a current PHM-Vibench HDF5 window loader for
the repository layout under `/home/user/data/PHMbenchdata/PHM-Vibench`. A tiny
CPU smoke on THU_018 loaded 8 HDF5 windows and completed with
`test_accuracy=0.0`, `test_f1_macro=0.0`; this only verifies the real-data
loader path and is not accepted accuracy evidence.

## Ablation Status

Command-bound dummy smokes exist for:

- disabling the PHM-Vibench 2D signal-processing path;
- STFT `n_fft=64`;
- STFT `hop_length=32`;
- fusion type `concat`;
- paper-local class-count sanity.

`scripts/run_fusion_ablation_smoke.py` now emits non-accepted metadata/metrics
for the FFT-only and legacy 1D-only/2D-only/no-statistical ablation surfaces.
The true TSPN_UXFD FFT-only forward path now passes a CPU shape/finite-logit
gate after length-changing signal operators skip incompatible residual addition.
The legacy ablation launcher now delegates to the current-root
`run_ablation_study.py`, uses the vibench min config rather than the removed
unified-baseline path, and restricts GPU binding to local devices `0` or `1`.
These ablations are still not accepted evidence until the true same-protocol
CWRU/XJTU component ablation package exists.

## Remaining Gaps

- IEEE Transactions final figures: the canonical draft now uses `IEEEtran` and
  compiles with placeholder figure boxes, but the accepted architecture and
  Grad-CAM figure artifacts still need to replace placeholders before final
  submission.
- Full CWRU/XJTU multi-seed six-baseline matrix.
- True Fusion1D2D component ablations: 1D-only, 2D-only, no-statistical,
  no-alignment, late/progressive fusion, and no-explainability.
- TOP representative artifacts for TimeMixer, MOMENT, CATCH, and DADA or
  local faithful proxies under the 2x4090 budget.
- Complete strict local GPU metadata from devices `0,1`.
- SOTA gate.

## Allowed Manuscript Wording

The manuscript may state that the repository now exposes runnable dummy-data
entrypoints for the PHM-Vibench comparison surface, the paper-local Fusion1D2D
demo, and non-accepted fusion-ablation smoke surfaces. It must not claim
accepted CWRU/XJTU superiority, final fusion/alignment ablation support,
TOP-method reproduction, GPU feasibility, or SOTA from this checkpoint.

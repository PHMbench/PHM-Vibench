# Status Report: Paper 04 - MOE Explainable FD

Status reports are generated control-plane summaries, not accepted experiment evidence.

- Generated: `2026-05-14`
- Goal file: `paper/UXFD_paper/goal/04_moe_explainable.md`

## Current Verdict

- Submission ready: `False`
- Baselines declared: `6`
- Ablations declared: `6`
- Strict blockers: `5`
- Accepted artifact coverage: `0/14`
- Dirty submodule entries: `2`
- TOP recent-work methods in matrix: `7`
- Has 2026 TOP method: `True`
- TOP binding: `TOP-Q4-TSPULSE` -> `RWTOP2026-TSPULSE`
- TOP evidence ready: `False`
- TOP binding status: `pending_gpu_and_artifacts`

## Strict Blockers

- No accepted CWRU/XJTU or industrial multi-seed baseline table yet.
- Only smoke MoE ablation runner artifacts exist for load-balance, sparsity, temperature, expert-family, and uniform-router surfaces; no accepted same-protocol MoE ablation artifacts exist yet.
- No accepted TOP representative command/log/artifact mapping yet.
- No GPU model/runtime metadata from local GPUs 0,1 yet.
- No SOTA claim is allowed from this matrix alone.

## Next Gate

Do not mark this paper submission-ready until same-protocol accepted baseline, ablation, TOP representative, GPU metadata, and SOTA evidence are present under the artifact gate.

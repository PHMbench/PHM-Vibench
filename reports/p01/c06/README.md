# P01 C06 decisive-pilot result pack

This directory is the reviewable, machine-readable snapshot of the frozen C06
three-seed/two-domain matrix. The scientific engine, configs, 24-row command
matrix, and parser were frozen at
`6dce633d2ae00420053a35aa0b5f6297895447d4` before any C06 process started.

## Execution

- Date: 2026-08-12.
- Hardware: physical GPU 3, exposed as the only visible CUDA device and mapped
  to logical `cuda:0` (`NVIDIA GeForce RTX 4090`).
- Matrix: 8 conditions (M1--M5/C1--C3) x 3 predeclared optimization seeds
  (42, 123, 456) = 24 separate one-iteration processes.
- Result: 24/24 manifests succeeded, 24 checkpoints exist, and all 72 result
  rows were emitted (48 integer-domain rows plus 24 descriptive means).
- Wall clock: 744 seconds for the strictly sequential matrix. No timeout,
  failed cell, replacement seed, rescue run, or protocol change occurred.
- Runtime artifacts remain under `results/p01/c06_*`; those generated trees are
  intentionally ignored by Git. Paths in the snapshot CSVs are repository
  relative. No new digest was created because the C06 protocol forbids it.

## Files

- `c06_condition_domain_matrix.csv`: all 48 planned condition/seed/domain
  cells, including observed status, metric, parameters, supported FLOPs, and
  checkpoint path.
- `c06_paired_contrasts.csv`: all six paired seed/domain observations with the
  exact frozen contrasts.
- `c06_contrast_summary.csv`: per-domain mean, sample SD, range, sign counts,
  and frozen stable-positive decision for every contrast.

## Direct and independent validation

The frozen parser reported a complete valid matrix and selected
`boundary_or_bounded_stop_no_stable_alignment_gain_or_synergy`. A separate
artifact audit recomputed macro-F1 directly from every group-prediction record
with maximum absolute error 0, recomputed all descriptive domain means with
maximum absolute error 0, confirmed all 24 data-protocol JSON objects are
identical, checked all 24 manifests/checkpoints, and observed all 72 required
gradient records as passed. For every seed, the C1 and M4 selected checkpoints
have identical epoch/global-step values and tensor-exact `state_dict` contents.

All six contrasts fail the frozen stable-positive rule. In particular,
alignment gain is +0.211111 in both domains for seed 42, -0.211111 in both
domains for seed 123, and -0.211111/-0.055556 in domains 2/3 for seed 456.
Multimodal synergy is +0.211111/+0.211111 for seed 42 and exactly zero for the
other four seed-domain cells. The favorable single seed is therefore not a
stable alignment result.

C06 supports only a negative performance-route decision. It does not establish
equivalence, mechanism, causality, nuisance suppression, physical-frequency
semantics, independent modalities, or population generalization. C07--C09 are
not admitted by the frozen C06 gate; no outcome-driven tuning or rescue sweep is
allowed.

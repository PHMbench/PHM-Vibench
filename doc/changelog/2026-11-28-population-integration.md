# Preserve declared populations across simulation, generation and target streams

## Problem

Several earlier candidate changes were based on an older runtime. The current
runtime already provides checkpoint prediction, explicit external Factory modules
and CFM/GAN/VAE/DDPM tasks. Replacing those owners with old branch files would lose
working behavior. The remaining failures concerned frozen population selection,
sample coordinates, generation counts and target-only stream initialization.

## Changes

- The public split schema accepts the existing `manifest_mode: read_only`,
  `partition_map` and optional `manifest_sha256` fields. Generative tasks can use
  the existing grouped partition implementation. No new manifest format is added.
- Flattened datasets retain both `file_id` and the per-file `window_id`, allowing
  exported predictions to resolve their original raw intervals.
- `scripts/prepare_simulated_phm.py --profile bearing4` exposes the existing
  uncalibrated impact/resonance simulator. The legacy signal mode and its optional
  CFM partition export remain available. Existing output directories are refused.
- Pipeline 06 rejects a condition count different from the requested sample
  count. Model construction and metadata-derived sampling conditions use training
  records. Explicit validation/test references consume their complete populations;
  test references require grouped metadata. The old bounded training-reference
  smoke remains available.
- The existing research entry accepts `TTA/stream` using the ordinary Data
  Factory readers and caches. It selects declared target domains and optional
  assets before waveform access, preserves complete tail batches, and supports
  frozen inference, BN-only statistics and Tent. Known-domain resets restore
  model, optimizer and random states without erasing full-stream budgets.
- Restricted caches do not copy excluded source/target records from prior cache
  contents. The existing source-only and prepared-H5 paths remain supported.

The older NPZ target-stream entry is not introduced. Existing checkpoint prediction
and generative algorithms are reused; no alternate prediction Runner or training
framework is added.

## Bounded execution evidence

The full bearing4 corpus was generated with data seed 2026: 288 complete records,
four classes, twelve simulated instances and six speed profiles. The unchanged
CSV reader matched all records' selected channel against independent float32
loading. The 48-record pilot is an identical subset. These are simulated instances,
not calibrated physical devices.

A caller-owned source view used 72 training and 24 validation records, restricted
to the three source conditions. Two CPU epochs with seed 17 produced selected
linear and small ResNet checkpoints. Public checkpoint-only prediction covered
all 384 windows of 48 independently assigned evaluation records. Coordinate
coverage and independent metric/logit recomputation passed; training caches held
only the 96 source records. Study configuration and outputs remain outside this
upstream repository.

Four public target-stream executions reused the same source-selected ResNet on
192 windows from 24 target records. Frozen/BN-only/Tent/domain-reset Tent retained
the entire population, with respectively 0/0/12/12 updates and 0/0/0/2 resets.
Permuting evaluator labels left persistent Tent's predictions and final algorithm
and random states unchanged.

Focused checks cover frozen splits through the public schema, population
coordinates, complete generation references, existing GAN/VAE/DDPM behavior,
target filtering, source cache exclusion, BN-only behavior and episodic reset.
The PR records exact local and CI outcomes separately.

## Limits and rollback

Single-seed bounded executions establish software and protocol behavior, not
comparative diagnostic or generation efficacy. Group-averaged window metrics do
not become record-level majority-vote diagnostics. No support registry is promoted,
real-data release qualification is unchanged, and no original candidate PR or
main/dev branch is merged. Revert the affected topic commits to roll back code;
preserve all existing data and results.

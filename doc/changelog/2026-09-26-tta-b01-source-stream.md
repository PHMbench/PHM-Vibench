# 2026-09-26 — B01 frozen source-only stream

## Change

Add `phmfactory.source_stream.run_source_only_stream`: a single x-only, closed-set
classification pass over an existing ordered target DataLoader. Explicit source weights
are loaded strictly using the existing Model Factory loader; the caller explicitly owns
eval mode and device. Every registered parameter and buffer must remain unchanged.
Predictions precede the caller's evaluator, which may consume labels and traceability.
The existing metric accumulators and result summary writer remain authoritative.

Base: dev `2ecadd8451e27acbb3077ab3bbff6a0898fb58b6`, already containing B00 merge
`1902d839c3ae8a6d7f72ec591983f698eb12d942`. Unlike the earlier handoff, merged B00 rejects
source-only execution in its generic protocol helper. That fail-closed boundary remains.

The new function accepts synchronous sequential or native unshuffled complete evaluation
samplers. It neither builds datasets nor changes their order, batch partitions, labels,
windowing, normalization, channels or device. TimesNet parity requires the same batch
partitions because its frequency selection includes a batch aggregate.

## Validation

Actual local source snapshot, Python 3.13.5 / torch 2.10.0+cpu:

- `test/test_tta_source_stream.py`: 47 passed; actual PyTorch, Data Factory, Model Factory,
  native evaluation sampler, GlobalAverageLinear and TimesNet; no mocked runtime imports.
- Together with unchanged B00 contract and label-isolation tests: 108 passed in 7.74 s.
  These totals overlap; they are not 155 independent tests.
- `python -m scripts.validate_docs`: passed, 94 maintained files scanned before this
  changelog was added.
- `python -m scripts.validate_configs`: passed, 13/13.
- Config Atlas regeneration: unchanged. `git diff --check`: passed.

Raw/Lightning-prefixed source fixtures, BatchNorm/Dropout state preservation,
true/permuted/zero labels, sample order/tails, native missing/duplicate sample rejection,
unsupported protocols, mutated states, original forward failures, non-finite outputs and
population-level macro-F1 are exercised. The factory tests use shipped Dummy inputs and
explicit synthetic source checkpoints; they do not train or access industrial data.

The existing public-package workflow now runs the same B01 suite from the installed
wheel outside the checkout and rejects skipped cases. CI and independent review are
separate merge gates; local tests do not substitute for them.

A temporary read-only Actions transport workflow supplied the complete source snapshot
because the local container could not resolve GitHub. It is removed from the final tree.
No transport workflow, new dependency, benchmark result or temporary data cache is shipped.

## Compatibility and limits

Existing CLI, TTA protocol fragment, Task registration, Trainer, factories and result
formats are unchanged. The new callable is a bounded Python runtime, not a new runnable
TTA YAML or a replacement training/evaluation pipeline. There is no Tent/SAR/EATA/CoTTA,
EMA, replay buffer, SFDA, target tuning, reset or adaptation-resume implementation.

Exact state checking costs one model-state copy and comparison work per batch. Arbitrary
unregistered Python state, malicious callbacks and stochastic preprocessing are not
sandboxed. Partial metrics must be discarded after an exception. No frozen-source or
adaptation PHM utility, latency advantage, or benchmark-ready promotion is claimed.

Merge only after exact-head CI and independent review; then stop B01. Revert this bounded
change to roll back without altering B00 or unrelated PRs.

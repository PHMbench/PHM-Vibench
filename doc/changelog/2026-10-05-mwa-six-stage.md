# Explicit six-stage MWA-CNN comparator

`X_model/MWA_CNN` now accepts `depth: 6`, extending the existing four-stage
architecture with the missing two wavelet/attention stages. Fixed db16 filters,
width 12, GroupNorm(6), attention and dropout 0.1 match the authors' six-stage
architecture. The old four-stage default and module names remain available for
existing configurations and checkpoints; it is not relabeled as MWA-CNN-6.

The constructor no longer places wavelet filters on CUDA. The unused `ptwt` and
inverse-wavelet imports are removed, and the optional package extra `mwa` declares
the actual wavelet dependencies. Input layout, channel count, minimum measured
support and the attention BatchNorm training-batch requirement fail explicitly.

Focused tests cover published architecture dimensions, independent db16 coefficient
values, forward/backward behavior, optimizer updates and strict checkpoint recovery
on synthetic CPU inputs. These checks do not establish industrial or CUDA results.
The model guide records the official recipe and why target-based checkpoint
selection from its notebook is not part of the DG execution path.

## Physical-condition DG integration

The existing P01 runner now validates documented physical condition tuples and
specimen identities before development. Source-only class support is checked
without opening held-out labels or waveforms; a missing held-out source control
is reported explicitly while the complete target population remains required.
Domain labels and repeated acquisitions cannot establish independent specimens.

Baseline-first execution uses a common source-calibrated update cap, at most
12 configurations per family, and per-condition qualification of the selected
reference and every final baseline seed. Recent source improvement is checked
again on those actual fits. Formal DG gives the reference the same paired
augmentation as the other systems while retaining its ordinary CE objective.
Failed candidates, failed qualifications and budget insufficiency remain visible.

Frozen bundles must reproduce source predictions in a separate process before
target release. The existing evaluator retains full target results and adds
descriptive within-specimen source/target contrasts with fixed equal class
weights, paired specimen uncertainty and explicit unavailable states. A dense
readout matched by nearest parameter count complements the existing mechanism
controls. No router, target adaptation, new industrial evidence or dataset
qualification is introduced by these software changes.

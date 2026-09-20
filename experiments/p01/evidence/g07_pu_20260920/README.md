# G07 PU frozen-output controls — post_hoc=true

This is the final new exploratory analysis of the existing seven PU permanent-test
bearings. It does not change D1 labels, splits, checkpoints, predictions, K4,
temperature, alpha, or decisions. These test bearings are now sealed against
further model/loss/operator/routing/calibration/seed/baseline/decision design.
No manuscript conclusion was changed by this execution.

## Execution and validity gate

Paper authority: `5e52b3710715c545b4493064a502d2b673536c2b` (G07).
Original Benchmark pin: `32f223b59ef33eb54baaffeb4af5a8c3595d5240`.
Actual G07 analysis implementation: `1d5e405735dc4b6d47685cc075cd8053e6f3b84d` (retained in Git history).
Original frozen export implementation: `72874750be3e53cf8f78f27e1991ed4749c9a4f9`.
All local analysis used the existing LQ_signal Python, CPU, empty CUDA visibility
and one BLAS/OpenMP thread. The run took 22.781 seconds and exited 0. No model,
torch, or h5py was imported. The entrypoint rejected H5/checkpoint opens and
writes in the original run root. No training or inference was performed.

The immutable input was the original workspace's `results/d1_pu_20260919/`.
Saved absolute paths were explicitly rebound from the known original temporary
run root to this retained copy; no alternative run was searched for or substituted.
Private outputs are in the new sibling `results/g07_frozen_controls_20260920/`.

Step A: **pass_with_limitations**. Of 94 recorded checks, 93 passed and one is
unresolved but non-blocking. Class order/identity label mapping, 64 kHz sample
rate, vibration channel 2, LC/squeeze-axis-2 observation, two 8192-point start/end
windows, no added normalization, common p0 with temperature 1, physical split,
development history and frozen descriptors agree. Original selected source
predictions equal the frozen source predictions exactly. Raw probabilities and
stable log probabilities agree exactly across all MLP16/O seeds within each
partition and across the saved K4 assessment outputs. All saved source restore
vectors were checked again without restoring or loading a model.

The unresolved item is the absence of a standalone before/after-restore
parameter/buffer tensor dump. It is **not** reported as a newly passed tensor
comparison. Existing execution records instead support strict state loading,
reference-state equality after training (including buffers), per-inference full
state_dict invariance/eval guards at the recorded export revision, and exact
separate-process source prediction equality. No contradictory restore evidence
or known semantic mismatch was found. This supports functional frozen-reference
semantics, not proof that specimen shift is the unique cause of degradation.

## Frozen-output sanity

All summaries use original labels `[healthy, IR-dominant, OR-dominant]`.
Counts below are acquisition counts; reported metrics weight physical bearings
equally within conditions, then conditions equally. Entropy is the similarly
weighted mean **window** predictive entropy in nats, not a calibration metric.

| p0 population | Brier | Accuracy | Entropy | True support | Predicted histogram |
| --- | ---: | ---: | ---: | --- | --- |
| Source selection | 0.534376 | 0.600000 | 0.737482 | [60,120,120] | [0,120,180] |
| Held-out source settings | 0.871446 | 0 | 0.878872 | [60,180,179] | [95,180,144] |
| Unseen 900 rpm | 0.963631 | 0 | 0.943612 | [20,60,60] | [76,23,41] |

p0 confusion matrices (rows=true, columns=predicted) are respectively
`[[0,60,0],[0,60,60],[0,0,120]]`,
`[[0,60,0],[36,0,144],[59,120,0]]`, and
`[[0,20,0],[19,0,41],[57,3,0]]`.
The held-out outputs are entirely off diagonal, but are not one deterministic
global class permutation. No relabelled or "corrected" scores were computed.
MLP16 seed42 always predicts class 1 in source selection and held-out source
settings. Other seed/condition histograms and all p0/MLP16/O summaries are in
[reference_validity.csv](reference_validity.csv). That file includes all
protocol checks, confusion matrices, support, Brier/CE/accuracy/macro-F1, entropy
and probability means, but excludes private per-bearing rows.

## Fixed constants and comparisons

[source_constants.json](source_constants.json) contains the seven source means
and the shared prior `pi_S=[0.2,0.4,0.4]` (up to floating-point arithmetic).
p0 is one fixed predictor, not three replicates. MLP16/O each have a separate
constant for each of seeds 42/123/456. All conditions use the same source-fixed
constant for that predictor. No test labels determine a constant. None of the
actual constants has a zero component; the implementation does not clip zeros.

The following Brier contrasts are fixed-p0 values or finite three-seed means.
Intervals are 2,000-draw paired global-bearing, condition-incidence-stratified,
post-hoc descriptive 95% intervals (seed 20260919). All predictors/seeds/conditions
share a draw's bearing multiplicities. Constants remain fixed. Intervals exclude
source-constant estimation, reference training and model-selection uncertainty.

| Population / predictor | R(q)-R(m_q) [interval] | R(q)-R(pi_S) [interval] |
| --- | --- | --- |
| Held-out source / p0 | +0.245861 [0.115937,0.378812] | +0.254303 [0.170100,0.354531] |
| Held-out source / MLP16 | +0.016028 [-0.005086,0.039526] | +0.015527 [-0.005407,0.030943] |
| Held-out source / O | +0.071709 [0.007061,0.150116] | +0.068595 [0.017460,0.140599] |
| Unseen / p0 | +0.338047 [0.141157,0.500937] | +0.346489 [0.266402,0.433687] |
| Unseen / MLP16 | +0.044563 [0.006143,0.083604] | +0.044062 [0.020141,0.063762] |
| Unseen / O | +0.203497 [0.057843,0.343455] | +0.200382 [0.071261,0.329515] |

Source-selection contrasts are reported separately in the CSV and are in-sample
for constant estimation, not independent effects. Per-seed values and sample SD
remain in [input_dependence_contrasts.csv](input_dependence_contrasts.csv).
Absolute metrics for all three populations, predictors and constants are in
[constant_controls_metrics.csv](constant_controls_metrics.csv).

**G07 Case B:** the current D1 does not establish the best candidate's
sample-dependent utility. MLP16's held-out-source intervals cross zero (unresolved,
not equivalence); unseen-condition contrasts favor its constants. This is not a
claim about Bayes information, representation quality, or absence of signal
information. No next experiment or model change is selected here.

## Pipeline accounting and verification

The original Delta_pipe and its original per-seed paired intervals were recovered
from the saved arrays. For finite seed means:

- Held-out source: `0.053067364629 = 0.071709298750 - 0.016028062464 - 0.002613871657`.
- Unseen: `0.156320252716 = 0.203496637923 - 0.044562513550 - 0.002613871657`.

Terms are O input dependence, minus MLP16 input dependence, plus the source-mean
level difference. Reconstruction residual is zero for these means; the complete
paired-draw identity is checked to 1e-12. These are accounting identities, not
causal decompositions. [pipeline_decomposition.csv](pipeline_decomposition.csv)
retains all terms, seed SD and shared-bootstrap intervals.

An independent nested-loop recomputation of MLP16 seed42 unseen Delta_input gave
`0.03766426412169112`, versus CSV `0.037664264121691104`. Focused constructed-array
and existing offline-analysis tests passed (28 tests before the additional
protected-open regression); no test fixture trained or inferred a model.
The final G07-only test run passed all 10 tests, including the protected-open
regression. Documentation validation also passed. These overlapping scopes are
not counted as separate independent validations of scientific performance.
The real G07 run has no failure record because it completed without an execution
failure; the unresolved restore-dump limitation is retained in the validity CSV.

## Public/private boundary

This public directory contains aggregates only. The public validity CSV removes
all per-bearing rows and the bearing_id column; the full private CSV retains
per-bearing Brier and acquisition accuracy. Source input locations are relative
to the known private run root here; the private JSON retains exact paths.
Other metric values are unchanged; CSV line endings may be normalized for Git.
Private predictions, identifiers, original metadata, checkpoint files and H5
are not redistributed. Command/stdout/stderr and exit records remain beside the
private outputs. This completion formally seals the same seven test bearings.

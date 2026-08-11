# g05_cross_dataset — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Cross-dataset portability, selected-population conservation, and reader/model boundary truth
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific contract
- P0: impossible target-domain requests are silently reduced.
- Cross-dataset claims are invalid until each dataset has explicit source/target domains and evaluation unit.
- Correction: fail closed and freeze one candidate at a time.

## R02 — Config/runtime
- P0: missing metadata/system facts can be skipped by the sampler rather than rejected.
- P0: explicit configuration may receive hidden defaults.
- Correction: a portable experiment must be fully declared before Data/Model construction.

## R03 — Reader/metadata
- P0: cache reuse depends on IDs, not reader semantics.
- Representative risks: MFPT single-channel MAT, SEU multichannel CSV, PU vibration/current MAT.
- Correction: reader output must be explicit `[L,C]`; compare selected columns, order, dtype, delimiter, and reader name directly.

## R04 — Split/sampling
- P0: `drop_last=True` can remove every batch from a small system.
- Impact: selected systems differ from trained systems.
- Correction: verify system/file coverage after sampler construction; do not build a universal balancing framework.

## R05 — Model/shape
- P0: HSE repeats channels/time to fit patches.
- Cross-dataset impact: channel count differences are hidden instead of exposed.
- Correction: fail on shape mismatch; use GlobalAverageLinear first to separate data defects from HSE defects.

## R06 — Task/loss
- P0: first parameter can be omitted from regularization.
- Cross-dataset impact: comparisons may optimize different effective objectives.
- Correction: objective participation tests must be dataset-independent and exact.

## R07 — Trainer/evaluation
- P0: Pipeline 02 can report stage success with empty metrics.
- Cross-dataset impact: one dataset may fail evaluation while the orchestration result appears complete.
- Correction: stage success requires evaluation success and finite non-empty metrics.

## R08 — Decoupling
- P0: cache semantics and selected-ID coverage are the real data replacement blockers.
- Existing CSV extension proves compatible datasets can reuse Data Factory/DG/sampler/loader seams.
- Decision: no ReaderPlugin, no DataProvider hierarchy, no second registry.

## R09 — User experience
- P1: user cannot easily distinguish current metrics from governance artifacts.
- Cross-dataset runs need a single result directory and explicit dataset/config identity in plain text.
- Correction: expose reader, selected files, split counts, checkpoint, and metrics—not hashes.

## R10 — Data Factory
- P0: sampler silently skips missing metadata samples.
- Required conservation:
  `selected IDs = materialized IDs = dataset IDs = loader-represented IDs`.
- Correction: verify equality and fail with missing IDs; do not silently filter.

## R11 — Model Factory
- P0: generic reader output is expanded to rank 3 and later flattened, hiding shape ownership.
- Correction: ordinary signal reader contract is exactly `[L,C]`; Model validates `input_dim == C`.
- Do not let Model Factory repair channels or infer modality.

## R12 — Task Factory
- P0: metric dataset name is taken only from the first file ID in a batch.
- Cross-dataset impact: mixed-dataset metrics may be assigned to the wrong dataset.
- Correction: enforce dataset-homogeneous metric batches or use a task that explicitly supports mixed datasets.

## R13 — Trainer Factory
- P0: Pipeline 02 wrong-success remains independent of dataset.
- Trainer must not alter device/training budget between datasets.
- Correction: exact same lifecycle across MFPT, SEU, PU, and later CWRU.

## R14 — Group meta-review
- Highest cross-dataset blocker: cache reuse does not encode reader semantics.
- Recommended validation ladder:
  1. Dummy/CSV fixture;
  2. MFPT + GlobalAverageLinear;
  3. SEU condition DG;
  4. PU multichannel classification;
  5. CWRU local acceptance.
- Rejected: ingesting every maintained reader before fixing core semantics.

# Candidate roles

| Dataset | Primary purpose |
|---|---|
| Dummy/CSV | repo-local semantic and 2×2 Data×Model tests |
| MFPT | minimal single-channel real-data portability |
| SEU | multichannel, bearing/gear, operating-condition DG |
| PU | vibration/current multichannel compatibility |
| CWRU | final local baseline-valid acceptance |

# Stop condition

Do not add another dataset until failure can be localized to reader/metadata, split, model, task, trainer, or estimator. Dataset count is not a success metric; semantic closure and local replacement cost are.

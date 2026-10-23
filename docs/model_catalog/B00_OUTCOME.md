# B00 outcome — deconstruct PR #266

## Fact

PR #266 accumulated 21 commits and 48 changed files across Model, Task, Data adapter,
generic regularization, configs, tests, docs and CI. It is too broad to serve as one
implementation invariant.

## Deconstruction result

Every changed path is assigned exactly one D0–D4 class, one logical owner and a decision in
[PR266_DIFF_INVENTORY.csv](PR266_DIFF_INVENTORY.csv). B00 creates no replacement
implementation PR, so every `future_pr` is explicitly `NOT_CREATED` rather than a
guessed number.

Counts:

```text
D0 DISCARD_FROM_CATALOG_SPLIT : 1
D1 SHARED_PREREQUISITE       : 2
D2 TASK_CONTRACT             : 2
D3 MODEL_SPECIFIC            : 15
D4 TEST_CONFIG_DOC           : 28
TOTAL                         : 48
```

The 15 model implementations are evaluated at candidate level in
[PR266_MODEL_AUDIT.csv](PR266_MODEL_AUDIT.csv). None is promoted to QUEUED or VERIFIED by
B00. Fourteen map to an original source row and remain audit candidates; TSMixer has no row
in the preserved 187-source workbook and cannot be salvaged without a new scientific/source
audit.

## Key findings

1. Forecasting is a Task change, not just model code. The Data adapter selection and
   `DG.point_forecasting` must be reviewed as an S2 Task contract before any forecast
   model can be adopted.
2. `_native_forecasting.py` imports the classification helper, so the mega branch creates
   an unnecessary cross-task shared dependency.
3. PAttn imports `InvertedEncoderLayer` from the iTransformer model file, so its diff has
   a hidden cross-model dependency and is not independently replayable.
4. The complex-parameter regularization patch is an independent generic correctness change,
   not a model-catalog prerequisite. It needs its own current-dev counterexample if pursued.
5. The aggregate workflow, bulk registry patch, public test parameter list and branch-level
   docs should not be replayed wholesale.
6. Multiple ports cite a different upstream route/task variant from the original workbook.
   This is an E0/E2 audit issue; sunk implementation work is not evidence for adoption.

## Salvage terminology

`SALVAGE_CANDIDATE_AUDIT_REQUIRED` means only that the code may be useful source material.
It does **not** mean approved, QUEUED, integrated, supported or benchmark-ready.

`DO_NOT_REPLAY_AS_IS` means structural coupling must be removed before the candidate could
be considered as a bounded implementation.

`DISCARD...` means the mega-PR change is not carried forward. It does not assert that the
underlying scientific method can never be revisited.

## Supersession decision

The durable B00 decision is that PR #266 is **not** an implementation authority and must
not be merged as a whole. Its retained source material is mapped to logical owners in the
inventory and dependency map; future implementation PRs are created only after current
scientific selection and source/fidelity audit.

Closing/branch-retention mechanics are operational PR state and are intentionally not
encoded here as a future command queue. The source branch should remain available while
selected material has not yet been independently reconstructed or explicitly rejected.

B00 adds no runtime code and does not start B01.

# B00 outcome — deconstruct PR #266

## Fact

PR #266 accumulated 21 commits and 48 changed files across Model, Task, Data adapter,
generic regularization, configs, tests, docs and CI. It is too broad to merge as one
implementation invariant.

## Deconstruction result

Every changed path is assigned exactly one D0–D4 class and logical owner in
[PR266_DIFF_INVENTORY.csv](PR266_DIFF_INVENTORY.csv).

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
in the preserved 187-source workbook and is not salvageable without a new scientific/source
audit.

## Key findings

1. Forecasting is a Task change, not just model code. The new Data adapter selection and
   `DG.point_forecasting` must be reviewed as an S2 Task contract before any forecast
   model can be adopted.
2. `_native_forecasting.py` imports the classification helper, so the mega branch creates
   an unnecessary cross-task shared dependency.
3. PAttn imports `InvertedEncoderLayer` from the iTransformer model file, so its diff has
   a hidden cross-model dependency.
4. The complex-parameter regularization patch is an independent generic correctness change,
   not a model-catalog prerequisite. It needs its own current-dev counterexample if pursued.
5. The aggregate workflow, registry patch, public test parameter list and branch-level docs
   should not be replayed wholesale.
6. Multiple PR ports cite a different upstream route/task variant from the original workbook.
   This is an E0/E2 audit issue, not something to resolve by preserving sunk implementation work.

## Salvage terminology

`SALVAGE_CANDIDATE_AUDIT_REQUIRED` means only that code may be useful source material.
It does **not** mean approved, QUEUED, integrated, supported or benchmark-ready.

`DO_NOT_REPLAY_AS_IS` means the diff contains structural coupling that must be removed
before the candidate could even be evaluated as a bounded PR.

`DISCARD...` means the mega-PR change is not carried forward; it does not assert that the
underlying scientific method can never be revisited.

## Supersession

After this B00 documentation PR is independently reviewed and merged, PR #266 can be closed
as superseded with a pointer to this inventory and dependency map. Do not merge #266.
Keep its source branch until any actually selected candidate has been independently
reconstructed/replayed or explicitly rejected.

B00 adds no runtime code and does not start B01.

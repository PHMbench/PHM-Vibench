# B00 — PR #266 dependency map

Source reviewed: PR #266 at `c31c888ec85c18c3db2f2cd2af89c5ec3e840867`.
Baseline for this deconstruction: `dev@7b1512b94175539d3db26119a090df8d7b795160`.

This document does **not** approve or replay any implementation. It identifies which parts
of the mega PR can be independently re-audited and which are coupled.

## 1. Actual logical graph

### S1 supervised classification

```text
existing DG.classification
        |
        +-- candidate source:M016 ModernTCN
        +-- candidate source:M019 TSLANet
        +-- candidate source:M187 TimesNet
        +-- candidate source:M022 iTransformer
                 |
                 +-- PR implementation is imported by PAttn forecasting
```

All four PR implementations also import `_native_classification.py`. That helper is not
an approved prerequisite by itself. Under B−1 it should only become shared code after two
or more independently adopted maintained consumers exist; otherwise the small validation
logic belongs with the selected model.

The iTransformer → PAttn edge is stronger: PAttn imports the model file's
`InvertedEncoderLayer`. A future PAttn PR therefore **cannot** be replayed independently
from #266 as written. If both candidates are scientifically selected, the shared attention
primitive must get a neutral owner without making one model depend on another.

### S2 deterministic point forecasting

```text
existing DG Classification_dataset
        |
        +-- adapters.py: (DG, point_forecasting)
        |
        +-- new DG.point_forecasting Task
                |
                +-- forecast candidates
                    NLinear / SparseTSF / FITS / SegRNN
                    SOFTS / FreTS / N-HiTS / LightTS
                    PAttn / TimeMixer
                    TSMixer (not in the 187-row source catalogue)
```

The Task changes scientific semantics: a full raw window is split into history and future
target, normalization/noise restrictions are imposed, MSE owns the objective, and the
model sees history only. It therefore belongs to a **Task-contract PR**, not inside a model
PR. No forecast candidate may be replayed before that contract is independently accepted.

All forecast ports import `_native_forecasting.py`, which itself imports validation
helpers from `_native_classification.py`. This classification→forecasting dependency is
an artifact of the mega PR, not a desired architecture. Do not replay it as-is. A neutral
helper is justified only after multiple approved consumers actually need the duplicate
logic.

PAttn adds an additional cross-model dependency on iTransformer, so its current diff is
not replayable as an isolated model even after the Task exists.

## 2. Independent change that does not belong to model adoption

`src/task_factory/Components/regularization.py` changes generic L2 semantics for complex
parameters. The model integration config does not establish that this generic change is a
required S1/S2 prerequisite. It must leave the model-catalogue replay. Only a current-dev
counterexample can justify a separate bounded correctness PR.

## 3. Tests/config/docs follow their owner

The 15 YAMLs, seven test modules, two aggregate model docs, two branch changelogs, the
bulk registry rows and the aggregate workflow are D4 evidence/support material. They are
not independent prerequisites.

Rules for replay:
- selected candidate config/tests travel with that candidate;
- Task-only counterexamples travel with the Task PR;
- shared public-path fixtures may be reused only after there are real bounded consumers;
- aggregated "all native models" CI is not replayed wholesale;
- registry entries are created only after the corresponding candidate is verified;
- old branch-progress changelogs are replaced by factual bounded changelogs.

## 4. Ordering constraints

Only the following partial ordering is justified:

```text
B−1 scientific rules (already in dev)
        |
        +-- S1 candidate audit -> one selected model PR at a time
        |
        +-- S2 Task audit/contract
              -> merge Task contract
              -> S2 candidate audit
              -> one selected model PR at a time
```

There is no justified order such as “replay all 15 models”. Candidate selection remains
coverage-driven.

## 5. Branch retention

#266 is superseded as an implementation authority after this B00 mapping is merged.
Its source branch should remain available while selected changes have not yet been replayed,
so closing the PR must not be confused with deleting source history.

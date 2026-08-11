# PHMBench 140-Review Program — 2026-08-11

## Scope

```text
Repository: PHMbench/PHM-Vibench
Authority reviewed: dev@7b604a06802b2053611430916d278ee807c6d772
Review branch: reviews/2026-08-11
Review population: 10 groups × 14 reviewers = 140 reviews
Source modifications: none
```

This branch contains review material only. It does not modify runtime, Factory, model, task, trainer, configuration, tests, CI, or release state.

## Storage

The 140 reviews are stored as **14 independently indexed reviewer sections inside each of 10 group dossiers**. This keeps the branch navigable while preserving all 140 reviewer identities. Use [REVIEW_INDEX.md](REVIEW_INDEX.md) to navigate every review directly.

The cross-group synthesis is [R00_CROSS_GROUP_META_REVIEW.md](R00_CROSS_GROUP_META_REVIEW.md).

## Groups

| Group | Primary attack surface | Dossier |
|---|---|---|
| G01 | Scientific validity and claim closure | [g01](groups/g01_scientific_validity.md) |
| G02 | Fail-fast and silent-fallback rejection | [g02](groups/g02_fail_fast.md) |
| G03 | User experience and result discoverability | [g03](groups/g03_user_experience.md) |
| G04 | Factory responsibility and module decoupling | [g04](groups/g04_factory_decoupling.md) |
| G05 | Cross-dataset portability | [g05](groups/g05_cross_dataset.md) |
| G06 | Determinism and numerical truth | [g06](groups/g06_determinism_numerics.md) |
| G07 | Estimator and statistical claim semantics | [g07](groups/g07_estimator_statistics.md) |
| G08 | Lifecycle, checkpoint, evaluation and resources | [g08](groups/g08_lifecycle_resources.md) |
| G09 | Research maturity and support boundaries | [g09](groups/g09_research_maturity.md) |
| G10 | Adversarial rejection and critical-path prioritization | [g10](groups/g10_adversarial_prioritization.md) |

Each group contains the same 14 orthogonal reviewer roles:

```text
R01 Scientific contract / claim alignment
R02 Config authority / runtime dispatch
R03 Reader / metadata / raw-to-tensor
R04 Split / sampling / transform / leakage
R05 Model / device / shape / determinism
R06 Task / loss / metric / estimator
R07 Trainer / checkpoint / evaluation lifecycle
R08 Module decoupling / replaceability
R09 User experience / CLI / results
R10 Data Factory
R11 Model Factory
R12 Task Factory
R13 Trainer Factory
R14 Group meta-review / responsibility arbitration
```

## Hard constraints applied by all reviewers

```text
No new hash / checksum / digest / hash chain
No receipt / ledger / artifact integrity auditing
No hypothetical-corner-case architecture
Fail fast; no silent fallback or hidden semantic repair
No new universal factory / wrapper / manager / adapter / registry / schema hierarchy
Tests protect scientific semantics and user main paths, not coverage numbers
Formatting, CI quantity and directory audits are not primary success criteria
```

## Cross-group verdict

```text
REQUEST_CHANGES
```

The strongest consensus is not “add more Factory abstraction.” It is:

1. close wrong-success and requested-versus-executed mismatches;
2. establish one device authority;
3. make HSE evaluation deterministic and reject shape repair;
4. make data population, labels, split and cache semantics explicit;
5. freeze an exact estimator;
6. promote one real ordinary-classification experiment to `baseline_valid`;
7. only then reopen ProtoNet, GFS, generative, multitask and foundation-model work.

## Review limitations

The review used GitHub code, config, document, test and PR inspection. External-data numerical experiments were not executed in the connector environment. MFPT, SEU, PU and CWRU runtime results therefore remain explicitly unverified and should be performed by a local Agent against the same reviewed baseline or a reviewed successor commit.

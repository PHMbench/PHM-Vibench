# g04_factory_decoupling — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Factory responsibility, local replacement cost, and stopping unnecessary abstraction
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific contract
- P0: impossible `target_domain_num` is clamped or becomes train-only.
- Impact: configured DG/CDDG protocol is rewritten.
- Correction: fail at ID-selection boundary; do not build a ProtocolManager.

## R02 — Config/runtime
- P0: explicit YAML without `pipeline` becomes Pipeline 01.
- P0: Pipeline 02 fills missing trainer semantics.
- Correction: explicit configuration must be complete; runtime must not repair it.

## R03 — Data/metadata
- P0: cache reuse checks IDs but not reader columns, dtype, delimiter, or channel order.
- Impact: stale tensors can be reused under a new experiment config.
- Correction: directly compare the few reader-relevant fields; no digest or cache schema framework.

## R04 — Split/protocol
- P0: impossible target-domain requests are silently reduced.
- P0: unknown task types can produce train=test=all IDs.
- Correction: protocol construction must either satisfy the request or fail.

## R05 — Model/device
- P0: HSE repeats time/channel dimensions when patches exceed the input.
- Impact: model changes physical signal semantics.
- Correction: reject incompatible shape; keep existing model seam.

## R06 — Task/loss
- P0: regularization consumes the first parameter generator element while probing device.
- Impact: configured regularization excludes one trainable parameter.
- Correction: materialize trainable parameters once and calculate on the complete list.

## R07 — Trainer/lifecycle
- P0: Pipeline 02 swallows `trainer.test()` exceptions and returns empty metrics.
- Correction: exception propagation, non-empty metrics, cleanup in `finally`.
- Out of scope: StageManager, ResultEnvelope, lifecycle registry.

## R08 — Replaceability
- P1: shared classification runtime special-cases Multitask and mutates Data/Model config.
- Existing evidence: CSV dataset extension and GlobalAverageLinear model extension already work through current seams.
- Decision: move task-specific mutation out of the shared spine; `STOP ABSTRACTING` after that.

## R09 — User experience
- P1: metrics, run attestation, artifacts manifest, and evidence registry create multiple result authorities.
- Correction: checkpoint + metrics + logs + minimal run status.
- Do not create another results manager.

## R10 — Data Factory
- P0: cache identity is weaker than reader semantics.
- Data Factory should own explicit data selection, reader, dataset, sampler, loader—nothing else.
- Do not introduce DataProvider/ReaderPlugin; current compatible dataset seam is proven.

## R11 — Model Factory
- P0: Trainer converts any non-CPU device request to `auto`; Task also calls `.cuda()`.
- Model Factory should resolve identity, construct, and explicitly load weights only.
- Existing `GlobalAverageLinear` replacement test proves a second model plugin layer is unnecessary.

## R12 — Task Factory
- P0: `Default_task` silently adds `task_id=classification`.
- Impact: wrong or incomplete batch semantics are rewritten.
- Correction: explicit task identity or a classification-only contract without that key.

## R13 — Trainer Factory
- P0: explicit device is not authoritative.
- Trainer Factory should be the sole device owner and should not fill epochs/gpus/pruning.
- Default audit callback should become explicit opt-in or be removed.

## R14 — Group meta-review
- PR #148 is governance-heavy, stale relative to current `dev`, and not required to prove the existing data seam.
- Accepted critical path: config fail-fast → protocol fail-fast → device authority → Pipeline 02 evaluation truth → one baseline-valid vertical slice.
- Rejected: FactoryManager, UniversalContext, second registry, capability graph, schema hierarchy.

# Factory responsibility decision

| Concern | Final owner |
|---|---|
| reader, selected IDs, split materialization, loaders | Data Factory |
| model identity, construction, explicit external weights | Model Factory |
| task identity, loss, metric lifecycle | Task Factory |
| device, checkpoint callbacks, logger lifecycle | Trainer Factory |
| orchestration and success gating | Runtime/Pipeline |

# Stop condition

A replacement is considered decoupled when changing Data or Model requires only that module plus config, and incompatibility fails at the nearest boundary. No new abstraction is authorized without two real maintained consumers and a demonstrated net reduction in complexity.

# g08_lifecycle_resources — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Fit/checkpoint/test lifecycle, resource cleanup, and wrong-success elimination
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific lifecycle
- P0: Pipeline 02 can complete training, fail evaluation, and still return stage success.
- A scientifically successful stage requires fit, valid checkpoint, evaluation and finite metrics.
- Correction: make stage status derived from the entire lifecycle.

## R02 — Config/runtime lifecycle
- P0: explicit YAML can receive Pipeline 01 implicitly.
- P0: stage trainer config is repaired by the orchestrator.
- Correction: configuration must be complete before lifecycle starts.

## R03 — Data lifecycle
- P0: cache semantics can outlive reader semantics.
- P1: data resources in Pipeline 02 are not closed through a guaranteed `finally` boundary.
- Correction: explicit cache invalidation and deterministic close.

## R04 — Protocol lifecycle
- P0: impossible domain requests are changed rather than rejected.
- Checkpoint selection is invalid if target data can enter validation or if test population is empty.
- Correction: split contract must be validated before trainer construction.

## R05 — Model lifecycle
- P0: Pipeline 02 wrong-success masks model evaluation defects.
- P0: HSE stochastic eval prevents a stable restore→test result.
- Correction: deterministic model evaluation before lifecycle promotion.

## R06 — Objective lifecycle
- P0: regularization omits the first parameter.
- P0: unsupported regularization may be silently dropped.
- Correction: verify each configured objective before fit and during backward.

## R07 — Trainer lifecycle
- P0: `run_pretrain()` and `run_adapt()` swallow `trainer.test()` exceptions.
- P1: data and logger cleanup is not guaranteed on all failures.
- Minimal correction: `try/finally`, original exception propagation, non-empty finite metrics.

## R08 — Decoupling lifecycle
- P0: Pipeline 02 fills trainer attributes and therefore owns training semantics it should only orchestrate.
- Correction: move semantic validation to configuration/Trainer boundary; orchestration passes values unchanged.
- Do not create LifecycleManager or StageRegistry.

## R09 — User lifecycle
- P1: success can depend on attestation/evidence finalization after scientific work completed.
- User-facing states should be: resolving, fitting, checkpoint ready, evaluating, succeeded/failed.
- These states can be represented by existing control flow; no new state framework required.

## R10 — Data Factory lifecycle
- P0: sampler can omit selected samples; selected population must remain stable throughout lifecycle.
- Data handles must close after success and every failure.
- Correction: selected-ID conservation plus explicit `close()` in runtime `finally`.

## R11 — Model Factory lifecycle
- P0: device ownership is split between Task and Trainer.
- A restored model must be placed by Trainer, not by Task constructor.
- Correction: Trainer is sole device authority; checkpoint loader remains strict by default.

## R12 — Task Factory lifecycle
- P0: unknown regularization is skipped.
- Task success requires the configured loss and metrics to exist; no warning-only objective removal.
- Correction: validate task/loss/metric before fit.

## R13 — Trainer Factory lifecycle
- P0: default Trainer supplies hidden epochs/gpus/pruning and attaches audit manifest callback.
- P0: evaluation errors can be suppressed in Pipeline 02.
- Correction: explicit trainer config, minimal callbacks, guaranteed cleanup.

## R14 — Group meta-review
- Highest-priority lifecycle PR:
  `fix(pipeline02): propagate evaluation failure and close resources`.
- Then separate device authority and remove hidden Trainer defaults.
- Default audit manifest callback and evidence/ledger finalizers are not scientific lifecycle requirements.

# Lifecycle invariant

```text
StageSuccess
= FitSuccess
  AND BestCheckpointExists
  AND CheckpointRestoreSuccess
  AND EvaluationSuccess
  AND MetricsNonemptyFinite
  AND RequiredResourcesClosed
```

# Wrong-success inventory

| Path | Current defect | Required behavior |
|---|---|---|
| Pipeline 02 | test exception swallowed | stage failure with original exception |
| Pipeline 03 | subexperiment exceptions become `None` and execution continues | experimental failure visible; no success claim |
| Explainability hooks | degraded/default metadata and warning-only failures | remain experimental; required semantics fail closed |
| Pipeline 06 | stage success can be overridden by missing ledger/evidence | scientific status independent of governance artifacts |

# Stop condition

No maintained path may emit a completion message or success object unless its declared evaluation completed with the declared checkpoint and a finite, non-empty result. Cleanup must occur independently of success or failure.

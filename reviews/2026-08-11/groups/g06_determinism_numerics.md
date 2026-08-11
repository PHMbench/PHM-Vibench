# g06_determinism_numerics — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Deterministic evaluation, numerical validity, exact objective participation
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific contract
- P0: HSE evaluation remains stochastic.
- Scientific claim is undefined unless randomness is part of an explicit estimator.
- Correction: deterministic eval first; Monte Carlo only as an explicit later protocol.

## R02 — Config/runtime
- P0: missing `pipeline` silently selects Pipeline 01.
- P1: malformed overrides may become strings.
- Correction: all numerical controls must be explicit and fail at config analysis.

## R03 — Data/numerics
- P0: HSE repeats input dimensions when patch size is invalid.
- P0: metadata parser guesses format/encoding.
- Correction: preserve raw tensor semantics and reject ambiguous parsing.

## R04 — Split/transform
- P0: impossible target-domain requests are reduced.
- P1: estimator semantics are not fully frozen.
- Correction: deterministic split construction plus declared evaluation unit and aggregation.

## R05 — Model determinism
- P0: `E_01_HSE.forward()` uses `torch.randint` in eval.
- P0: patch-size mismatch triggers signal/channel repetition.
- Acceptance: fixed checkpoint + fixed input + `eval()` yields bitwise-identical outputs; invalid patch shape fails.

## R06 — Objective numerics
- P0: regularization consumes the first parameter while detecting device.
- P0: unknown regularization is skipped.
- Correction: exact parameter list, finite scalar losses, unknown objective names fail.

## R07 — Evaluation lifecycle
- P0: Pipeline 02 test exceptions are swallowed.
- Numerical outputs may be absent while stage status says success.
- Correction: evaluation must return a non-empty finite mapping.

## R08 — Decoupling
- P1: shared runtime mutates Multitask configuration.
- Numerical behavior should be owned by Task/Model, not repaired by shared orchestration.
- Decision: remove special-case mutation rather than add UniversalContext.

## R09 — User experience
- P1: multiple manifests/evidence records obscure the authoritative metric.
- User-facing result should state metric name, unit, aggregation, sample count, checkpoint rule, and value.
- No digest or artifact-audit requirement.

## R10 — Data Factory
- P0: sampler can silently omit samples with missing metadata.
- Deterministic execution requires deterministic selected population, not only deterministic seeds.
- Correction: selected-ID conservation and explicit sampler coverage.

## R11 — Model Factory
- P0: non-CPU device requests become `auto`; Task also calls `.cuda()`.
- Device nondeterminism is a configuration semantic defect.
- Correction: Trainer is sole device authority; explicit CUDA unavailable means fail.

## R12 — Task Factory
- P0: unknown regularization types disappear from the objective.
- P0: missing task identity becomes classification.
- Correction: exact task/objective identity with no silent defaults.

## R13 — Trainer Factory
- P0: Pipeline 02 wrong-success and hidden trainer defaults undermine reproducibility.
- Correction: explicit epochs/device/monitor and fail-closed evaluation.
- Cleanup resources in `finally`.

## R14 — Group meta-review
- Highest-priority numerical chain:
  1. HSE deterministic eval;
  2. reject patch repetition;
  3. include every regularized parameter;
  4. deterministic validation augmentation;
  5. finite non-empty evaluation metrics.
- Deferred: repeated-seed statistics until a single-run estimator is semantically correct.

# Required mathematical contracts

```text
Fixed checkpoint + fixed input + eval mode -> fixed prediction
Configured objective weight > 0 -> finite scalar requiring gradients -> contributes to total loss
Reported metric -> declared population, unit, aggregation and checkpoint
```

# Stop condition

A run is numerically admissible only when all declared objectives are present and finite, all evaluation randomness is declared, and the reported metric can be recomputed from the exact test population without hidden weighting or omitted samples.

# g09_research_maturity — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Research maturity, support claims, and sequencing baseline before advanced algorithms
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific maturity
- P1: all maintained demos are `smoke_only`; none is `baseline_valid`.
- Importability, registry discovery and bounded smoke are not protocol validity.
- Correction: promote exactly one ordinary classification vertical slice only after real-data closure.

## R02 — Configuration maturity
- P0: explicit YAML without pipeline becomes Pipeline 01.
- A research claim cannot be mature when execution identity is supplied implicitly.
- Correction: canonical candidate config must declare every semantic field.

## R03 — Data maturity
- P0: cache semantics are weaker than reader semantics.
- Real-data maturity requires known reader, channels, units, sample rate, labels and selected population.
- Correction: certify only the candidate reader; do not audit every reader first.

## R04 — Protocol maturity
- P0: impossible target-domain requests are clamped.
- DG/CDDG maturity requires explicit source and target domains, no target use in validation/checkpoint selection, and declared overlap rules.
- Correction: one protocol-specific split, not a universal SplitEngine.

## R05 — Model maturity
- P0: HSE repeats dimensions and is stochastic in eval.
- A model with undefined deterministic estimator cannot support a baseline claim.
- Correction: deterministic eval and fail on incompatible patch shape.

## R06 — Task maturity
- P0: regularization can omit a parameter; unsupported metric/regularization names can disappear.
- Objective maturity requires every configured loss to be finite, differentiable and present in total loss.
- Correction: objective participation tests, no generic validation framework.

## R07 — Trainer maturity
- P0: Pipeline 02 evaluation failure may still return success.
- Multi-stage maturity is therefore below ordinary Pipeline 01 lifecycle maturity.
- Correction: keep Pipeline 02 `supported_limited` until fail-closed stage evaluation is proven.

## R08 — Decoupling maturity
- Existing evidence proves ordinary model replacement and compatible CSV dataset replacement.
- It does not prove episodic, generative, multi-task or multimodal plug-and-play.
- Decision: stop core abstraction; implement advanced methods as complete vertical slices later.

## R09 — User-facing maturity
- P1: result/evidence/attestation terminology can overstate scientific support.
- User docs must say exactly `smoke_only`, `baseline_valid`, `benchmark_candidate`, or `experimental`.
- Correction: no benchmark language for current maintained demos.

## R10 — Data Factory maturity
- P0: selected samples/systems can be silently omitted by sampler/drop_last.
- Data maturity requires selected-ID conservation through loader construction.
- Correction: fail if any selected file/system is not represented.

## R11 — Model Factory maturity
- P0: device semantics are split across Task and Trainer.
- Mature model replacement requires same input/label/device contract under both GlobalAverageLinear and ISFM.
- Correction: Trainer owns device, Model validates shape.

## R12 — Task Factory maturity
- P0: missing task identity becomes classification.
- FS/GFS names remain smoke semantics unless support/query, episodic labels/loss and estimator are present.
- Correction: do not implement true ProtoNet/GFS until ordinary baseline closes.

## R13 — Trainer Factory maturity
- P0: Pipeline 02 wrong-success remains.
- Default audit manifest callback is not evidence of training maturity.
- Correction: minimal lifecycle and explicit callbacks only.

## R14 — Group meta-review
- PR #147 is stale, experimental and conflicts with the no-ledger/no-hash direction.
- PR #148 is stale and governance-heavy; existing compatible data seam is already proven.
- Both remain deferred until current `dev` achieves one baseline-valid ordinary experiment.

# Maturity ladder

```text
not_run
→ smoke_only
→ baseline_valid
→ benchmark_candidate
→ benchmark_ready
```

Promotion to `baseline_valid` requires in one PR:

1. canonical config and exact command;
2. reader semantics test;
3. split requirements test;
4. deterministic replay test;
5. metric estimator test;
6. real-data end-to-end run;
7. checkpoint and non-empty finite metrics;
8. claim text matching the protocol.

# Research sequencing

```text
ordinary classification baseline_valid
→ true ProtoNet
→ true generalized few-shot
→ true two-stage pretrain/adapt
→ generative/multitask/foundation vertical slices
```

# Stop condition

No advanced research path may inherit a maturity label from component names, registry presence, CI count, hashes, ledgers or smoke completion. Maturity is granted only to an exact experiment combination with complete scientific semantics.

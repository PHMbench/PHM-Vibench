# g10_adversarial_prioritization — 14-Reviewer Dossier

- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Lens: Adversarial rejection, deduplication, and shortest critical path
- Verdict: `REQUEST_CHANGES`
- Source modifications: none

## R01 — Scientific contract
- Accepted: current maintained demos are smoke-only; no baseline-valid claim is authorized.
- Accepted P0: impossible domain requests are rewritten.
- Rejected: solving every pipeline before producing one ordinary valid baseline.

## R02 — Config/runtime
- Accepted P0: missing pipeline becomes Pipeline 01.
- Accepted P1: malformed override may be converted to string and lossy config decoding remains.
- Minimal action: strict explicit config; preserve the single ConfigAnalysis path.

## R03 — Data/metadata
- Accepted P0: cache identity ignores reader semantics.
- Accepted P0: automatic format/provider fallback hides data provenance and parsing errors.
- Rejected: universal metadata schema or certification framework.

## R04 — Split/protocol
- Accepted P0: unknown task type can create train=test=all IDs.
- Accepted P0: target-domain request can be clamped.
- Minimal action: exact task type and exact source/target domains for one candidate.

## R05 — Model/device
- Accepted P0: HSE repeats input dimensions and remains stochastic in eval.
- Accepted P0: device authority is split between Task and Trainer.
- Minimal action: Trainer-only device authority; deterministic eval; shape mismatch fails.

## R06 — Task/loss/metric
- Accepted P0: first parameter omitted from regularization.
- Accepted P0: unknown metric/regularization can be skipped.
- Accepted P0: class count inferred without label-ontology validation.

## R07 — Trainer/lifecycle
- Accepted P0: Pipeline 02 swallows test errors and returns empty metrics.
- Accepted P1: resources not always closed in `finally`.
- Minimal action: fail-closed stage lifecycle, no StageManager.

## R08 — Decoupling
- Accepted: existing CSV Data extension and GlobalAverageLinear Model extension prove current ordinary seams.
- Accepted P1: shared runtime mutates Multitask config.
- Rejected: PluginSpec, ReaderPlugin, ComponentSpec, UniversalBatch, second registry.

## R09 — User experience
- Accepted P1: multiple result and governance authorities confuse users.
- Accepted P1: evidence/attestation finalization can control public success.
- Minimal action: one command, one experiment, one result location.

## R10 — Data Factory
- Accepted P0: sampler/drop_last can remove selected samples or systems.
- Required conservation: selected = materialized = dataset = loader-represented IDs.
- Rejected: adding a DataManager around the existing factory.

## R11 — Model Factory
- Accepted P0: explicit CUDA may become auto/CPU; model/task constructors may move device.
- Accepted: strict checkpoint loader is a good boundary and should remain.
- Rejected: automatic prefix guessing or universal checkpoint adapter.

## R12 — Task Factory
- Accepted P0: missing task identity becomes classification.
- Accepted P0: dataset metric identity uses first file ID.
- Rejected: universal task context; require exact batch semantics instead.

## R13 — Trainer Factory
- Accepted P0: hidden epochs/device defaults and wrong-success.
- Accepted P1: default audit callback does not serve core scientific lifecycle.
- Minimal action: explicit trainer config and minimal callbacks.

## R14 — Final group arbitration
- PR #148 is stale and governance-heavy; not a first-baseline blocker.
- PR #147 is stale, experimental and ledger/hash/evidence-heavy; keep deferred.
- WIP limit: one critical scientific PR + one independent correctness PR; total implementation PRs ≤2.

# Deduplicated P0 set

1. explicit config missing pipeline is silently repaired;
2. unknown/impossible split requests are silently rewritten;
3. selected files/systems can be silently omitted;
4. HSE changes input shape and is stochastic in evaluation;
5. Task/Trainer conflict on device;
6. label ontology and metric identity are under-specified;
7. regularization omits a parameter and unsupported objectives can disappear;
8. Pipeline 02 evaluation failure can still return success.

# Rejected proposals

```text
new hash/checksum/digest/receipt/ledger
a second resolver or registry
FactoryManager / LifecycleManager / DataManager
UniversalContext / UniversalBatch
five split strategies for one candidate
coverage-driven test expansion
release/rename work before scientific closure
```

# Final critical path

```text
PR1  config fail-fast: explicit pipeline and override parsing
PR2  protocol fail-fast: task/system/domain/label population
PR3  device + HSE deterministic/shape truth
PR4  task objective + metric estimator truth
PR5  Pipeline 02 wrong-success (parallel P1 if first candidate uses Pipeline 01)
PR6  one real ordinary classification baseline_valid promotion
```

# Candidate validation ladder

```text
Dummy/CSV fixture
→ MFPT transparent baseline
→ SEU condition-DG
→ PU multichannel
→ CWRU local final acceptance
```

# Stop condition

The program stops reviewing and starts implementation when each accepted P0 has a minimal owner, allowed-file boundary, focused acceptance test and explicit non-goals. No cleanup or architecture proposal may enter the critical path without a current user burden or correctness risk.

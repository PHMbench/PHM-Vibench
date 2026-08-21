# PHMFactory Core Contract

This file defines the smallest stable authority for PHMFactory development. The latest
`dev` code and tests remain the factual source of current behavior. User instructions
belong in [`README.md`](README.md) and [`docs/quickstart.md`](docs/quickstart.md).
Historical plans and audits are evidence about earlier states, not current authority.

## 1. Product goal

PHMFactory exists to let a user declare, execute, and inspect an industrial PHM experiment
without the framework silently changing its scientific meaning.

The governing invariant is:

```text
requested experiment = executed experiment
```

Represent an experiment as:

$$
\mathcal E=(\mathcal D,\Pi,f_\theta,\mathcal L,\widehat R),
$$

where:

- $\mathcal D$: data population, metadata, labels, channels, and domains;
- $\Pi$: split, sampling, preprocessing, training, checkpoint, and test protocol;
- $f_\theta$: the model actually constructed and evaluated;
- $\mathcal L$: the objective actually used for optimization;
- $\widehat R$: the estimator actually reported.

A run is scientifically successful only when all five terms match the visible request.
An exit code, passing CI job, manifest, or hash cannot replace this condition.

## 2. Maintained user path

```text
phmfactory --config <yaml> [--local-config <yaml>] [--override key=value ...]
    ↓
resolve and validate one visible configuration
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → selected checkpoint → test → complete finite metrics
    ↓
direct result paths
```

`phmfactory doctor`, `phmfactory preflight --config smoke`, and `phmfactory demo` are the
bounded offline first-run path. `python main.py` remains a compatibility launcher, not a
second runtime.

## 3. Responsibility boundaries

| Boundary | Owns | Must not own |
| --- | --- | --- |
| Data Factory | reader, metadata, selected IDs, datasets, samplers, loaders | model, objective, device, or metric repair |
| Model Factory | model identity, construction, explicitly requested weights | data split, task choice, or device movement |
| Task Factory | task identity, objective, metric lifecycle | hardware, checkpoint selection, or data repair |
| Trainer Factory | device, callbacks, checkpoint selection, fit/test lifecycle | missing task or data semantics |
| Pipeline | orchestration, success gating, direct result locations | silent repair of any Factory input |

Replacing one compatible component should require changing that component and its
configuration, not the other factories or the public command router.

## 4. Occam decision rule

Before adding code or documentation, answer:

```text
Which current user action fails?
Which current scientific claim is unsupported?
What is the smallest change that distinguishes or fixes the root cause?
What can be deleted instead of added?
```

Preferred order:

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

A new abstraction is allowed only when at least two current maintained consumers need the
same behavior, it immediately removes duplicate logic, and it does not create another
user-visible concept or hide a failure boundary.

## 5. Permanent prohibitions

Do not add or restore:

- consumerless hash, checksum, digest, hash chain, receipt, ledger, or attestation;
- artifact-integrity infrastructure as a substitute for data, split, reader, checkpoint,
  or estimator validation;
- silent fallback to another data source, model, task, device, loss, metric, checkpoint,
  backend, or test population;
- warning-and-continue behavior that drops selected samples or declared metrics;
- automatic repair of labels, channels, patch size, domains, or scientific configuration;
- `FactoryManager`, `BackendManager`, `ResultManager`, `UniversalContext`, `SchemaV2`,
  registry-of-registries, or plugin frameworks without an immediate current need;
- large refactors justified only by hypothetical future datasets, backends, distributed
  execution, or model families;
- broad exception wrappers that replace the original useful error;
- tests that merely encode an obsolete claim or suppress a real failure.

A hash may remain only when a current consumer reads it and changes a real maintained
runtime decision. Git/provider revisions may identify source versions, but are not
scientific or security proofs.

## 6. Failure contract

The default behavior is fail-fast:

```text
invalid request
→ fail at the owning boundary
→ preserve the original exception and context
→ do not execute an easier experiment
```

Useful errors identify the location, requested value, actual value, expected contract, and
minimal repair. Cleanup belongs in `finally`; cleanup must not replace the source failure.

## 7. Evidence and support terms

```text
discoverable       source or registry entry exists
runnable           a reviewed execution path exists
execution-verified the exact command has bounded current execution evidence
baseline-valid     the exact complete experiment passed its current scientific protocol
```

`baseline-valid` is configuration-specific. It cannot be inferred from importability,
source presence, another configuration, or historical results.

Current source truth:

- the offline Dummy path is maintained and execution-verified;
- the MFPT transparent experiment is a reviewed candidate at `smoke_only` until the exact
  current-source protocol is requalified;
- there is currently no current-source `baseline_valid` registry row;
- v0.3 release readiness is therefore blocked;
- IoTDB and `phm-data-factory` are optional/deferred and are not core dependencies.

The status authorities are:

- [`configs/config_registry.csv`](configs/config_registry.csv);
- [`SUPPORTED_COMBINATIONS.md`](SUPPORTED_COMBINATIONS.md);
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md);
- [`docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).

## 8. Pull-request discipline

One PR protects one primary invariant and produces one user-observable outcome.

Each PR must state:

```text
Current fact
Root cause
Scope
Out of scope
Behavior after the change
Focused validation
Known limitation
Rollback: revert the squash commit
```

Keep one critical implementation PR in progress. Do not mix runtime, broad cleanup,
research methods, generated documentation, and release claims in one change.

Use path-relevant validation. All runtime changes protect the offline Dummy path; only
changes that affect a real-data protocol should trigger its heavy workflow.

## 9. Current convergence order

The next bounded sequence is:

```text
shared strict public schema
→ single immutable invocation root
→ declared metric closure
→ explicit scheduler semantics
→ default-only public Data Factory
→ remove consumerless public identities
→ current-source MFPT requalification
```

Do not restore a stronger release or benchmark claim merely to make a gate pass. New
evidence changes the claim; the claim does not dictate the evidence.

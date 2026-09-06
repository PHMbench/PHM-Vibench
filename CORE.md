# PHMFactory Core Contract

This file records the smallest stable development contract. Current behavior is defined by
the latest `dev` code and tests. User instructions belong in `README.md` and
`docs/quickstart.md`; historical plans do not override this file.

## 1. Product goal

A user declares one PHM experiment and PHMFactory executes that experiment without
silently changing its scientific meaning.

```text
requested experiment = executed experiment
```

Represent an experiment as:

$$
\mathcal E=(\mathcal D,\Pi,f_\theta,\mathcal L,\widehat R),
$$

where $\mathcal D$ is the data population, $\Pi$ the protocol, $f_\theta$ the constructed
model, $\mathcal L$ the optimized objective, and $\widehat R$ the reported estimator.
All five terms must match the visible request.

## 2. Maintained path

```text
phmfactory --config <yaml> [--local-config <yaml>] [--override key=value ...]
    ↓
resolve and validate one visible configuration
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → selected checkpoint → test → finite metrics
    ↓
direct result paths
```

`phmfactory doctor`, `phmfactory preflight --config smoke`, and `phmfactory demo` form the
offline first-run path. `python main.py` is a compatibility launcher, not a second runtime.

## 3. Responsibility boundaries

| Boundary | Owns | Must not repair |
| --- | --- | --- |
| Data Factory | metadata, readers, sample selection, datasets, samplers, loaders | model, task, device, metric |
| Model Factory | model identity, construction, explicit weights | split, task, device |
| Task Factory | objective, metrics, optimizer, scheduler | hardware, checkpoint, data |
| Trainer Factory | device, callbacks, checkpoint, fit/test lifecycle | missing data or task semantics |
| Pipeline | orchestration and success gating | any Factory input |

Replacing one compatible component should require changing that component and its
configuration, not the other factories or the command router.

## 4. Decision rule

Before adding code or documentation, answer:

```text
Which current user action fails?
Which scientific claim lacks evidence?
What is the smallest change that reaches the root cause?
What can be deleted instead?
```

Preferred order:

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

A new abstraction is justified only when at least two current maintained consumers need
the same behavior and the change immediately removes duplicate logic.

## 5. Prohibited patterns

Do not add or restore:

- duplicate audit or provenance control planes in the runtime;
- silent fallback to another data source, model, task, device, loss, metric, checkpoint,
  backend, or test population;
- warning-and-continue behavior that drops selected samples or declared metrics;
- automatic repair of labels, channels, patch size, domains, or experiment configuration;
- manager, context, plugin, schema, or registry layers without an immediate maintained
  consumer;
- broad exception wrappers that replace the useful source error;
- large refactors justified only by hypothetical future backends, datasets, or models;
- tests that preserve an obsolete architecture instead of a user or scientific invariant.

Comments should explain *why* a scientific or compatibility constraint exists. Remove
comments that merely restate code or describe behavior that no longer exists.

## 6. Failure contract

```text
invalid request
→ fail at the owning boundary
→ preserve the source error and useful context
→ do not run an easier experiment
```

A useful error identifies the location, requested value, observed value, expected
contract, and smallest repair. Cleanup belongs in `finally` and must not replace the
source failure.

## 7. Support terms

| Term | Meaning |
| --- | --- |
| `discoverable` | source or registry entry exists |
| `runnable` | a reviewed execution path exists |
| `execution-verified` | the exact command has current bounded execution evidence |
| `baseline-valid` | the exact full experiment passed its current scientific protocol |

Support is configuration-specific. Source presence, importability, another configuration,
or a historical result cannot establish it.

Current state:

- the offline Dummy path is maintained;
- the MFPT transparent experiment remains `smoke_only` pending current-source
  requalification;
- there is no current `baseline_valid` registry row;
- release readiness is blocked;
- IoTDB and `phm-data-factory` are optional/deferred, not core dependencies.

Current status sources:

- `configs/config_registry.csv`;
- `SUPPORTED_COMBINATIONS.md`;
- `KNOWN_LIMITATIONS.md`;
- `docs/PHMFACTORY_V0_3_RELEASE_READINESS.md`.

## 8. Pull-request discipline

One PR protects one primary invariant and produces one user-visible outcome.

Each PR states:

```text
Current fact
Root cause
Scope
Out of scope
Behavior after the change
Focused validation
Known limitation
Rollback
```

Keep one critical implementation PR in progress. Use validation that matches the changed
risk surface. Runtime changes protect the offline Dummy path; only changes affecting a
real-data protocol should trigger its heavy workflow.

## 9. Completed convergence steps

Current `dev` already enforces:

- one strict public configuration acceptance boundary;
- explicit experiment selection at public entrypoints;
- explicit `environment.seed` and `environment.iterations`;
- explicit classification `trainer.num_epochs` and `trainer.test_after_fit`;
- one maintained device-count field: `trainer.devices`.

Do not reopen these items without a new executable counterexample.

## 10. Current convergence order

```text
normal wheel installation evidence
→ one immutable result root per invocation
→ one successful Pipeline result contract
→ declared evaluation metric closure
→ explicit checkpoint and scheduler behavior
→ default-only public Data Factory
→ current-source MFPT requalification
```

Do not strengthen a release or benchmark claim to make a gate pass. Evidence changes the
claim; the claim does not change the evidence.

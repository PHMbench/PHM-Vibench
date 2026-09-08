# PHMFactory Core Contract

This file states the stable project requirements, not a release status or task queue.
Current code and tests establish actual behavior. If they conflict with this contract,
record the defect; do not silently lower the requirement to match the implementation.
User instructions belong in [README.md](README.md) and [Quickstart](docs/quickstart.md).
Historical plans do not override the current task or this contract.

## 1. Product goal

A user declares one PHM experiment and PHMFactory executes it without silently changing
its scientific meaning.

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

`phmfactory doctor`, `phmfactory preflight --config smoke`, and `phmfactory demo` are the
first-run commands. The Dummy runtime uses bundled inputs; installing dependencies is a
separate operation. `python main.py` is a compatibility launcher, not another runtime.
An explicitly training-only run must not be presented as evaluated.

## 3. Responsibility boundaries

| Boundary | Owns | Must not repair |
| --- | --- | --- |
| Data Factory | metadata, readers, sample selection, datasets, samplers, loaders | model, task, device, metric |
| Model Factory | model identity, construction, explicit weights | split, task, device |
| Task Factory | objective, metrics, optimizer, scheduler | hardware, checkpoint, data |
| Trainer Factory | device, callbacks, checkpoint, fit/test lifecycle | missing data or task semantics |
| Pipeline | orchestration and success gating | any Factory input |

Replacing one compatible component should require changing that component and its
configuration, not the other factories or command router.

## 4. Decision rule

Before adding code or documentation, identify the current user action or scientific
question, the verified failure or uncertainty, and the smallest useful correction.
Consider what can be removed instead:

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

A new abstraction needs at least two current maintained consumers and must immediately
remove duplicate logic. A demonstration of an idea is not an obligation to add it to the
public runtime.

## 5. Prohibited patterns

Do not add or restore:

- hash, checksum, digest, receipt or ledger systems as substitutes for direct scientific
  validation, or duplicate audit/provenance control planes;
- silent fallback to another source, model, task, device, objective, checkpoint, backend
  or test population;
- warning-and-continue behavior that drops selected samples or declared metrics;
- automatic repair of labels, channels, patch size, domains or experiment configuration;
- manager, context, plugin, schema or registry layers without current consumers;
- broad exception wrappers that replace the useful source error;
- large refactors justified only by hypothetical future uses;
- tests that preserve obsolete architecture instead of user or scientific behavior.

Comments explain why a constraint exists. Remove comments that merely repeat code or
state behavior that no longer exists. Do not bulk-format unrelated files.

## 6. Failure contract

```text
invalid request
→ fail at the owning boundary
→ preserve the source error and useful context
→ do not run an easier experiment
```

A useful error identifies the location, requested value, observed value, expected
contract and smallest repair. Cleanup belongs in `finally` and must not replace the
source failure. A local environment limitation is not a permanent project requirement.

## 7. Support terms

| Term | Meaning |
| --- | --- |
| `discoverable` | source or catalogue entry exists |
| `runnable` | a reviewed execution path exists |
| `execution-verified` | the exact command has bounded execution evidence |
| `baseline-valid` | the exact full experiment passed its scientific protocol |

Support is configuration-specific. Source presence, importability, another configuration
or a historical result cannot establish it. Do not strengthen a claim to pass a check.
Software regression, scientific acceptance, source merge and package publication are
separate outcomes and must be reported separately.

For the current state, inspect [config registry](configs/config_registry.csv),
[supported combinations](SUPPORTED_COMBINATIONS.md), [known limitations](KNOWN_LIMITATIONS.md),
[release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md), and the associated current
checks. Use [changelog](doc/changelog/) and PR records to determine completed work.
Do not freeze commit IDs, open PR numbers or a next-task queue in this contract.

## 8. Change discipline

One PR protects one primary invariant and states the current fact, root cause, scope,
non-goals, observable result, focused validation, limitations and rollback.
Keep one critical implementation change in progress; keep unrelated research separate.
Follow [CONTRIBUTING.md](CONTRIBUTING.md) for branch and merge rules.

Validation follows the changed risk surface. Test runtime changes on the affected
maintained path. A documentation edit does not require a new real-data experiment.
Existing automatic checks remain enabled; distinguish their actual results from checks
that were not run. Do not substitute a mock or static inspection for claimed execution.

## 9. Shared AI guidance

[AGENTS.md](AGENTS.md) is a short operational entrypoint; root `CLAUDE.md` imports it.
This is an explicit exception to the former root-file ban, not permission to commit
personal Agent workspaces. Module knowledge belongs in neutral READMEs, not duplicated
or nested instruction files. The existing boundary check enforces that distinction.
No personal credentials, tool permissions, hooks, local settings or conversation logs
belong in these shared documents.

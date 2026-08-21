# Agent Instructions

Read [`README.md`](README.md) for the user path and [`CORE.md`](CORE.md) for the project
contract before changing the repository.

## Source of truth

1. Fetch the latest `dev` and inspect current code, tests, open PRs, and generated support
   files.
2. Treat historical handoffs, audits, plans, old release notes, and archived documents as
   evidence about earlier states, not current authority.
3. Do not repeat a task already completed on `dev`.
4. Do not claim a run, baseline, release, or support status without current evidence.

## Work loop

For each task:

```text
state the strongest current claim
→ identify the largest uncertainty or user failure
→ choose the smallest discriminating action
→ define possible outcomes
→ implement one coherent change
→ run focused validation
→ update the claim or boundary
```

Stop when the issue is resolved. Do not expand the scope because adjacent cleanup is
possible.

## Required engineering behavior

- Preserve `requested experiment = executed experiment`.
- Fail at the owning boundary; preserve useful original exceptions.
- Keep the public path configuration-first.
- Use repository Dummy data or tiny generated fixtures for normal correctness work.
- Keep Data, Model, Task, Trainer, and Pipeline responsibilities separate.
- Prefer deletion, inlining, and simplification over a new abstraction.
- Keep one primary invariant per PR and target routine PRs to `dev`.
- Record commands and results accurately; distinguish local results from GitHub Actions.

## Forbidden by default

Do not add:

- consumerless hashes, checksums, digests, receipts, ledgers, or attestations;
- silent fallback or warning-and-continue scientific behavior;
- synthetic replacement data in production paths;
- automatic label, channel, domain, device, checkpoint, loss, or metric repair;
- Factory/Manager/Registry nesting;
- a second config resolver, runtime, schema, result authority, or evaluation path;
- a broad future-oriented refactor without a current maintained consumer;
- a large goal registry, policy directory, manifest family, or audit bureaucracy;
- unrelated formatting, CI, coverage, or directory work as the main deliverable.

A new abstraction needs at least two current maintained consumers and must delete existing
duplication immediately. Otherwise use a function or direct code.

## Factory boundaries

```text
Data Factory    reader, metadata, selected IDs, datasets, samplers, loaders
Model Factory   model identity, construction, explicit weights
Task Factory    task identity, objective, metric lifecycle
Trainer Factory device, callbacks, checkpoints, fit/test lifecycle
Pipeline        orchestration, success gating, direct result locations
```

Do not let one boundary repair another.

## Validation

Run the smallest relevant set first. Runtime work normally includes:

```bash
python -m pytest <focused-tests> -q
python -m scripts.validate_configs
python -m scripts.validate_docs
phmfactory preflight --config smoke
phmfactory demo
```

Regenerate the configuration atlas or support matrices only when their source registry or
generator changes. Run a real-data workflow only when the change can alter that exact
protocol.

## Pull requests

A PR description must include:

```text
problem and verified evidence
scientific or user invariant
minimal change
explicit non-goals
validation performed
remaining limitation
rollback: revert the squash commit
```

Do not merge while relevant checks fail. Do not modify a test solely to hide a real
failure; update a test only when its asserted authority is demonstrably obsolete.

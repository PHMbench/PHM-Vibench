# Contributing to PHMFactory

<div align="center">
  <p>
    <a href="CONTRIBUTING.md"><strong>English</strong></a> |
    <a href="CONTRIBUTING_CN.md">中文</a>
  </p>
</div>

Read [`README.md`](README.md), [`CORE.md`](CORE.md), and the relevant Factory guide before
changing the repository.

PHMFactory is configuration-first. Contributions must preserve:

```text
requested experiment = executed experiment
```

and the public path:

```bash
phmfactory --config <yaml> [--local-config <yaml>] [--override key=value ...]
```

## 1. Decide whether the change belongs here

A proposal should identify:

```text
current user action or scientific claim
verified failure or uncertainty
smallest useful intervention
simpler alternative considered
observable acceptance result
```

Do not add a feature because it may be useful for hypothetical future datasets, backends,
models, distributed systems, or workflows. Research variants without a maintained user
need belong in `configs/experiments/` or a separate research repository until their
interface is stable.

## 2. Permanent constraints

Contributions are normally rejected when they add or restore:

- consumerless hashes, checksums, digests, receipts, ledgers, or attestations;
- silent fallback to another data source, backend, model, task, device, loss, metric,
  checkpoint, or test population;
- warning-and-continue behavior that drops requested samples or declared metrics;
- automatic label, channel, split, domain, patch, or objective repair;
- Factory/Manager/Registry nesting or a second config/runtime/result authority;
- a broad exception wrapper that hides the useful source error;
- a large future-oriented refactor without current maintained consumers;
- a goal registry, policy tree, or manifest family that duplicates `CORE.md`, the config
  registry, or direct result paths;
- test changes whose only purpose is to suppress a real failure.

Prefer:

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

A new abstraction needs at least two current maintained consumers and must immediately
remove duplicate logic without adding another user concept.

## 3. Branch and pull-request model

Long-lived branches:

```text
main  user-facing stable/release line
dev   integration line for routine work
```

Create routine topic branches from current `dev` and target PRs to `dev`:

```bash
git switch dev
git pull --ff-only origin dev
git switch -c <type>/<short-topic>
```

Suggested prefixes:

```text
fix/       correctness or compatibility
docs/      documentation only
feat/      bounded user-visible capability
test/      test or legal fixture
ci/        path-relevant automation
cleanup/   deletion or simplification
release/   explicitly authorized release work
```

`main` accepts only an authorized release-promotion PR or emergency hotfix. Routine PRs
do not target `main`.

One PR protects one primary invariant and produces one user-observable result. If a change
modifies Data, Model, Task, Trainer, Pipeline, UI, release claims, and broad documentation
at once, split it.

Routine PRs may be squash-merged. Rollback is the revert of that squash commit.

## 4. Factory boundaries

```text
Data Factory    reader, metadata, selected IDs, datasets, samplers, loaders
Model Factory   model identity, construction, explicit weights
Task Factory    task identity, objective, metric lifecycle
Trainer Factory device, callbacks, checkpoints, fit/test lifecycle
Pipeline        orchestration, success gating, direct result locations
```

Do not let one boundary repair another.

Factory guides:

- [Data](src/data_factory/contributing.md)
- [Models](src/model_factory/contributing.md)
- [Tasks](src/task_factory/contributing.md)
- [Trainers](src/trainer_factory/contributing.md)

## 5. Bug reports

Include:

- exact command and exit code;
- config path, explicit local config, and all overrides;
- expected and actual behavior;
- full text error and traceback;
- operating system, Python, PyTorch, Lightning, and CUDA versions when relevant;
- data source and whether repository Dummy or external data are involved;
- the smallest shareable config/fixture;
- the repository commit.

Do not replace text logs with screenshots. Security reports follow
[`SECURITY.md`](SECURITY.md), not public issues.

## 6. Data and reader contributions

Provide:

- source, stable revision/location, citation, license, and redistribution boundary;
- expected local directory and metadata layout;
- reader input/output shape, dtype, channel order, units, and preprocessing;
- explicit failure behavior for malformed input;
- a small legal or synthetic fixture;
- a focused reader test and one bounded config;
- the exact split and estimator claim boundary.

A reader must not synthesize substitute data, guess incompatible formats, reorder channels
silently, or skip selected files after a failure.

Large raw data and model weights normally remain outside Git.

## 7. Model, task, trainer, and config contributions

A public component normally includes:

- implementation in its existing Factory boundary;
- constructor and tensor/dtype/device contract;
- focused positive and negative tests;
- a smallest runnable configuration;
- explicit compatible and incompatible combinations;
- optional dependency and license information;
- checkpoint behavior when relevant;
- honest evidence level: discoverable, runnable, execution-verified, or baseline-valid.

Do not modify `main.py` to add a compatible component. Do not claim support because a
source file imports.

New experiment configs belong under `configs/experiments/`. Promotion to a maintained
demo or baseline requires exact current execution evidence and the corresponding registry
change.

## 8. Documentation changes

Before adding a page, check [`docs/index.md`](docs/index.md). Update the existing authority
instead of creating a duplicate.

Documentation must:

- verify commands, paths, config keys, and filenames;
- distinguish current, experimental, deferred, unsupported, and historical behavior;
- distinguish smoke evidence from scientific protocol evidence;
- avoid unsupported accuracy, SOTA, compatibility, release, or dataset-count claims;
- link to installation, configuration, support, and limitation authorities rather than
  copying them.

`README.md` is the user entrypoint; `CORE.md` holds project invariants;
`KNOWN_LIMITATIONS.md` holds current boundaries; release blockers live in the release
readiness page.

## 9. Validation

Run focused tests first. A typical runtime change uses:

```bash
python -m pytest <focused-tests> -q
python -m scripts.validate_configs
python -m scripts.validate_docs
phmfactory preflight --config smoke
phmfactory demo
```

Run generated-document checks only when their source registry or generator changes:

```bash
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md

python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
```

Run a real-data workflow only when the change can alter that exact protocol. Do not make
every PR download external data.

Record each command as:

```text
PASS
FAIL
EXPECTED FAILURE
NOT EXECUTED — <reason>
```

Local results are not GitHub Actions results.

## 10. Pull-request description

Include:

```text
verified problem and root cause
scientific or user invariant
minimal change
explicit non-goals
public behavior after the change
focused tests and outcomes
remaining limitations
rollback: revert the squash commit
```

Do not merge while relevant checks fail. Do not lower a test, status, or release standard
to make the PR pass. A test may change when its asserted authority is demonstrably stale;
state that reason explicitly.

## 11. Current release boundary

The source version is `0.3.0rc1`, but current release readiness is blocked until an exact
real-data experiment is requalified as `baseline_valid` on current source. Do not restore
that claim merely to satisfy a release gate.

IoTDB and `phm-data-factory` remain optional/deferred. They are not core dependencies and
must not be introduced through a fallback or broad backend abstraction.

## 12. Community and licensing

Participation follows [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Contributions are
licensed under the repository's [Apache License 2.0](LICENSE), subject to separately
identified third-party and dataset licenses.

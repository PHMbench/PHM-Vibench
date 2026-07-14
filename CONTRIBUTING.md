# Contributing to PHM-Vibench

<div align="center">
  <p>
    <a href="CONTRIBUTING.md"><strong>English</strong></a> |
    <a href="CONTRIBUTING_CN.md">中文</a>
  </p>
</div>

PHM-Vibench welcomes focused bug fixes, tests, documentation, configurations,
data readers, models, tasks, trainers, and reproducibility improvements.

The project is configuration-first. Contributions must preserve:

```bash
python main.py --config <yaml> [--override key=value ...]
```

and the five public configuration blocks:

```text
environment / data / model / task / trainer
```

Read the [documentation index](docs/index.md),
[developer guide](docs/developer_guide.md), and
[testing guide](docs/testing.md) before making a broad change.

## Before opening an issue or pull request

1. Search existing issues and pull requests.
2. Reproduce the behavior on a current `main` checkout when possible.
3. Start from the nearest maintained config under `configs/demo/`.
4. Keep local variants under `configs/experiments/` or in an untracked local
   config; do not commit personal absolute paths.
5. Separate unrelated runtime, documentation, data-artifact, and cleanup work.
6. Do not claim support merely because a registry row or source file exists.

For major architecture changes, new pipelines, public compatibility changes, or
large data/model additions, open an issue before implementation.

## Report a bug

Use the bug-report issue template and include:

- a concise problem statement;
- exact reproduction steps;
- expected and actual behavior;
- operating system and hardware;
- Python, PyTorch, PyTorch Lightning, and CUDA versions when relevant;
- repository commit or release tag;
- configuration path and all CLI overrides;
- data source and whether dummy or external data is used;
- complete error output as text;
- the smallest reproducible config or test case that can be shared.

A missing dependency, invalid configuration, and code defect are different failure
classes. Include the command's exit code and avoid replacing logs with screenshots
when text is available.

Do not report security vulnerabilities in a public issue. Follow
[SECURITY.md](SECURITY.md).

## Propose a feature

A feature request should explain:

- the user or research scenario;
- the current limitation;
- the behavior being requested;
- why the change belongs in PHM-Vibench rather than a local experiment;
- simpler alternatives considered;
- expected compatibility, dependency, test, documentation, and maintenance cost;
- whether the feature is intended to be maintained, experimental, or research-only.

For a new method, include a primary paper or stable technical reference, but do not
present a paper citation as evidence that the repository implementation works.

## Development setup

Follow the [installation guide](docs/installation.md). A typical branch starts
from current `main`:

```bash
git switch main
git pull --ff-only origin main
git switch -c <type>/<short-topic>
```

Suggested branch prefixes:

```text
fix/       bug or compatibility fix
feat/      user-visible capability
docs/      documentation-only work
test/      tests and fixtures
ci/        workflow or automation
cleanup/   bounded removal or repository hygiene
release/   release preparation
```

Keep a branch focused. Prefer small commits that leave the repository in a
reviewable state.

Suggested commit format:

```text
<type>: <imperative summary>
```

Examples:

```text
fix: reject unknown task registry entries
test: cover TSPN_UXFD CPU assembly
docs: clarify external data layout
```

Commit history does not need to be artificially expanded; pull requests are
normally squash-merged.

## Make a code contribution

The standard sequence is:

```text
create a focused branch
→ make the smallest coherent change
→ add or update focused tests
→ update the authoritative documentation
→ run the relevant local gates
→ review the diff for unrelated changes
→ open a pull request with exact evidence
```

Architecture constraints:

- extend `src/data_factory/`, `src/model_factory/`, `src/task_factory/`, or
  `src/trainer_factory/` instead of adding component-specific branches to
  `main.py`;
- keep the public CLI and five-block config model compatible;
- make invalid combinations fail early with a useful message;
- avoid hidden fallback, silent partial checkpoint loading, and machine-specific
  defaults unless they are explicitly documented compatibility behavior;
- provide migration notes or a compatibility layer for intentional behavior
  changes;
- do not edit tests merely to hide a real failure.

Factory-specific guides:

- [Data and readers](src/data_factory/contributing.md)
- [Models](src/model_factory/contributing.md)
- [Tasks](src/task_factory/contributing.md)
- [Trainers](src/trainer_factory/contributing.md)

## Contribute a dataset or reader

Provide all applicable items:

- dataset name, original source, stable download location, and citation;
- license and redistribution constraints;
- expected directory and metadata layout;
- reader implementation and input/output contract;
- preprocessing and split procedure;
- a configuration under `configs/experiments/` or a justified maintained demo;
- a small legal fixture or synthetic contract test when raw data cannot be
  redistributed;
- the exact inspection and smoke commands used;
- expected output structure, not invented benchmark metrics;
- known limitations and reproducibility notes.

Large dataset payloads should normally remain outside Git. Reference notes and
metadata do not imply redistribution rights. See [data/README.md](data/README.md).

## Contribute a model, task, trainer, or configuration

A public component contribution should normally include:

- implementation in the correct factory boundary;
- registry or config entry where required;
- documented constructor, batch, tensor-shape, dtype, device, and output contract;
- focused positive and negative tests;
- checkpoint or state behavior when relevant;
- a smallest runnable config;
- explicit compatible and incompatible components;
- dependency and license information for copied or adapted code;
- limitations and evidence level.

New local experiments belong in `configs/experiments/`. Promotion to
`configs/demo/` and `sanity_ok` requires reviewable runtime evidence. Update
`configs/config_registry.csv`, regenerate `docs/CONFIG_ATLAS.md`, and update
support documents only when the maintained public surface intentionally changes.

## Contribute documentation

Before adding a page, check the [documentation index](docs/index.md) and update the
existing authority when one exists.

Documentation contributions must:

- identify the reader and task;
- define new abbreviations and terminology;
- use repository-relative links for internal files;
- verify commands, paths, configuration keys, and filenames;
- distinguish maintained, experimental, planned, deprecated, and historical
  behavior;
- avoid unsupported performance, compatibility, dataset-count, and maturity
  claims;
- add a new maintained page to the documentation navigation;
- preserve historical evidence when deletion would break provenance or external
  references.

Do not copy installation, quickstart, configuration precedence, test gates, or
support matrices into a new page. Link to their authorities instead.

## Run validation

Choose the relevant commands from [docs/testing.md](docs/testing.md). The general
local gate is:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
python -m pytest test/ -q
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Run focused tests first. A documentation-only pull request can use a narrower gate
when it does not modify executable commands or runtime claims; explain omitted
commands in the pull-request description.

Record commands and outcomes accurately:

```text
PASS
FAIL
EXPECTED FAILURE
NOT EXECUTED — <reason>
```

Local evidence is not GitHub Actions evidence.

## Open a pull request

A pull request must include:

- problem and rationale;
- exact scope and explicit non-goals;
- changed public behavior and migration impact;
- files or components affected;
- commands run and results;
- tests added or changed;
- documentation and registry updates;
- known risks and limitations;
- rollback method;
- confirmation that local goal packs, caches, logs, credentials, raw data, and
  machine paths are not included.

Do not mix broad formatting with behavioral changes. Do not submit a generated
atlas without its registry source change. Do not lower a test or support standard
to make a pull request pass.

At least one maintainer review is expected for merge. The repository normally uses
squash merge after required checks pass.

## Changes that will normally be rejected

- direct pushes or unreviewed large changes to `main`;
- hard-coded personal paths, credentials, or private infrastructure details;
- a parallel framework that bypasses the existing config and factory contracts;
- a giant PR combining unrelated runtime, cleanup, docs, and research work;
- generated, copied, or AI-produced documentation that was not checked against
  the repository;
- claims of accuracy, efficiency, SOTA status, or universal compatibility without
  reproducible evidence;
- code copied from another project without source and license information;
- tests that only suppress, skip, or catch the failure being fixed;
- raw dataset or model artifacts without a license and repository-storage review.

## Community and licensing

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). By
submitting a contribution, you agree that it may be distributed under the
repository's [Apache License 2.0](LICENSE), subject to any separately documented
third-party licenses.

For general questions, use GitHub Issues or Discussions when available. Keep
security and conduct reports out of public issue threads.

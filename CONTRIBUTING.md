# Contributing to PHM-Vibench

<div align="center">
  <a href="CONTRIBUTING.md"><strong>English</strong></a> |
  <a href="CONTRIBUTING_CN.md">中文</a>
</div>

Thank you for contributing. PHM-Vibench is configuration-first and factory-driven;
contributions should make one behavior clearer, more reliable, or better supported
without creating a parallel framework.

Before starting, read:

- [Documentation index](docs/index.md)
- [Developer guide](docs/developer_guide.md)
- [Testing and evidence](docs/testing.md)
- [Configuration system](configs/README.md)
- [Known limitations](KNOWN_LIMITATIONS.md)

The maintained runtime contract is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The maintained configuration sections remain:

```text
environment / data / model / task / trainer
```

## Ways to contribute

You can help by:

- reporting a reproducible bug;
- proposing a bounded feature;
- fixing runtime, configuration, test, documentation, or CI defects;
- adding or improving a dataset reader, model, task, trainer, sampler, or config;
- improving documentation without duplicating an existing source of truth;
- reviewing compatibility, reproducibility, data licensing, or release evidence.

Large architectural changes, new public pipelines, dataset redistribution, and
scientific-performance claims require discussion before implementation.

## Report a bug

Search existing issues first. A useful bug report includes:

- concise problem description;
- exact reproduction steps;
- expected and actual behavior;
- operating system and hardware;
- Python, PyTorch, CUDA, PyTorch Lightning, and relevant package versions;
- repository commit or release tag;
- config file and every CLI override;
- full command, exit code, traceback, and log;
- smallest reproducible data/config case that can legally be shared.

Capture environment details with:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

When possible, reproduce against the offline config:

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Do not report security vulnerabilities in a public issue. Follow
[SECURITY.md](SECURITY.md).

## Propose a feature

A feature proposal should explain:

- user or research scenario;
- current limitation;
- desired behavior and acceptance criteria;
- why the problem belongs in PHM-Vibench;
- simpler alternatives considered;
- affected factories, configs, tests, and documentation;
- backward-compatibility impact;
- dependency, runtime, data, and maintenance cost;
- what would remain explicitly unsupported.

A registry entry or paper reference alone is not sufficient justification for a
release-supported feature.

## Development setup

Use Python 3.10, matching the maintained documentation and CI baseline.

```bash
git clone https://github.com/<your-account>/PHM-Vibench.git
cd PHM-Vibench

conda create -n phm-vibench-dev python=3.10
conda activate phm-vibench-dev
python -m pip install -r requirements.txt
```

See [Installation](docs/installation.md) for CPU/CUDA and platform boundaries.

Verify the checkout before modifying it:

```bash
python main.py --help
python -m scripts.validate_configs
python -m scripts.validate_docs
```

## Branch and commit conventions

Create a focused branch from current `main`:

```bash
git switch main
git pull --ff-only origin main
git switch -c <type>/<short-topic>
```

Recommended prefixes:

```text
fix/       defect or regression
feature/   new bounded capability
docs/      documentation only
test/      tests/evidence only
ci/        workflow or automation
cleanup/   reviewed removal or consolidation
release/   release preparation only
```

Use clear imperative commit messages, preferably Conventional Commit style:

```text
fix: reject unknown pipeline overrides
docs: clarify external data setup
test: cover sampler metadata errors
```

Do not mix unrelated formatting, file moves, documentation cleanup, and runtime
behavior in one commit or pull request.

## Implement a code change

The normal flow is:

```text
create focused branch
→ make the smallest coherent change
→ add/update focused tests
→ update the authoritative documentation
→ run applicable validation
→ inspect the final diff
→ open a pull request
```

Preserve these boundaries:

- data integration: `src/data_factory/`;
- model construction: `src/model_factory/`;
- task/loss/metric logic: `src/task_factory/`;
- trainer construction: `src/trainer_factory/`;
- shared configuration: `configs/`;
- public entrypoint: `main.py`.

Do not add model-, task-, dataset-, or trainer-specific branches to `main.py`.
Do not create a second config loader, registry, or training framework.

A behavior change requires one of:

- backward-compatible behavior;
- a compatibility alias or adapter;
- an explicit migration note and deprecation path.

Silent fallback is not a compatibility strategy. Invalid combinations should fail
early with actionable errors.

## Contribute a dataset or reader

Read:

- [Data directory policy](data/README.md)
- [Custom dataset tutorial](docs/custom_dataset.md)
- [Data factory contribution guide](src/data_factory/contributing.md)

Provide:

- original source and stable download identifier;
- dataset license and redistribution constraints;
- raw and processed formats;
- metadata fields and units;
- preprocessing/windowing/splitting procedure;
- reader implementation and registry/factory trace;
- small legal fixture or synthetic contract fixture;
- config under `configs/experiments/` first;
- minimal inspection and runtime command;
- expected output structure, not an invented performance number;
- known limitations and reproducibility notes.

Do not commit a full external dataset or a personal absolute data path unless the
repository policy explicitly authorizes it.

## Contribute a model

Read [Model factory contribution guide](src/model_factory/contributing.md).

Provide:

- implementation module and public model identity;
- input/output, shape, dtype, and device contract;
- constructor parameters and defaults;
- model registry/config trace;
- focused construction and forward tests;
- task/loss compatibility and rejected combinations;
- checkpoint behavior when relevant;
- minimal experimental config and smoke command;
- source paper/license when porting an external implementation;
- limitations and unsupported modes.

Start with `configs/experiments/`. Promotion to `configs/demo/` requires runtime
evidence, registry status, documentation, and maintainer review.

## Contribute a task, sampler, trainer, or pipeline

Use the relevant local guide:

- [Task factory](src/task_factory/contributing.md)
- [Trainer factory](src/trainer_factory/contributing.md)
- [Data samplers](src/data_factory/samplers/README.md)

Document the batch contract, model output contract, loss/metric behavior,
configuration parameters, device behavior, and invalid combinations.

A new pipeline is justified only when existing pipelines cannot express a
coherent runtime stage. It must still use the five configuration sections and
existing factories.

## Contribute a configuration

Local or research variants belong under `configs/experiments/`.

Before proposing a maintained demo:

```bash
python -m scripts.config_inspect --config <yaml> --override trainer.num_epochs=1
python -m scripts.validate_configs
python main.py --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

For promotion, update:

- `configs/config_registry.csv`;
- relevant config README;
- `docs/CONFIG_ATLAS.md`, generated with `python -m scripts.gen_config_atlas`;
- `SUPPORTED_COMPONENTS.md`, `SUPPORTED_COMBINATIONS.md`, or
  `KNOWN_LIMITATIONS.md` only when the public support boundary actually changes.

Do not mark a config `sanity_ok` before the stated smoke command has passed.

## Contribute documentation

Choose the reader and authoritative location first:

- project positioning and shortest path: `README.md`;
- installation: `docs/installation.md`;
- first run: `docs/quickstart.md`;
- configuration: `configs/README.md`;
- data policy: `data/README.md`;
- testing/evidence: `docs/testing.md`;
- architecture/development: `docs/developer_guide.md`;
- contribution process: `CONTRIBUTING.md`;
- component-local detail: nearest maintained `README.md`;
- history/research: clearly marked historical or research location.

Documentation rules:

- do not copy an existing procedure into a second page; link to it;
- validate every command, path, config key, and relative link;
- define an abbreviation on first use;
- distinguish maintained, experimental, planned, deprecated, and historical;
- do not add unsupported performance, scale, compatibility, or status claims;
- add new pages to `docs/index.md` or the nearest local navigation page;
- update docs in the same PR when code behavior changes.

English and Chinese root/contribution pages should keep the same structure and
support boundary. Deep technical pages may remain English-only rather than
creating an unmaintained partial translation.

## Run tests and quality checks

Use the narrowest relevant test, then the broader maintained gate. See
[Testing and evidence](docs/testing.md).

Common checks:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m pytest test/ -q
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
git diff --check
```

If a gate is not applicable, explain why. If it cannot be executed, report
`NOT_EXECUTED` and the exact limitation. Never convert missing data/dependency or
a skipped test into a pass.

## Pull request requirements

A pull request should include:

- problem and motivation;
- exact scope and non-goals;
- files and public behavior changed;
- compatibility and migration impact;
- commands run with results and environment;
- tests added or updated;
- documentation and registry changes;
- risks, known limitations, and rollback method;
- evidence paths or attached CI artifacts;
- related issue, paper, dataset, or design source when applicable.

Before requesting review:

```bash
git status --short
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
```

Keep the diff reviewable. Prefer squash merge for a focused PR after required
checks and maintainer review.

## Changes that will not be accepted as-is

- personal absolute paths, credentials, caches, logs, or local goal packs;
- broad unrelated refactors without behavior-equivalence evidence;
- a new public demo without a passing smoke path;
- weakened tests or broad skips used to hide a failure;
- copied third-party code/data without source and license;
- generated results presented as benchmark evidence without provenance;
- duplicate configuration loaders, factories, registries, or entrypoints;
- mass deletion of history/research files without inventory and recovery evidence;
- AI-generated filler, placeholder contacts, or unverified claims.

## Community and conduct

Be respectful, precise, and constructive. The repository's conduct expectations
are defined in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Questions that are not bugs may be raised in GitHub Discussions. Security reports
must follow [SECURITY.md](SECURITY.md).

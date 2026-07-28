# PHM-Vibench Developer Guide

This guide describes the maintained development path for PHM-Vibench. Start from
the configuration-first runtime contract and extend the existing factories rather
than adding special cases to `main.py`.

```bash
python main.py --config <yaml> [--override key=value ...]
```

## Architecture

```text
main.py
  └── configured pipeline
      ├── data factory
      ├── model factory
      ├── task factory
      └── trainer factory
```

Primary implementation areas:

- `src/data_factory/`: metadata loading, readers, datasets, samplers, and data construction
- `src/model_factory/`: model families, component registries, and model construction
- `src/task_factory/`: task implementations and task registry
- `src/trainer_factory/`: trainer implementations
- `configs/`: shared base blocks, maintained demos, experiments, and config registry
- `test/`: maintained pytest gate
- `apps/streamlit/`: optional interface around the public CLI contract

## Sources of truth

Use the following files before changing behavior:

- `configs/config_registry.csv`: maintained config inventory
- `docs/CONFIG_ATLAS.md`: generated view of the config registry
- `src/model_factory/model_registry.csv`: discovered model inventory
- `src/task_factory/task_registry.csv`: discovered task inventory
- `SUPPORTED_COMPONENTS.md`: release-supported component boundary
- `SUPPORTED_COMBINATIONS.md`: maintained model/task/config combinations
- `KNOWN_LIMITATIONS.md`: explicit runtime and evidence limitations

A registry entry records discoverability. It does not establish release support
without a maintained config and runtime evidence.

## Branch topology

`main` is the user-facing stable branch. `dev` is the integration and development
branch. Routine work branches from and returns to `dev`:

```text
feat/* | fix/* | docs/* | test/* | ci/* | cleanup/* | migration/*
                              ↓
                             dev
                              ↓ explicit release promotion only
                            main
```

Only an authorized release promotion or emergency hotfix targets `main`. Hotfixes
must be synchronized back to `dev`. Direct pushes to either long-lived branch are
outside the maintained workflow. Canonical v0.3 PR #127 is the one transition
exception because it predates this policy and must preserve its direct-to-`main`
ancestry contract.

Repository administrators should protect both long-lived branches. At minimum,
changes should require a pull request, the applicable checks, and maintainer review;
force pushes and branch deletion should be disabled. This document records the
required policy, but it does not itself prove that every GitHub branch-protection
setting has been configured.

## Development workflow

1. Update local `dev` and create a focused branch that will open a PR back to `dev`.
2. Identify the nearest maintained demo under `configs/demo/`.
3. Put local experiment variants under `configs/experiments/`.
4. Change one coherent capability at a time.
5. Add or update a focused test near the affected contract.
6. Run config inspection and the smallest applicable smoke command.
7. Update registries, generated docs, and support documentation only when the
   capability is intentionally promoted to the maintained surface.
8. Open one reviewable pull request to `dev` with exact validation commands and results.

Typical setup:

```bash
git switch dev
git pull --ff-only origin dev
git switch -c <type>/<short-topic>
```

Do not combine runtime changes, broad documentation cleanup, data artifact
removal, and research roadmaps in one pull request.

## Configuration changes

Maintained configs use five logical blocks:

```yaml
environment: {}
data: {}
model: {}
task: {}
trainer: {}
```

Inspect a resolved config and its field sources before changing code:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

Validate the maintained config registry and generated atlas:

```bash
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
```

## Extending a factory

Use the factory-specific contribution guides:

- Data and readers: `src/data_factory/contributing.md`
- Models: `src/model_factory/contributing.md`
- Tasks: `src/task_factory/contributing.md`
- Trainers: `src/trainer_factory/contributing.md`

A public component extension should normally include:

- implementation within the correct factory boundary;
- registry and configuration entry;
- documented input/output or batch contract;
- focused import, assembly, or behavior test;
- maintained or explicitly experimental config;
- exact validation command;
- clear failure behavior without silent fallback.

Do not modify `main.py` merely to add a dataset, model, task, or trainer. Add a
new pipeline only when the existing pipeline contract cannot express a coherent
runtime stage, and document that boundary separately.

## Data changes

The repository-shipped dummy data is the only fully offline maintained path.
Non-dummy demos require local metadata/raw data and may need a
`data.data_dir` override. See `data/README.md` and
`configs/base/data/README.md` before changing data layout or readers.

Reader changes should make raw-to-runtime shape handling explicit. Dataset split,
normalization, metadata requirements, and fallback behavior must be reviewable;
they should not be hidden inside a demo-specific branch.

## Tests and validation

Use focused tests during implementation. Before merging a runtime or config
change, run the maintained gate where applicable:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
python -m pytest test/ -q
```

A documentation-only pull request may use a narrower gate, but its description
must explain why runtime tests are not applicable. Local command output is local
evidence, not GitHub Actions evidence.

## Streamlit development

The maintained optional UI lives under `apps/streamlit/`. See the
[Streamlit application guide](../apps/streamlit/README.md).

Streamlit code should:

- preserve `python main.py --config ...` as the execution contract;
- remain optional for CLI users;
- avoid importing pipeline internals directly;
- use argument lists rather than shell command strings;
- keep process lifecycle and result discovery bounded and testable.

## Research and historical material

Research ideas, paper workflows, historical configs, and unverified components
must remain clearly separated from the release-supported core. Promote them only
through a small pull request with a protocol, config, test, and runtime evidence.

## Pull request checklist

Before requesting review, confirm:

- the pull request targets `dev` unless it is an explicitly authorized release or hotfix;
- the diff implements one coherent capability;
- no machine-specific paths or personal tooling are introduced;
- public CLI and five-block config contracts remain intact;
- failures are explicit rather than silently falling back;
- registry and generated documentation are synchronized where applicable;
- validation commands and outcomes are recorded accurately;
- unsupported performance or compatibility claims are not added.

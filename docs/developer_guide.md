# Develop PHM-Vibench

This guide is the architecture and extension map. It does not duplicate the
installation, first-run, configuration, testing, or contribution procedures:

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Configuration system](../configs/README.md)
- [Testing and evidence](testing.md)
- [Contributor guide](../CONTRIBUTING.md)

The public runtime contract is:

```bash
python main.py --config <yaml> [--override key=value ...]
```

## Runtime architecture

```text
main.py
  └── pipeline selected by config/override
      ├── data factory
      ├── model factory
      ├── task factory
      ├── trainer factory
      └── fit / test / artifact lifecycle
```

Maintained configuration remains divided into:

```text
environment / data / model / task / trainer
```

Do not create a second config system, registry, or training framework. Do not add
component-specific branches to `main.py`.

## Implementation boundaries

| Location | Responsibility | Local contribution guide |
|---|---|---|
| `src/data_factory/` | metadata, readers, dataset wrappers, samplers, loaders | [`contributing.md`](../src/data_factory/contributing.md) |
| `src/model_factory/` | model families, components, construction, checkpoints | [`contributing.md`](../src/model_factory/contributing.md) |
| `src/task_factory/` | task modules, losses, metrics, optimizers, task registry | [`contributing.md`](../src/task_factory/contributing.md) |
| `src/trainer_factory/` | Lightning trainer construction, callbacks, loggers, extensions | [`contributing.md`](../src/trainer_factory/contributing.md) |
| `configs/` | base composition, demos, experiments, registry | [`README.md`](../configs/README.md) |
| `apps/streamlit/` | optional UI/process/result adapter around the CLI | [`README.md`](../apps/streamlit/README.md) |
| `test/` | maintained pytest gate | [Testing guide](testing.md) |

Shared helpers belong under the narrowest existing utility/component module. An
abstraction is useful only when input/output, state, lifecycle, and error
semantics are genuinely shared.

## Sources of truth

| Information | Source |
|---|---|
| Shipped config inventory | `configs/config_registry.csv` |
| Generated config reference | `docs/CONFIG_ATLAS.md` |
| Model discovery inventory | `src/model_factory/model_registry.csv` |
| Task discovery inventory | `src/task_factory/task_registry.csv` |
| Dataset-task mapping | `src/data_factory/dataset_task/dataset_task_mapping.csv` |
| Release-supported components | `SUPPORTED_COMPONENTS.md` |
| Release-supported combinations | `SUPPORTED_COMBINATIONS.md` |
| Explicit limits | `KNOWN_LIMITATIONS.md` |
| Test/evidence terminology | `docs/testing.md` |

Discovery is not support. A model/task/config row requires compatible tests and a
maintained runtime path before it belongs in the support documents.

## Standard extension flow

1. Update `main` and create one focused branch.
2. Identify the nearest maintained component and demo.
3. Put a new config under `configs/experiments/`.
4. Define input/output, batch, parameter, error, and side-effect contracts.
5. Make the smallest coherent code change inside the existing factory boundary.
6. Add focused positive and negative regression tests.
7. Inspect the resolved config and run the smallest applicable smoke path.
8. Update the authoritative local documentation.
9. Promote registry/demo/support status only after evidence exists.
10. Open one reviewable pull request with commands, results, limits, and rollback.

Avoid combining runtime behavior, broad docs cleanup, case-only renames, data
artifact removal, and research planning in one pull request.

## Add or change configuration behavior

The configuration guide defines precedence and promotion rules. Changes to the
loader or schema should test:

- base/YAML/local/CLI precedence;
- type conversion and null/list/bool behavior;
- aliases and deprecation;
- unknown or misspelled keys;
- source reporting in `scripts.config_inspect`;
- generated registry/Atlas consistency;
- backward compatibility for maintained configs.

A config inspector pass is not an end-to-end pass.

## Add a component

Every public component should have:

- stable identity and import/factory trace;
- constructor/config contract;
- input/output or batch contract;
- dtype/device behavior;
- compatible and rejected combinations;
- focused tests that assert behavior;
- an experimental config;
- minimal runtime evidence;
- dependency/license/provenance notes;
- known limitations.

Optional dependencies should load only when the selected component needs them.
Importing a lightweight model should not require every optional research model.

## Add a pipeline

Add a pipeline only when existing pipelines cannot express a coherent stage or
lifecycle. A new pipeline must still:

- use the public CLI;
- consume the five configuration sections;
- construct components through existing factories;
- reject unsupported combinations early;
- define expected outputs and side effects;
- have config inspection, focused tests, and a mini end-to-end path;
- remain experimental until its support boundary is documented.

The [Pipeline 06 migration contract](PIPELINE_06_GENERATIVE_MIGRATION.md) is an
example of defining gates before promoting runtime code.

## Data and scientific evidence

Only the Dummy smoke path is fully repository-shipped. External-data claims must
record source, license, metadata version, preprocessing, split/leakage controls,
seed, environment, config, overrides, and evidence paths.

Synthetic data validates software contracts only. Do not infer benchmark quality,
state-of-the-art performance, or dataset correctness from a smoke run.

## Optional Streamlit development

The Streamlit application must remain an adapter around the public CLI. It should
not import pipeline internals or become a scheduler. Process management, path
safety, config precedence, and artifact scanning must remain isolated and tested.
See [Streamlit architecture](../apps/streamlit/README.md).

## Historical and research material

`docs/v0.1.0/`, `docs/past/`, `dev/`, `paper/`, `.claude/`, `.codex/`, and old
plan trees contain evidence or research material. They are not current user
instructions and do not expand release support. Preserve provenance; move or
delete only with inventory, reference checks, and recovery evidence.

## Before requesting review

Follow the [pull request requirements](../CONTRIBUTING.md#pull-request-requirements)
and run the applicable commands from [Testing and evidence](testing.md). Report
`NOT_EXECUTED` for unavailable gates; never claim missing data, missing dependency,
skipped tests, or local output as a CI pass.

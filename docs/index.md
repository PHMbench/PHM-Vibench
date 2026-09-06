# Documentation

Use the shortest page that answers the current task. Historical plans and audits are not
runtime or user guidance.

## Start here

1. [README](../README.md) — project scope and shortest successful path.
2. [Quickstart](quickstart.md) — complete offline first run.
3. [Installation](installation.md) — Python, PyTorch, and platform setup.
4. [Configuration](../configs/README.md) — config composition and overrides.
5. [Known limitations](../KNOWN_LIMITATIONS.md) — current unsupported or unresolved work.

## User tasks

| Task | Document |
| --- | --- |
| Run the bundled offline experiment | [Quickstart](quickstart.md) |
| Prepare local data | [Data layout](../data/README.md) |
| Add a compatible dataset | [Custom dataset guide](custom_dataset.md) |
| Select or add a model | [Model Factory](../src/model_factory/README.md) |
| Select or add a task | [Task Factory](../src/task_factory/README.md) |
| Configure training | [Trainer Factory](../src/trainer_factory/README.md) |
| Use the browser workspace | [Streamlit](../apps/streamlit/README.md) |
| Inspect the MFPT candidate | [MFPT guide](../configs/baselines/01_mfpt/README.md) |

## Development

- [Core contract](../CORE.md)
- [Contributing](../CONTRIBUTING.md)
- [Developer guide](developer_guide.md)
- [Testing](testing.md)
- [Data Factory contribution guide](../src/data_factory/contributing.md)
- [Model Factory contribution guide](../src/model_factory/contributing.md)
- [Task Factory contribution guide](../src/task_factory/contributing.md)
- [Trainer Factory contribution guide](../src/trainer_factory/contributing.md)

Current behavior is defined by the latest code and tests. `CORE.md` records the stable
engineering and scientific constraints. Do not create another plan hierarchy or repeat
the same contract in new documents.

## Support and release

- [Configuration registry](../configs/config_registry.csv)
- [Configuration Atlas](CONFIG_ATLAS.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [Release readiness](PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [v0.2 to v0.3 migration](../MIGRATION_v0.2_to_v0.3.md)

The source version is `0.3.0rc1`, but the release gate remains blocked until a current
real-data experiment is requalified as `baseline_valid`.

## Optional and experimental paths

- [CWRU data bundle](CWRU_DEMO_V0_3.md)
- [Dependency boundaries](DEPENDENCY_BOUNDARIES_V0_3.md)
- [Deferred data backend](PHM_DATA_FACTORY_BACKEND_V0_3.md)
- [Pipeline 06 migration](PIPELINE_06_GENERATIVE_MIGRATION.md)
- [HPC notes](HPC.md)

These documents describe optional or experimental work. They do not change the default
local-file path or the support status of an exact configuration.

## Historical material

Historical audits live under [`docs/archive/`](archive/README.md). Obsolete implementation
plans are preserved by Git history rather than kept beside current user documentation.

`configs/v0.0.9/` is compatibility material, not the maintained quickstart.

## Single-authority map

| Information | Source |
| --- | --- |
| Product scope and first command | `README.md` |
| First complete run | `docs/quickstart.md` |
| Installation | `docs/installation.md` |
| Configuration semantics | `configs/README.md` |
| Scientific and engineering constraints | `CORE.md` |
| Maintained config inventory | `configs/config_registry.csv` |
| Current limitations | `KNOWN_LIMITATIONS.md` |
| Release blockers | `docs/PHMFACTORY_V0_3_RELEASE_READINESS.md` |
| Contribution and tests | `CONTRIBUTING.md`, `docs/testing.md` |
| Streamlit behavior | `apps/streamlit/README.md` |

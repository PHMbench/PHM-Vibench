# Documentation Index

This is the maintained navigation for PHMFactory. Start with the user path; consult
maintainer and historical material only when that work is relevant.

## Start here

1. [README](../README.md) — product scope and shortest successful path.
2. [Core contract](../CORE.md) — scientific semantics, Factory boundaries, and Occam rules.
3. [Quickstart](quickstart.md) — complete offline first run.
4. [Installation](installation.md) — platform and environment setup.
5. [Configuration](../configs/README.md) — composition, overrides, and maintained configs.
6. [Known limitations](../KNOWN_LIMITATIONS.md) — current unsupported or unresolved areas.

## User tasks

| Task | Authority |
| --- | --- |
| Run the bundled offline experiment | [Quickstart](quickstart.md) |
| Inspect maintained configurations | [Configuration Atlas](CONFIG_ATLAS.md) |
| Check exact supported combinations | [SUPPORTED_COMBINATIONS.md](../SUPPORTED_COMBINATIONS.md) |
| Prepare local data | [Data layout](../data/README.md) |
| Add a compatible dataset | [Custom dataset guide](custom_dataset.md) |
| Select or add a model | [Model Factory](../src/model_factory/README.md) |
| Select or add a task | [Task Factory](../src/task_factory/README.md) |
| Configure training | [Trainer Factory](../src/trainer_factory/README.md) |
| Use the browser workspace | [Streamlit workspace](../apps/streamlit/README.md) |
| Inspect the MFPT candidate | [MFPT guide](../configs/baselines/01_mfpt/README.md) |

## Development

- [Contributing](../CONTRIBUTING.md)
- [Developer guide](developer_guide.md)
- [Testing](testing.md)
- [Data Factory contribution guide](../src/data_factory/contributing.md)
- [Model Factory contribution guide](../src/model_factory/contributing.md)
- [Task Factory contribution guide](../src/task_factory/contributing.md)
- [Trainer Factory contribution guide](../src/trainer_factory/contributing.md)

Development authority is intentionally small:

```text
latest dev code/tests
→ CORE.md
→ README/Quickstart
→ config registry and generated support tables
```

Do not create another goal registry, policy tree, manifest family, or documentation
hierarchy to restate these sources.

## Release and migration

- [Current release readiness](PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [v0.3 release notes](../RELEASE_NOTES_v0.3.0.md)
- [v0.2 to v0.3 migration](../MIGRATION_v0.2_to_v0.3.md)
- [Backend deferral](releases/v0.3.0-backend-deferral.yaml)
- [CWRU compatibility contract](CWRU_DEMO_V0_3.md)

The source version is `0.3.0rc1`, but the current release gate is blocked until an exact
real-data experiment is requalified as `baseline_valid` on the current source. Release
documents do not imply a tag or publication.

## Generated authority

- [`docs/CONFIG_ATLAS.md`](CONFIG_ATLAS.md) is generated from
  [`configs/config_registry.csv`](../configs/config_registry.csv).
- [`SUPPORTED_COMPONENTS.md`](../SUPPORTED_COMPONENTS.md) and
  [`SUPPORTED_COMBINATIONS.md`](../SUPPORTED_COMBINATIONS.md) are generated from the
  registry and resolved maintained configurations.

Update the source registry first; regenerate only the documents affected by that source.
Do not hand-edit generated claims into a stronger status.

## Historical material

Historical audits and migration evidence live under [`docs/archive/`](archive/README.md).
They preserve earlier facts but do not override current code, `CORE.md`, the registry,
known limitations, or release readiness.

`configs/v0.0.9/` is historical compatibility material, not the maintained quickstart.
Old release provenance, submodule inventories, and past task plans should not appear in the
normal user reading path.

## Single-authority map

| Information | Authority |
| --- | --- |
| Project positioning and first command | `README.md` |
| Scientific/engineering invariants | `CORE.md` |
| First complete run | `docs/quickstart.md` |
| Installation | `docs/installation.md` |
| Configuration semantics | `configs/README.md` |
| Maintained config inventory | `configs/config_registry.csv` |
| Supported components/combinations | generated support documents |
| Current limitations | `KNOWN_LIMITATIONS.md` |
| Release blockers | `docs/PHMFACTORY_V0_3_RELEASE_READINESS.md` |
| Contribution and tests | `CONTRIBUTING.md`, `docs/testing.md` |
| Optional Streamlit behavior | `apps/streamlit/README.md` |

# Documentation Index

This index is the maintained entry point for PHMFactory documentation.

## Start here

- [README](../README.md)
- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Configuration](../configs/README.md)
- [Supported components](../SUPPORTED_COMPONENTS.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [v0.3.0 release and migration notes](../RELEASE_NOTES_v0.3.0.md)
- [v0.3.0 release readiness](PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [v0.2.0 release-candidate provenance](releases/v0.2.0-rc-provenance.md)

## Architecture and extension

- [Developer guide](developer_guide.md)
- [Configuration Atlas](CONFIG_ATLAS.md)
- [Data Factory](../src/data_factory/README.md)
- [Model Factory](../src/model_factory/README.md)
- [Task Factory](../src/task_factory/README.md)
- [Trainer Factory](../src/trainer_factory/README.md)

## User interfaces

- [Streamlit workspace](../apps/streamlit/README.md)
- [CWRU v0.3 demo contract](CWRU_DEMO_V0_3.md)

## Contribution and governance

- [Contributing](../CONTRIBUTING.md)
- [Testing](testing.md)
- [Security](../SECURITY.md)
- [Code of conduct](../CODE_OF_CONDUCT.md)
- [Repository optimization SOP](REPOSITORY_OPTIMIZATION_SOP.md)
- [PHMFactory v0.3.0 repository contract](PHMFACTORY_V0_3_REPOSITORY_CONTRACT.md)
- [PHMFactory v0.3.0 reader preservation contract](PHMFACTORY_V0_3_READER_PRESERVATION.md)
- [PHMFactory v0.3.0 task and PR plan](PHMFACTORY_V0_3_TASK_PLAN.md)
- [Pipeline 06 migration contract](PIPELINE_06_GENERATIVE_MIGRATION.md)

The PHMFactory v0.3.0 documents describe a staged migration and pre-release state.
They do not by themselves claim that the GitHub repository rename, public data pins,
final version, tag, or package publication have completed. The release-readiness page
is the authority for those blockers.

## Generated and historical material

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`; update the
registry, run `python -m scripts.gen_config_atlas`, and commit both changes.

Earlier release material remains under:

- `docs/v0.1.0/`
- `docs/past/`
- `configs/v0.0.9/`

The removed `dev/` and `.archive/` workspaces are preserved in the approved
personal-fork archive and in Git history. They are not public framework inputs.
Generated experiment outputs are local artifacts and are not tracked through
`results/` or `metrics_reports/` placeholder directories.

## Single-source-of-truth map

| Information | Authority | Other documents should do |
| --- | --- | --- |
| Project positioning and shortest successful path | `README.md` | Link to the README or the detailed page below |
| Installation | `docs/installation.md` | Keep only a minimal command and link |
| First successful run | `docs/quickstart.md` | Avoid copying the complete walkthrough |
| v0.2 release-candidate baseline | `docs/releases/v0.2.0-rc-provenance.yaml` | Do not create or imply a retroactive final v0.2.0 tag |
| v0.2 to v0.3 user migration | `RELEASE_NOTES_v0.3.0.md` | Link rather than duplicate the full migration map |
| Release blockers | `docs/PHMFACTORY_V0_3_RELEASE_READINESS.md` | Do not claim release completion while blockers remain |
| Configuration semantics | `configs/README.md` | Link rather than redefine precedence |
| Maintained config inventory | `configs/config_registry.csv` | Regenerate `docs/CONFIG_ATLAS.md` |
| Supported components and combinations | `SUPPORTED_COMPONENTS.md`, `SUPPORTED_COMBINATIONS.md` | Do not infer support from registry presence |
| Known limitations | `KNOWN_LIMITATIONS.md` | Link rather than duplicate caveats |
| Testing and contribution gates | `CONTRIBUTING.md`, `docs/testing.md` | Keep commands synchronized |
| Streamlit UI contract | `apps/streamlit/README.md` | Keep UI-specific behavior there |

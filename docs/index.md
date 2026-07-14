# PHM-Vibench Documentation

This page is the maintained navigation entry for PHM-Vibench documentation.
Start with the shortest path that matches your role. Historical, paper, and
agent-workflow material is intentionally separated from the current user path.

## New users

1. Read the [project overview](../README.md) or
   [Chinese overview](../README_CN.md).
2. [Install PHM-Vibench](installation.md) in a Python 3.10 environment.
3. [Run the repository-shipped offline experiment](quickstart.md).
4. Learn configuration composition and overrides in the
   [configuration guide](../configs/README.md).
5. Confirm the current support boundary in
   [supported components](../SUPPORTED_COMPONENTS.md),
   [supported combinations](../SUPPORTED_COMBINATIONS.md), and
   [known limitations](../KNOWN_LIMITATIONS.md).
6. For external data, read the [data directory policy](../data/README.md).

Common failures are covered in [Troubleshooting](troubleshooting.md).

## Configuration reference

- [Configuration guide](../configs/README.md)
- [Generated configuration atlas](CONFIG_ATLAS.md)
- [Configuration registry schema](config_registry_schema.md)
- [Base configuration blocks](../configs/base/README.md)
- [Maintained demos](../configs/demo/README.md)
- [Local experiment configs](../configs/experiments/README.md)
- [Reference and unverified configs](../configs/reference/README.md)

`configs/config_registry.csv` is the authoritative configuration inventory.
`docs/CONFIG_ATLAS.md` is generated from it and should not be edited manually.

## Optional application

- [Streamlit user workflow](app_usage.md)
- [Streamlit architecture, testing, and extension guide](../apps/streamlit/README.md)

The Streamlit workspace is an optional adapter around the same
`python main.py --config ...` command; it is not a second training framework.

## Developers and contributors

- [Contributor guide](../CONTRIBUTING.md)
- [Chinese contributor guide](../CONTRIBUTING_CN.md)
- [Developer guide](developer_guide.md)
- [Testing and evidence guide](testing.md)
- [Custom dataset tutorial](custom_dataset.md)
- [Data factory contribution guide](../src/data_factory/contributing.md)
- [Model factory contribution guide](../src/model_factory/contributing.md)
- [Task factory contribution guide](../src/task_factory/contributing.md)
- [Trainer factory contribution guide](../src/trainer_factory/contributing.md)
- [Maintainer runbook](../AGENTS.md)
- [Architecture/change constraints](../CLAUDE.md)

## Release and policy

- [Changelog](../CHANGELOG.md)
- [v0.2.0 release notes](../RELEASE_NOTES_v0.2.0.md)
- [v0.1 to v0.2 migration guide](../MIGRATION_v0.1_to_v0.2.md)
- [Security policy](../SECURITY.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Citation metadata](../CITATION.cff)
- [Apache License 2.0](../LICENSE)

## Advanced and design material

- [Pipeline 06 generative migration contract](PIPELINE_06_GENERATIVE_MIGRATION.md)
- [Repository optimization SOP](REPOSITORY_OPTIMIZATION_SOP.md)
- [Branch-governance record](branch_governance_20260709.md)
- [HPC notes](HPC.md)

These pages describe advanced operation, design constraints, or future migration
work. Their presence does not expand the release-supported component matrix.

## Historical and research material

The following locations are evidence or research records, not current user
instructions:

- `docs/v0.1.0/`
- `docs/past/`
- `src/configs/plan/`
- `dev/`
- `paper/`
- `.claude/` and `.codex/`

Do not copy commands from these locations into current documentation without
checking them against the current code, configuration registry, and tests.

## Documentation governance

The [documentation audit](DOCUMENTATION_AUDIT.md) records current duplication,
conflicts, planned single sources of truth, and cleanup constraints.

Run the maintained documentation check from the repository root:

```bash
python -m scripts.validate_docs
```

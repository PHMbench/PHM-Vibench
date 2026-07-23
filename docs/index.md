# PHM-Vibench Documentation

This page is the navigation entrypoint for maintained PHM-Vibench documentation.
The public runtime contract is configuration-first:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Documentation uses four status labels:

- **Maintained** — expected to match current `main` and covered by repository checks.
- **Experimental** — implemented or under active development, but outside the release-supported surface.
- **Historical** — retained as evidence of earlier decisions or releases; not current instructions.
- **Generated** — produced from another source of truth and should not be edited manually.

## Start here

| Need | Maintained entrypoint |
| --- | --- |
| Install PHM-Vibench | [Installation](installation.md) |
| Run the offline example | [Quickstart](quickstart.md) |
| Understand YAML composition and overrides | [Configuration guide](../configs/README.md) |
| Find maintained configurations | [Configuration atlas](CONFIG_ATLAS.md) |
| Prepare local datasets | [Data directory guide](../data/README.md) |
| See the supported release surface | [Supported components](../SUPPORTED_COMPONENTS.md) and [supported combinations](../SUPPORTED_COMBINATIONS.md) |
| Check known constraints | [Known limitations](../KNOWN_LIMITATIONS.md) |
| Use the optional web interface | [Streamlit workspace](../apps/streamlit/README.md) |
| Troubleshoot a first run | [Quickstart troubleshooting](quickstart.md#troubleshooting) |

## Develop and contribute

| Task | Maintained entrypoint |
| --- | --- |
| Development workflow and architecture | [Developer guide](developer_guide.md) |
| Tests and validation gates | [Testing guide](testing.md) |
| General contribution process | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Community standards | [Code of Conduct](../CODE_OF_CONDUCT.md) |
| Add a dataset or reader | [Data factory contribution guide](../src/data_factory/contributing.md) |
| Add a model | [Model factory contribution guide](../src/model_factory/contributing.md) |
| Add a task | [Task factory contribution guide](../src/task_factory/contributing.md) |
| Add or change a trainer | [Trainer factory contribution guide](../src/trainer_factory/contributing.md) |
| PHMFactory v0.3 migration constraints | [Repository contract](PHMFACTORY_V0_3_REPOSITORY_CONTRACT.md) |
| Repository optimization process | [Repository optimization SOP](REPOSITORY_OPTIMIZATION_SOP.md) |

## Releases and governance

- [Changelog](../CHANGELOG.md)
- [v0.2.0 release notes](../RELEASE_NOTES_v0.2.0.md)
- [v0.1 to v0.2 migration guide](../MIGRATION_v0.1_to_v0.2.md)
- [Security policy](../SECURITY.md)
- [Citation metadata](../CITATION.cff)
- [Apache License 2.0](../LICENSE)
- [Branch governance record](branch_governance_20260709.md)
- [Repository optimization SOP](REPOSITORY_OPTIMIZATION_SOP.md)
- [PHMFactory v0.3.0 repository contract](PHMFACTORY_V0_3_REPOSITORY_CONTRACT.md)
- [PHMFactory v0.3.0 reader preservation contract](PHMFACTORY_V0_3_READER_PRESERVATION.md)
- [PHMFactory v0.3.0 task and PR plan](PHMFACTORY_V0_3_TASK_PLAN.md)
- [Pipeline 06 migration contract](PIPELINE_06_GENERATIVE_MIGRATION.md)

The PHMFactory v0.3.0 documents describe an accepted staged migration plan.
They do not by themselves claim that the repository, package, Pipeline files,
data providers, or GitHub repository have already been renamed or migrated.

## Generated and historical material

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`; update the
registry, run `python -m scripts.gen_config_atlas`, and commit both changes.

Earlier release material remains under:

- `docs/v0.1.0/`
- `docs/past/`
- `configs/v0.0.9/`

The removed `dev/` and `.archive/` workspaces are preserved in the approved
personal-fork archive and in Git history. They are not public framework inputs.

## Single-source-of-truth map

| Information | Authority | Other documents should do |
| --- | --- | --- |
| Project positioning and shortest successful path | `README.md` | Link to the README or the detailed page below |
| Installation | `docs/installation.md` | Keep only a minimal command and link |
| First successful run | `docs/quickstart.md` | Avoid copying the complete walkthrough |
| Configuration semantics | `configs/README.md` | Link rather than redefine precedence |
| Maintained config inventory | `configs/config_registry.csv` | Regenerate `docs/CONFIG_ATLAS.md` |
| Supported components and combinations | `SUPPORTED_COMPONENTS.md`, `SUPPORTED_COMBINATIONS.md` | Do not infer support from registry presence |
| Data layout and external-data boundary | `data/README.md` | Link from reader and dataset pages |
| Contribution process | `CONTRIBUTING.md` | Factory guides describe only factory-specific steps |
| Community behavior | `CODE_OF_CONDUCT.md` | Link instead of copying conduct policy |
| Security reporting | `SECURITY.md` | Do not disclose sensitive details in public templates |
| Citation metadata | `CITATION.cff` | Cite the exact release or commit used |
| Test commands and evidence terms | `docs/testing.md` | Link to the relevant gate |
| Optional Streamlit behavior | `apps/streamlit/README.md` | Keep compatibility pages as short redirects |
| v0.3 repository migration scope and boundaries | `docs/PHMFACTORY_V0_3_REPOSITORY_CONTRACT.md` | Do not infer completed implementation from an accepted plan |
| v0.3 protected reader policy | `docs/PHMFACTORY_V0_3_READER_PRESERVATION.md` | Do not modify protected behavior in cleanup PRs |
| v0.3 task order and PR boundaries | `docs/PHMFACTORY_V0_3_TASK_PLAN.md` | Keep each PR within its recorded scope |
| Release changes | `CHANGELOG.md` and release notes | Historical plans must not override release documents |

# docs/

Maintained documentation for PHM-Vibench. Keep this directory as the canonical
index for project-level docs; module-specific usage belongs in the module
`README.md` next to the code.

## Canonical docs

- Project overview and onboarding: [`../README.md`](../README.md)
- Config system and runnable templates: [`../configs/README.md`](../configs/README.md)
- Config atlas (generated): [`CONFIG_ATLAS.md`](CONFIG_ATLAS.md)
- Config registry schema: [`config_registry_schema.md`](config_registry_schema.md)
- Hydra compatibility notes: [`HYDRA_CONFIG.md`](HYDRA_CONFIG.md)
- Developer guide: [`developer_guide.md`](developer_guide.md)
- Custom dataset guide: [`custom_dataset.md`](custom_dataset.md)
- Testing guide: [`testing.md`](testing.md)
- HPC / Slurm notes: [`HPC.md`](HPC.md)

## Module docs

- Streamlit app usage: [`../app/README.md`](../app/README.md)
- GUI refactor notes: [`../app/README_GUI_Refactored.md`](../app/README_GUI_Refactored.md)
- Config package internals: [`../src/configs/README.md`](../src/configs/README.md)
- Data factory: [`../src/data_factory/README.md`](../src/data_factory/README.md)
- Model factory: [`../src/model_factory/README.md`](../src/model_factory/README.md)
- Task factory: [`../src/task_factory/README.md`](../src/task_factory/README.md)
- PHM generative task/module docs: [`../src/task_factory/task/generative/README.md`](../src/task_factory/task/generative/README.md)
- PHM generative model docs: [`../src/model_factory/generative_model/README.md`](../src/model_factory/generative_model/README.md)
- PHM generative component docs: [`../src/task_factory/Components/generative/README.md`](../src/task_factory/Components/generative/README.md)
- Trainer factory: [`../src/trainer_factory/README.md`](../src/trainer_factory/README.md)

## Archived docs

Historical docs (`docs/past/`, `docs/v0.1.0/`, `docs/reports/`) and the Chinese
translations (`*_CN.md`) were archived to `obsidian/history/docs/` during the
2026-07-05 cleanup. They are no longer in the tree — link to the maintained
docs above for current content.

## Consolidation notes

- `docs/app_usage.md` was merged into `app/README.md`.
- `docs/grace.md` and historical `docs/past/grace.md` were merged into `docs/HPC.md`.
- English is canonical; Chinese (`*_CN.md`) translations are archived, not
  maintained in-tree. Use `README.md` / `CONTRIBUTING.md` filenames for new docs.

## Generation / validation

See [`../AGENTS.md`](../AGENTS.md) for copy-paste commands:

- `python -m scripts.gen_config_atlas`
- `python -m scripts.validate_configs`
- `python -m scripts.validate_docs`

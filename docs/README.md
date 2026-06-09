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

- `docs/past/` contains historical v0.0.x guides retained for reference.
- `docs/v0.1.0/` contains release planning and completed migration notes.
- Do not copy archived guides into new docs. Link to the maintained docs above
  and move only still-current content into the canonical location.

## Consolidation notes

- `docs/app_usage.md` was merged into `app/README.md`.
- `docs/grace.md` and `docs/past/grace.md` were merged into `docs/HPC.md`.
- Lowercase duplicate docs were either migrated to `README_CN.md` when they
  contained Chinese/bilingual content, or removed when they were compatibility
  stubs. Use canonical `README.md` / `README_CN.md` / `CONTRIBUTING.md`
  filenames for new docs.

## Generation / validation

See [`../AGENTS.md`](../AGENTS.md) for copy-paste commands:

- `python -m scripts.gen_config_atlas`
- `python -m scripts.validate_configs`
- `python -m scripts.validate_docs`

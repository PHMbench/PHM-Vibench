# `configs/reference/` (legacy)

This directory contains legacy experiment configs that are not part of the maintained onboarding path.

- Use `configs/demo/` as the template source for this repo.
- `configs/reference/` is planned to be migrated/removed (likely into a paper/research submodule) to avoid confusion with
  the main `main.py --config configs/demo/...` workflow.

If you need to run one of these configs, validate it first:

```bash
python -m scripts.config_inspect --config <path_under_configs/reference/...yaml>
```

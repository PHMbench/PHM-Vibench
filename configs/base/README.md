# Base Config Blocks (`configs/base/`)

Base configs are reusable building blocks for the 5-block config model:
`environment` / `data` / `model` / `task` / `trainer`.

They are meant to be **composed** by demos (and your experiments) via top-level `base_configs`:

```yaml
base_configs:
  environment: "configs/base/environment/base.yaml"
  data: "configs/base/data/base_cross_domain.yaml"
  model: "configs/base/model/backbone_dlinear.yaml"
  task: "configs/base/task/dg.yaml"
  trainer: "configs/base/trainer/default_single_gpu.yaml"
```

## Readmes (Per Block)

- `configs/base/environment/README.md`
- `configs/base/data/README.md`
- `configs/base/model/README.md`
- `configs/base/task/README.md`
- `configs/base/trainer/README.md`

## How to Add a New Base Block

1) Create a YAML under the appropriate subfolder.
2) Add a new row to `configs/config_registry.csv` with `category=base_*`.
3) Regenerate atlas: `python -m scripts.gen_config_atlas`.


# Base configuration blocks

Base files are reusable fragments for the five sections: `environment`, `data`, `model`,
`task`, and `trainer`. A fragment is not a complete runnable experiment. Demos and
experiments compose them with `base_configs` and supply their own explicit values:

```yaml
base_configs:
  environment: "configs/base/environment/base.yaml"
  data: "configs/base/data/base_cross_domain.yaml"
  model: "configs/base/model/backbone_dlinear.yaml"
  task: "configs/base/task/dg.yaml"
  trainer: "configs/base/trainer/default_single_gpu.yaml"
```

Read the relevant section guide:
[environment](environment/README.md), [data](data/README.md), [model](model/README.md),
[task](task/README.md), or [trainer](trainer/README.md).

Add a fragment under its owning section and check the demos that compose it. Index a
maintained fragment in `configs/config_registry.csv`; regenerate the Atlas when that
source changes. A local prototype does not need to become a maintained example.

Keep machine paths and credentials out of shared fragments. Machine-local values must be
supplied with an explicit `--local-config` or `--override`; a file's presence does not
activate it. See the [configuration guide](../README.md).

Validate a complete consumer with `python -m scripts.validate_configs` and
`phmfactory preflight --config <yaml>`. Do not force a fragment to satisfy the complete
experiment schema on its own.

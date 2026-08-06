# Demo Configs (`configs/demo/`)

Maintained, runnable example configurations. Use these as the template source for new
experiments by copying the nearest example into `configs/experiments/`.

This directory is intended to remain beginner-runnable:

- configs use repository-shipped dummy data or clearly document required external data;
- every maintained demo is indexed in `configs/config_registry.csv`;
- the registry generates `docs/CONFIG_ATLAS.md` and the execution/protocol status tables;
- comments and documentation describe the executed task, sampler and objective rather
  than an intended or historical method.

For architecture and change guidance, see
[`docs/developer_runtime_control_plane.md`](../../docs/developer_runtime_control_plane.md)
and [`docs/developer_guide.md`](../../docs/developer_guide.md).

## Fastest Start (No External Data)

- Registry id: `demo_00_smoke_dummy_dg`
- Config: `configs/demo/00_smoke/dummy_dg.yaml`

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## Demo Categories (Current Layout)

| Subdirectory | Executed semantics | Compatibility config |
|---|---|---|
| `00_smoke/` | Offline execution smoke with repository Dummy data | `dummy_dg.yaml` |
| `01_cross_domain/` | Held-out-domain ERM classification | `cwru_dg.yaml` |
| `02_cross_system/` | Known single-system CE with held-out domains | `multi_system_cddg.yaml` |
| `03_fewshot/` | Non-episodic supervised classification with held-out windows | `cwru_protonet.yaml` |
| `04_cross_system_fewshot/` | Hierarchically sampled CE classification | `gfs_dlinear.yaml` |
| `05_pretrain_fewshot/` | Single-stage HSE contrastive pretraining | `pretrain_hse_then_fewshot.yaml` |
| `06_pretrain_cddg/` | Single-stage HSE contrastive pretraining view | `pretrain_hse_cddg.yaml` |

Several filenames are retained temporarily for compatibility. The YAML content and its
README are authoritative about what is actually executed. A filename must not be used
as evidence that ProtoNet, generalized few-shot learning, multi-system generalization,
or a two-stage adaptation protocol is implemented.

## Demo Index

- `configs/demo/00_smoke/README.md`
- `configs/demo/01_cross_domain/README.md`
- `configs/demo/02_cross_system/README.md`
- `configs/demo/03_fewshot/README.md`
- `configs/demo/04_cross_system_fewshot/README.md`
- `configs/demo/05_pretrain_fewshot/README.md`
- `configs/demo/06_pretrain_cddg/README.md`

## Adding a New Demo

1. Put the YAML under the closest current category directory.
2. Follow the five-block model: `environment/data/model/task/trainer`.
3. Describe the real batch, objective and evaluation behavior.
4. Add one truthful entry to `configs/config_registry.csv`.
5. Validate and inspect the fully resolved config:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <your_yaml> --override trainer.num_epochs=1
python -m scripts.gen_config_atlas
python -m scripts.gen_support_matrix
```

6. Set `status=sanity_ok` only after the exact execution smoke passes. Set
   `protocol_status` independently; smoke execution does not establish scientific
   protocol validity.

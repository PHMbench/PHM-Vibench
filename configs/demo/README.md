# Demo Configs (`configs/demo/`)

Maintained, runnable example configurations. Use these as the template source for new
experiments by copying the nearest example into `configs/experiments/`.

This directory is intended to remain beginner-runnable:

- configs use repository-shipped dummy data or clearly document required external data;
- every maintained demo is indexed in `configs/config_registry.csv`;
- the registry generates `docs/CONFIG_ATLAS.md` and the support matrices;
- names must describe the resolved task/model rather than an intended or historical method.

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

| Subdirectory | Purpose | Example config |
|---|---|---|
| `00_smoke/` | Offline validation (repo-shipped dummy data) | `dummy_dg.yaml` |
| `01_cross_domain/` | Domain generalization (single-source) | `cwru_dg.yaml` |
| `02_cross_system/` | Cross-system/domain generalization | `multi_system_cddg.yaml` |
| `03_fewshot/` | Few-shot learning (FS) | `cwru_protonet.yaml` |
| `04_cross_system_fewshot/` | Generalized few-shot (GFS / cross-system) | `gfs_dlinear.yaml` |
| `05_pretrain_fewshot/` | Pretrain + few-shot pipeline | `pretrain_hse_then_fewshot.yaml` |
| `06_pretrain_cddg/` | Pretrain for CDDG pipeline | `pretrain_hse_cddg.yaml` |

## Naming Convention

Prefer names that expose the resolved task and material model distinction:

```text
{dataset-or-scope}_{task}_{model-or-variant}.yaml
```

Examples:

- `cwru_dg.yaml`
- `cwru_protonet.yaml`
- `multi_system_cddg.yaml`
- `gfs_dlinear.yaml`

A config must not claim TSPN, ProtoNet, or another method in its filename when its
resolved `model` block selects a different implementation.

## Demo Index

- `configs/demo/00_smoke/README.md`
- `configs/demo/01_cross_domain/README.md`
- `configs/demo/02_cross_system/README.md`
- `configs/demo/03_fewshot/README.md`
- `configs/demo/04_cross_system_fewshot/README.md`
- `configs/demo/05_pretrain_fewshot/README.md`
- `configs/demo/06_pretrain_cddg/README.md`

## Adding a New Demo

1. Put the YAML under the correct category directory.
2. Follow the five-block model: `environment/data/model/task/trainer`.
3. Add one truthful entry to `configs/config_registry.csv`.
4. Validate and inspect the fully resolved contract:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config <your_yaml> --override trainer.num_epochs=1
python -m scripts.gen_config_atlas
python -m scripts.gen_support_matrix
```

5. Add or update runtime evidence before setting `status=sanity_ok`.

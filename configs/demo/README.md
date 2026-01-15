# Demo Configs (`configs/demo/`)

Maintained, runnable example configurations. Use these as the template source for new experiments (copy into
`configs/experiments/`).

This directory is meant to stay “beginner runnable”:
- configs should run with repo-shipped dummy data, or clearly document required external data
- every maintained demo should be indexed in `configs/config_registry.csv` and appear in `docs/CONFIG_ATLAS.md`

For architecture/change guidance (acceptance criteria, how to evolve demos safely), see [@CLAUDE.md].

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
| `04_cross_system_fewshot/` | Generalized few-shot (GFS / cross-system) | `cross_system_tspn.yaml` |
| `05_pretrain_fewshot/` | Pretrain + few-shot pipeline | `pretrain_hse_then_fewshot.yaml` |
| `06_pretrain_cddg/` | Pretrain for CDDG pipeline | `pretrain_hse_cddg.yaml` |

## Naming Convention (Recommended)

Prefer names that make task + dataset obvious:

```
{dataset}_{task}_{variant}.yaml

Examples:
- cwru_dg.yaml
- cwru_protonet.yaml
- multi_system_cddg.yaml
```

## Demo Index

- `configs/demo/00_smoke/README.md`
- `configs/demo/01_cross_domain/README.md`
- `configs/demo/02_cross_system/README.md`
- `configs/demo/03_fewshot/README.md`
- `configs/demo/04_cross_system_fewshot/README.md`
- `configs/demo/05_pretrain_fewshot/README.md`
- `configs/demo/06_pretrain_cddg/README.md`

## Adding a New Demo (Checklist)

1. Put the YAML under the right category subdir.
2. Ensure it follows the 5-block model: `environment/data/model/task/trainer`.
3. Add an entry to `configs/config_registry.csv`.
4. Validate and inspect:
   - `python -m scripts.validate_configs`
   - `python -m scripts.config_inspect --config <your_yaml> --override trainer.num_epochs=1`

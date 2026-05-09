# Scripts

Utilities used by PHM‑Vibench tooling and maintenance.

Core commands (maintained):
- `python -m scripts.validate_configs`
- `python -m scripts.config_inspect --config <yaml> --override key=value`
- `python -m scripts.gen_config_atlas`
- `python -m scripts.validate_docs`
- `python -m scripts.paperpack_generative --run_dir <run_dir>`
- `python -m scripts.generative_sweep --config configs/demo/10_generative/dummy_generative_cfm.yaml`

UXFD merge utilities:
- `python -m scripts.collect_uxfd_runs --input save/ --out_dir docs/reports/` (collect `artifacts/manifest.json` into CSV)

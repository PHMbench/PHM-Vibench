# Scripts

Utilities used by PHM‑Vibench tooling and maintenance.

Core commands (maintained):
- `python -m scripts.validate_configs`
- `python -m scripts.config_inspect --config <yaml> --override key=value`
- `python -m scripts.gen_config_atlas`

UXFD merge utilities:
- `python -m scripts.collect_uxfd_runs --input save/ --out_dir reports/` (collect `artifacts/manifest.json` into CSV)


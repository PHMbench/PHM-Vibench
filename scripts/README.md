# Scripts

Utilities used by PHM‑Vibench tooling and maintenance.

Core commands (maintained):
- `python -m scripts.validate_configs`
- `python -m scripts.config_inspect --config <yaml> --override key=value`
- `python -m scripts.gen_config_atlas`
- `python -m scripts.validate_docs`

UXFD merge utilities:
- `python -m scripts.collect_uxfd_runs --input results --out_dir reports/` (collect `artifacts/manifest.json` into CSV)
- `python -m scripts.uxfd_postrun --config paper/LQ_vibench_fix/merge_uxfd/12_23/uxfd_postrun_config_example.yaml` (offline checks + plotting)

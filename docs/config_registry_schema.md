# `configs/config_registry.csv` Schema

This document defines the required columns and formatting rules for `configs/config_registry.csv`.

## 1) Purpose

`configs/config_registry.csv` is the single index for the configuration system:
- It lists every shipped base/demo config and how demo configs are composed via `base_configs`.
- It links configs to code owners (which pipeline/factory consumes them).
- It provides copy-paste runnable commands for users and reviewers.
- It is used to generate `docs/CONFIG_ATLAS.md` automatically.

## 2) CSV Format Rules

- Encoding: UTF-8
- Separator: `,`
- Quote any cell that contains commas.
- For list-like fields, use **semicolon-separated** items (e.g. `a;b;c`).
- Paths are repo-relative and must exist in this repository.

## 3) Base Columns (existing; do not change/remove)

| Column | Type | Meaning |
|---|---:|---|
| `id` | str | Unique identifier, stable over time (used by docs/README links). |
| `category` | str | One of: `base_environment`, `base_data`, `base_model`, `base_task`, `base_trainer`, `demo`. |
| `path` | path | YAML path of this config. |
| `description` | str | Human readable summary. |
| `pipeline` | str | Pipeline name (e.g. `Pipeline_01_default`); blank for base configs. |
| `base_environment` | path | For `demo` rows: base environment YAML path. |
| `base_data` | path | For `demo` rows: base data YAML path. |
| `base_model` | path | For `demo` rows: base model YAML path. |
| `base_task` | path | For `demo` rows: base task YAML path. |
| `base_trainer` | path | For `demo` rows: base trainer YAML path. |
| `status` | str | Validation status. Convention: `/` = unknown; `sanity_ok` = basic load verified. |

## 4) Extended Columns (appended; allowed to add more at the end)

| Column | Type | Meaning | Example |
|---|---:|---|---|
| `owner_code` | str | Primary code entry consuming this config (file:symbol). | `src/Pipeline_01_default.py:pipeline` |
| `keyspace` | str | Key namespaces mainly controlled by this config. Semicolon-separated. | `data.*;task.*` |
| `minimal_run` | str | Copy-paste minimal runnable command. | `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml` |
| `common_overrides` | str | 2–5 common overrides (semicolon-separated). | `trainer.num_epochs=1;data.num_workers=0` |
| `outputs` | str | Typical output layout template. | `{environment.output_dir}/{experiment_name}/iter_{i}/` |
| `related_docs` | str | Related docs paths (semicolon-separated). | `configs/README.md;docs/CONFIG_ATLAS.md` |

## 5) Notes

- The registry is intentionally descriptive: it does not replace the config loader or YAML schema.
- `scripts/gen_config_atlas.py` treats this CSV as the authoritative index.


# Root Directory Cleanup Audit

This note records why the former root-level directories were moved under `docs/`.

## Source History

| Root path | Source commit / status | Date | Notes |
| --- | --- | --- | --- |
| `projects/phm_generative/` | `7a37c5b` `feat: add generative CFM subsystem` | 2026-05-06 12:38 +0800 | Generative planning and research-frontier notes. |
| `examples/quickstart.py`, `examples/config_usage.py` | `d935983` `feat: Add legacy compatibility layer and configuration presets for PHM-Vibench` | 2025-08-28 21:29 -0400 | Legacy example scripts. Earlier removed examples were added by `b27b7b` on 2025-08-19. |
| `examples/README.md`, `metrics_reports/README.md`, `reports/README.md` | `1ebf40f` `docs: add README placeholders for tracked directories` | 2026-01-15 19:06 +0800 | Placeholder documentation for root directories. |
| `reports/uxfd_runs.csv` | `729acd0` `Merge remote-tracking branch 'origin/lqfix_25-12' into lq_merge_UXFD` | 2026-01-06 11:15 +0800 | Historical UXFD run summary. |
| `schemas/*.schema.json` | untracked local files, ignored by `.gitignore` `*.json` | mtimes 2026-05-04 to 2026-05-05 | Generative/domain-map JSON schemas. |

## New Locations

| Old root path | New docs path |
| --- | --- |
| `projects/phm_generative/` | `docs/phm_generative/` |
| `examples/` | `docs/past/examples/` |
| `reports/` | `docs/reports/` |
| `metrics_reports/README.md` | `docs/reports/metrics_reports.md` |
| `schemas/` | `docs/schemas/` |

## Follow-up Rule

Do not add new source-of-truth directories at the repository root for reports, schemas, examples, or project notes.
Use `docs/` for maintained documentation and `save/`, `results/`, or `environment.output_dir` for runtime outputs.

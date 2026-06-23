# Repository Index

This is the lightweight navigation map for PHM-Vibench. Use it before reading
large directory trees. It points to the stable entry documents and the files that
usually answer "where is this wired?" questions.

## Reading Order

1. Start with `AGENTS.md` for working rules and copy-paste validation commands.
2. Read `CLAUDE.md` for architecture intent, config contracts, and change order.
3. Use this file to choose the smallest relevant subsystem.
4. Read the subsystem README or `CLAUDE.md` named below.
5. Only then inspect concrete source, config, script, paper, or result files.

Avoid recursive full-repo reads. Prefer:

```bash
rg --files
rg "<symbol-or-keyword>" <focused-path>
find <focused-path> -maxdepth 2 -type f
```

## Core Entrypoints

| Area | Purpose | Read first | Key files |
|---|---|---|---|
| Root run path | Maintained CLI and onboarding | `README.md`, `AGENTS.md`, `CLAUDE.md` | `main.py`, `requirements.txt` |
| Configs | Experiment contracts | `configs/README.md`, `docs/CONFIG_ATLAS.md` | `configs/config_registry.csv`, `src/config_schema/` |
| Source code | Pipeline and factory wiring | `src/README.md`, relevant `src/*_factory/README.md` | `src/Pipeline_*.py`, `src/*_factory/` |
| Scripts | Validation and tooling | `scripts/README.md` | `scripts/validate_configs.py`, `scripts/config_inspect.py`, `scripts/gen_config_atlas.py` |
| Tests | Maintained checks | `test/README.md` | `test/` |
| Docs | User-facing docs, audits, and generated atlas | `docs/README.md` | `docs/bug_redundancy_optimization_report.md`, `docs/CONFIG_ATLAS.md`, `docs/config_registry_schema.md` |
| Paper | Research assets and submodules | `paper/README.md`, `paper/README_SUBMODULE.md` | `paper/UXFD_paper/README.md` |

## Source Map

- `src/configs/`: config loading, composition, local overrides, and path naming.
  Read `src/configs/CLAUDE.md` before changing loader behavior.
- `src/data_factory/`: readers, dataset tasks, samplers, and dataloader assembly.
  Read `src/data_factory/CLAUDE.md` and `src/data_factory/reader/README.md`
  before adding a dataset.
- `src/model_factory/`: registry-driven model construction. Read
  `src/model_factory/CLAUDE.md`; for UXFD model work, read
  `src/model_factory/X_model/README.md`.
- `src/task_factory/`: task registry, Lightning-style task modules, losses, and
  metrics. Read `src/task_factory/CLAUDE.md` and
  `src/task_factory/task_registry.csv`.
- `src/trainer_factory/`: trainer, callbacks, loggers, and device wiring. Read
  `src/trainer_factory/README.md`.
- `src/explain_factory/` and `src/plot_factory/`: explanation and plotting
  helpers. Read their local READMEs before extending.

## Config Map

- Maintained demos live under `configs/demo/`.
- Local research variants live under `configs/experiments/`.
- Legacy material lives under `configs/reference/`; do not template from it.
- The config registry is `configs/config_registry.csv`.
- The generated human-readable atlas is `docs/CONFIG_ATLAS.md`.

For config questions, use:

```bash
python -m scripts.config_inspect --config <yaml>
python -m scripts.validate_configs
```

## Data And Runtime Outputs

- `data/` contains small repo data plus raw/local datasets. Start with
  `data/README.md`.
- `data/raw/` and dataset submodules can be large. Do not read recursively unless
  the task explicitly needs raw data inspection.
- Runtime outputs go to `save/`, `results/`, or `environment.output_dir`. Start
  with `results/README.md`; inspect specific run directories only when needed.
- `pic/`, `plot/`, `metrics_reports/`, and `reports/` are artifact/document
  areas. Use their READMEs as entrypoints.

## Paper Map

- `paper/README.md`: top-level paper asset index.
- `paper/README_SUBMODULE.md`: parent-repo rules for paper submodules.
- `paper/2025-10_foundation_model_0_metric/`: metric/foundation-model paper
  submodule.
- `paper/LQ_vibench_fix/`: LQ fix and UXFD merge-history submodule.
- `paper/UXFD_paper/README.md`: UXFD suite index for the 7 UXFD paper
  submodules.
- `paper/UXFD_paper/thu_liqi_phd_thesis/`: local thesis workspace ignored by
  this repo; do not treat it as a tracked UXFD submodule.

Paper directories are often large and contain historical drafts, results, and
submodule-local agent artifacts. Read their index files first and inspect only
the named paper, config, script, or result needed for the task.

## Avoid By Default

Do not read or search these recursively unless the task requires them:

- `.venv/`, `.pytest_cache/`, `.cache/`, `__pycache__/`
- `data/raw/`, large dataset submodules, and local dataset dumps
- `save/`, `results/**`, `outputs/**`, checkpoint files, and generated figures
- `paper/**/results/`, `paper/**/outputs/`, `paper/**/.agent/`,
  `paper/**/.claude/`, `paper/**/.codex/`
- `configs/reference/` unless the task is explicitly about legacy migration

## Validation Anchors

Use the smallest relevant check first:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_configs
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
python -m pytest test/
```

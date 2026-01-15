# PHM-Vibench Project Context (GEMINI)

This file is lightweight context for AI assistants working on PHM-Vibench.

- Canonical onboarding + runnable configs: [@README.md] and [@configs/README.md]
- Copy-paste commands + validation gates: [@AGENTS.md]
- Change strategy / constraints: [@CLAUDE.md]

## Project Invariants (keep stable)

- Single maintained entrypoint: `python main.py --config <yaml> [--override key=value ...]`
- 5-block config model: `environment` / `data` / `model` / `task` / `trainer`
- Factory + registry wiring (avoid hard-coded imports in pipelines):
  - `src/data_factory/`, `src/model_factory/`, `src/task_factory/`, `src/trainer_factory/`

## “Single Source of Truth” (SSOT)

- Config index: `configs/config_registry.csv` → rendered to `docs/CONFIG_ATLAS.md`
- Task registry: `src/task_factory/task_registry.csv`
- Model registry: `src/model_factory/model_registry.csv`

## Where to Look First

- Config usage + examples: `configs/README.md` and `configs/demo/README.md`
- Factory module docs (canonical per module): `src/*_factory/README.md`
- Config resolution debugging: `python -m scripts.config_inspect --config <yaml> --dump targets`

## Common Assistant Pitfalls

- Don’t assume values in YAML are final; always check `config_inspect` for resolved sources/targets.
- Avoid breaking the public CLI or the 5-block keyspace without a compatibility layer + migration note.
- Keep paper/submodule workflows out of core validation gates.

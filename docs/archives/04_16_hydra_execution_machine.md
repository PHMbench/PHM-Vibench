# PHM-Vibench Hydra Execution Machine Archive

Date: 2026-04-16

## Scope

This archive records the spec-driven implementation that turns PHM-Vibench into a
config-first scientific execution machine:

```text
config -> preflight -> pipeline -> trainer -> artifacts
```

## Source Artifacts

- Spec root: `docs/ignore/04_16/hydra_execution_machine/`
- Master progress: `docs/ignore/04_16/hydra_execution_machine/progress/MASTER.md`
- Analysis:
  - `docs/ignore/04_16/hydra_execution_machine/analysis/project-overview.md`
  - `docs/ignore/04_16/hydra_execution_machine/analysis/module-inventory.md`
  - `docs/ignore/04_16/hydra_execution_machine/analysis/risk-assessment.md`
- Plan:
  - `docs/ignore/04_16/hydra_execution_machine/plan/task-breakdown.md`
  - `docs/ignore/04_16/hydra_execution_machine/plan/dependency-graph.md`
  - `docs/ignore/04_16/hydra_execution_machine/plan/milestones.md`
- Project skill: `.codex/skills/phm-vibench-hydra-execution-machine/SKILL.md`

## Completion Status

- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete
- Phase 5: complete
- Phase 6: complete

## Delivered Gates

- Strict entrypoint: `python main.py --config <yaml> [--override key=value ...]`
- Preflight before pipeline/trainer construction.
- Hydra-compatible config tree and validation coverage.
- Explicit P02 `pipeline_mode`.
- Fail-fast P03/P04 maintained paths.
- Explicit nonzero unlabeled InfoNCE pairing semantics.
- Required run manifest and metrics artifact contract.
- Demo matrix smoke/full runner.
- Core CI workflow.

## Validation Commands

```bash
python -m py_compile main.py scripts/config_inspect.py src/configs/preflight.py src/utils/training/run_contract.py
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/hydra/experiments/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m pytest -q test/test_hydra_config_matrix.py test/test_pipeline_02_modes.py test/test_hse_contrastive_failfast.py test/test_infonce_pairing.py test/test_config_env_expansion.py test/test_preflight.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py test/test_pipeline_failfast.py test/test_demo_matrix_script.py test/test_main_strictness.py
bash scripts/run_demo_matrix.sh --mode smoke
```

## Notes

- Full demo matrix requires `PHM_VIBENCH_DATA` for real-data demos.
- The original date-scoped spec files remain in place for detailed traceability.

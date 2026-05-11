# GOAL-FFU-P0-002: Strict Main Preflight

## Objective

Add strict `python main.py --config <yaml> --preflight-only` support and remove
silent YAML parse fallback.

## Required Behavior

- Parse file configs strictly.
- Resolve presets only through the existing config loader.
- Validate pipeline against `ALLOWED_PIPELINES`.
- Validate the 5-block config contract.
- Run schema and generative mode checks.
- Exit before trainer or pipeline execution work starts.

## Acceptance Criteria

- Dummy default and dummy generative configs pass preflight.
- Malformed YAML, invalid pipeline, missing file, and missing required block fail.
- No `trainer.fit` or `trainer.test` executes in preflight mode.

## Validation Commands

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m pytest test/smoke/test_preflight.py -q
```

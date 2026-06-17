/goal

## Goal ID
GEN-V3-005-PIPELINE-TO-NCL-STRICT

## Objective
Make the pipeline-level `_to_ncl` shape conversion fail fast when channel axis cannot be inferred.

## Why
Task-level `_to_ncl` is strict, but pipeline-level `_to_ncl` can return an ambiguous tensor unchanged. This risks silent metric or normalization errors.

## Scope
Allowed:
- src/Pipeline_06_generative.py
- tests/generative/test_pipeline_shape_contract.py

## Required behavior
1. `_to_ncl(x, channels)` accepts [N,C,L] or [N,L,C].
2. It raises ValueError if neither axis matches expected channels.
3. Error message includes shape and expected channels.
4. Existing dummy generative smoke still passes.

## Validation commands
python -m pytest tests/generative/test_pipeline_shape_contract.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

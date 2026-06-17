/goal

## Goal ID
GEN-V3-004-STRICT-CONDITION-SPLIT

## Objective
Make `condition_sampling_policy=train_distribution` benchmark-safe by requiring explicit train split evidence.

## Why
If metadata has no split field, train_distribution can sample all metadata rows. That must stay exploratory and be recorded.

## Scope
Allowed:
- src/Pipeline_06_generative.py
- src/task_factory/Components/generative/manifests/synthetic_data_manifest.py
- tests/generative/test_condition_sampling_policy.py

## Required behavior
1. Detect whether metadata rows had explicit split fields.
2. If train_distribution uses rows without split evidence, record `condition_sampling_split_verified=false`.
3. Manifest benchmark evidence must require split verification for train_distribution.
4. Smoke demos remain exploratory.

## Validation commands
python -m pytest tests/generative/test_condition_sampling_policy.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

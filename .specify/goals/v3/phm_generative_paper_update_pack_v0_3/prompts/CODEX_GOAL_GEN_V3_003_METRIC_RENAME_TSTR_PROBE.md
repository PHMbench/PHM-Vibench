/goal

## Goal ID
GEN-V3-003-METRIC-RENAME-TSTR-PROBE

## Objective
Rename nearest-centroid TSTR/TRTS metrics to make clear they are probe metrics, not full downstream classifier training.

## Why
The current nearest-centroid metric is useful, but paper claims must not confuse it with a full TSTR classifier protocol.

## Scope
Allowed:
- src/task_factory/Components/generative/metrics/tstr.py
- src/task_factory/task/generative/generative_eval.py
- scripts/paperpack_generative.py
- scripts/generative_benchmark_effect.py
- tests/generative/test_tstr_metrics.py

## Required behavior
1. Emit `tstr_nearest_centroid_accuracy`.
2. Emit `trts_nearest_centroid_accuracy`.
3. Preserve backward-compatible aliases only if clearly marked deprecated.
4. Paperpack utility prefixes still capture these metrics.
5. Missing labels get status/reason.

## Validation commands
python -m pytest tests/generative/test_tstr_metrics.py
python -m pytest tests/scripts/test_paperpack_generative.py

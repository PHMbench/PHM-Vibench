/goal

## Goal ID
GEN-V3-002-EVAL-EVIDENCE-MANIFEST

## Objective
Write `eval_evidence_manifest.json` next to `generative_eval_metrics.csv`.

## Why
The sample-time synthetic manifest cannot know whether metrics have status/reason fields. Eval must produce a sidecar that records metric completeness and promotion eligibility.

## Scope
Allowed:
- src/Pipeline_06_generative.py
- src/task_factory/task/generative/generative_eval.py
- src/task_factory/Components/generative/manifests/
- test/generative/test_eval_evidence_manifest.py

Out of scope:
- Do not change model outputs.
- Do not introduce full downstream classifier TSTR.

## Required behavior
1. Eval writes `eval_evidence_manifest.json`.
2. It records generated_path, synthetic_manifest_path if resolvable, metrics_path, reference split, and allow_test_reference_eval.
3. It counts metric statuses: ok, not_computable.
4. It records promotion.eligible=false unless all required evidence exists.
5. It never mutates sample manifest in place.

## Validation commands
python -m pytest test/generative/test_eval_evidence_manifest.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

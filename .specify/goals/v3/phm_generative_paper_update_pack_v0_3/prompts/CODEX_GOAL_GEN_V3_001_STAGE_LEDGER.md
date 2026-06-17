/goal

## Goal ID
GEN-V3-001-STAGE-LEDGER

## Objective
Add a stage ledger for PHM-GenBench train/sample/eval/paperpack artifacts so paperpack can link metrics to the correct synthetic manifest.

## Why
Current paperpack scans an eval run directory for manifests. In paper workflows, sample manifests can live in a sibling sample stage directory. A stage ledger prevents missing manifest evidence and makes paper rows auditable.

## Scope
Allowed:
- scripts/generative_benchmark_effect.py
- scripts/paperpack_generative.py
- src/Pipeline_06_generative.py
- test/scripts/test_generative_stage_ledger.py
- test/scripts/test_paperpack_stage_ledger.py

Out of scope:
- Do not add new models.
- Do not change loss math.
- Do not mark any row benchmark-valid by default.

## Required behavior
1. Train/sample/eval stages can write or update a common `stage_ledger.json`.
2. The ledger records checkpoint path, samples path, synthetic manifest path, metrics path, and paperpack dir when available.
3. `paperpack_generative.py` accepts `--stage_ledger`.
4. Paperpack uses the ledger to locate synthetic manifests outside the eval dir.
5. Missing ledger does not break old behavior, but produces a warning in appendix/run_index.csv.

## Acceptance criteria
- A synthetic manifest in a sample sibling dir is included in manifest_completeness.csv.
- Missing ledger is reported, not silently ignored.
- Existing `--run_dir` behavior still works.

## Validation commands
python -m pytest test/scripts/test_generative_stage_ledger.py test/scripts/test_paperpack_stage_ledger.py
python -m scripts.validate_docs

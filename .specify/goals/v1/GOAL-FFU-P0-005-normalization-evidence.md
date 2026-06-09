# GOAL-FFU-P0-005: Normalization Evidence

## Objective

Write and attach per-channel normalization evidence for generative runs.

## Required Behavior

- Write `normalization_params.json`.
- Write `normalization_params.sha256`.
- Support `standardization` and `robust_scaler`.
- Compute stats from train/source data only.
- Attach path/hash to synthetic manifest.

## Acceptance Criteria

- Benchmark-valid requires params artifact and hash.
- No Min-Max default is introduced.
- Missing params evidence is visible in manifest validity.

## Validation Commands

```bash
python -m pytest test/generative/test_normalization_manifest.py -q
```

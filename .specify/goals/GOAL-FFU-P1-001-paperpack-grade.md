# GOAL-FFU-P1-001: Paper-Grade Paperpack

## Objective

Upgrade `scripts/paperpack_generative.py` for multi-seed PHM paper artifacts.

## Required Behavior

Output:

- mean/std quality, utility, and efficiency tables
- leakage table
- ablation table
- run index
- manifest completeness table
- missing metric appendix
- spectra, temporal, and barplot figure-source CSVs

## Acceptance Criteria

- Single-run and multi-run inputs both work.
- Every aggregate row preserves source paths.
- Missing metrics are listed with reasons when available.

## Validation Commands

```bash
python -m pytest test/generative/test_paperpack_generative.py -q
python -m compileall scripts/paperpack_generative.py
```

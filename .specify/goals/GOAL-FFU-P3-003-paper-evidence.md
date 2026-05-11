# GOAL-FFU-P3-003: Final Paper Evidence

## Objective

Produce the final paper artifact structure for PHM-GenBench.

## Required Behavior

- Quality, utility, efficiency, leakage, and ablation tables.
- Reproducibility statement.
- Figure-source CSVs for temporal and spectral overlays.
- Appendix for run index, manifest completeness, and missing metrics.

## Acceptance Criteria

- All paper artifacts are reproducible from run directories.
- No table row lacks a source path.
- Missing metrics are explained.

## Validation Commands

```bash
python -m pytest test/generative/test_paperpack_generative.py -q
python -m scripts.validate_docs
```

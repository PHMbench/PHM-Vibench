# P08 HSE-Prompt CDDG project overlay

This branch uses the real Metric parent repository as the historical CDDG
implementation source. The `CDDGpaper/` LaTeX subtree is only a shared
narrative seed; it is not a complete implementation, run, or result record.

## Active Vibench entry points

- Paper repository: `AI4Engineering-L/P08-HSE-Prompt-CDDG`
- Maintained bridge configs: `configs/experiments/p08/`
- Commit-pinned Metric source: `paper/project/metric_source/`
- Provenance and exclusions: `paper/project/SOURCE_MAP.yaml`

The Metric files are preserved for traceability and porting. They are not
accepted empirical evidence. Positive claims still require target-excluded
runs, frozen configs, seeds, and hash-bound artifacts in the PaperTrace paper
repository.

## Validation

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m scripts.config_inspect \
  --config configs/experiments/p08/p08_exp0_backbone_baseline.yaml
```

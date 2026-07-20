# P09 HSE-Prompt GFS project overlay

This branch uses the real Metric parent repository as the historical GFS
implementation source. The `Few-shotpaper/` subtree is byte-identical to the
P08 CDDG LaTeX seed and is not an independent GFS implementation or result.

## Active Vibench entry points

- Paper repository: `AI4Engineering-L/P09-HSE-Prompt-GFS`
- Maintained bridge configs: `configs/experiments/p09/`
- Commit-pinned Metric source: `paper/project/metric_source/`
- Provenance and exclusions: `paper/project/SOURCE_MAP.yaml`

The Metric files are preserved for traceability and porting. They do not
promote any claim. Independent target-excluded GFS runs must report base,
novel, and harmonic-mean metrics with seeds and hash-bound artifacts in the
PaperTrace paper repository.

## Validation

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m scripts.config_inspect \
  --config configs/experiments/p09/p09_exp0_backbone_baseline.yaml
```

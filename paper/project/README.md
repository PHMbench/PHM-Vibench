# P07 XOAN project overlay

This directory preserves the paper-specific XOAN material used to prepare the
`paper/p07-xoan-operator-attention` Vibench branch. It is an isolated research
overlay: files here are not imported by the PHM-Vibench runtime automatically.

## Active Vibench entry points

- Paper repository: `AI4Engineering-L/P07-XOAN-Operator-Attention`
- Maintained bridge configs: `configs/experiments/p07_xoan_operator_attention/`
- Filtered legacy source: `paper/project/legacy_source/`
- Provenance and exclusions: `paper/project/SOURCE_MAP.yaml`

The legacy source remains unverified historical material. Its reports,
manuscript numbers, figures, and code do not establish a positive paper claim
unless a run is reproduced in Vibench and bound to hashed artifacts in the
paper repository.

Large or generated material was deliberately left behind: results, outputs,
checkpoints, model weights, caches, PDFs, ZIP archives, and agent prompt
directories. The original PaperTrace archive branch remains the recovery path
for those files.

## Validation

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m scripts.config_inspect \
  --config configs/experiments/p07_xoan_operator_attention/p0_synthetic_operator_attention_smoke.yaml
```

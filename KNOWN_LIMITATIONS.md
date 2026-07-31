# Known Limitations for v0.2.0

## Scope

- v0.2.0 supports the maintained demo combinations listed in
  `SUPPORTED_COMBINATIONS.md`, not every discovered model/task pair.
- `Pipeline_02_Pretraining_Few_Shot` is validated for the current single-stage
  pretrain demo. Multi-stage `stages:` workflows need separate evidence.
- `Pipeline_03` remains outside the public release-supported surface.

## Data And Environment

- Only the dummy smoke demo is offline and repo-shipped.
- Non-dummy demos require local PHM-Vibench metadata/raw data and may need a
  machine-specific `data.data_dir` override.
- Maintained validation uses the `LQ_signal` conda environment. Base Python may
  lack required packages such as `pytorch_lightning`.

## Evidence Boundaries

- Smoke evidence is functional evidence, not performance evidence.
- Registry rows are traceability evidence; runtime support requires a passing
  maintained demo or an explicit additional smoke matrix.
- Broad `python -m pytest -q` is diagnostic because it collects historical and
  stray tests. The maintained gate is `python -m pytest test/ -q`.

## Compatibility Boundaries

- Sampler compatibility must be derived from current `Get_sampler.py`; stale
  compatibility matrices are not release evidence.
- Dataset adapter fallback to `Default_dataset` remains a behavior to inspect
  carefully when adding new task names.
- Unknown pipeline, model, and task values are expected to fail explicitly.


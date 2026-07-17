# Known Limitations for v0.2.1

## Scope

- v0.2.1 supports the maintained demo combinations listed in
  `SUPPORTED_COMBINATIONS.md`, not every discovered model/task pair.
- `Pipeline_02_pretrain_fewshot` is validated for the current single-stage
  pretrain demo. Multi-stage `stages:` workflows need separate evidence.
- `Pipeline_03` remains outside the public release-supported surface.

## Data And Environment

- Only the maintained dummy smoke demos are offline and repo-shipped.
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

## Pipeline 06 CFM Boundary

- Conditional Flow Matching support is limited to the exact maintained
  `configs/demo/10_generative/dummy_generative_cfm.yaml` combination.
- The evidence is a one-epoch seed-0 functional smoke on CPU and one NVIDIA
  GeForce RTX 4090. It is not benchmark-valid, paper-ready, or a performance
  comparison.
- The smoke generates one condition, so `downstream_classifier_utility` is
  explicitly `not_computable` with a recorded reason. The other required metric
  records must remain finite and non-failed.
- Direct model conditions are limited to `fault_label` and `domain_id`; the
  maintained sampler is Euler ODE.
- Support does not extend to arbitrary datasets, backbones, samplers, GPUs,
  multi-GPU execution, paper configurations, or parameter combinations.
- Train, sample, and eval remain separate public invocations. Sample and eval
  require the exact checkpoint, normalization, protocol, sample, and manifest
  paths and hashes produced by the preceding stages.

## Compatibility Boundaries

- Sampler compatibility must be derived from current `Get_sampler.py`; stale
  compatibility matrices are not release evidence.
- Dataset adapter fallback to `Default_dataset` remains a behavior to inspect
  carefully when adding new task names.
- Unknown pipeline, model, and task values are expected to fail explicitly.

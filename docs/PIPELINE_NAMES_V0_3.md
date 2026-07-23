# PHMFactory v0.3 Pipeline Names

PHMFactory v0.3 keeps the established Pipeline numbers while replacing terse or
implementation-timeline names with descriptive task names.

## Canonical identifiers

| Legacy identifier | Canonical v0.3 identifier |
| --- | --- |
| `Pipeline_01_default` | `Pipeline_01_Fault_Diagnosis` |
| `Pipeline_02_pretrain_fewshot` | `Pipeline_02_Pretraining_Few_Shot` |
| `Pipeline_03_multitask_pretrain_finetune` | `Pipeline_03_Multitask_Pretraining_Finetuning` |
| `Pipeline_04_unified_metric` | `Pipeline_04_Unified_Evaluation` |
| `Pipeline_05_default_w_explain` | `Pipeline_05_Explainable_Fault_Diagnosis` |
| `Pipeline_06_generative` | `Pipeline_06_Generative_Modeling` |

The Python modules under `src/` use the canonical identifiers directly. The old
module filenames are not retained as wrapper modules.

## Configuration compatibility

Maintained v0.3 configurations use canonical identifiers:

```yaml
pipeline: Pipeline_01_Fault_Diagnosis
```

Legacy configuration values remain accepted through
`phmfactory.pipelines.PIPELINE_ALIASES` and emit a visible deprecation warning.
This compatibility applies to configuration identifiers, not direct Python imports
of the removed legacy module filenames.

## Direct-import migration

```python
# Before
from src.Pipeline_01_default import pipeline

# v0.3
from src.Pipeline_01_Fault_Diagnosis import pipeline
```

Downstream projects should prefer the public CLI rather than importing Pipeline
modules directly:

```bash
phmfactory --config path/to/config.yaml
```

## Scope boundary

The rename does not change Pipeline algorithms, seeds, split behavior, metrics,
checkpoint formats, factory construction, signal shapes, or reader behavior. Each
renamed module must retain the exact pre-rename file SHA-256 recorded by the v0.3
protected-runtime baseline.

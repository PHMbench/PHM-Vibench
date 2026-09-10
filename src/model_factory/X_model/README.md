# Explainability and auxiliary models

This folder holds reusable signal-processing, explanation and auxiliary implementations.
Not every helper is a top-level Factory model. Select an exact model module, for example:

```yaml
model:
  type: X_model
  name: TSPN_UXFD
```

This is a fragment; use the corresponding complete configuration for operator and shape
settings. Catalogue entries include `MWA_CNN`, `TSPN`, `TSPN_UXFD`, `XOANOperatorPath` and
`BASE_ExplainableCNN`. Consult [model_registry.csv](../model_registry.csv) and the selected
source, rather than assuming `Feature_extract.py` or every helper exposes `Model`.

## Boundaries

Model assembly follows the parent [Factory contract](../README.md). Keep reusable code
here; paper-specific methods, figures and results belong to their research repository.
Use [paper/project/README.md](../../../paper/project/README.md) for migrated source
locations, not the removed historical paper-submodule paths.

The [LLM explanation integration guide](../../../docs/LLM_EXPLANATION_INTEGRATION.md)
describes adapters for model traces. A valid trace or citation reference does not by
itself prove causal, physical or natural-language faithfulness. Preserve the actual
forward branch, intervention settings and active masks when reporting explanations.

For a change, use the relevant assembly/trace tests and exact configuration. Do not infer
universal model support, numerical performance or a paper claim from helper imports.

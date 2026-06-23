# Contract: Model, Loss, And Baseline Registry

## Sources Of Truth

- Model entries come from `src/model_factory/model_registry.csv`.
- ISFM component entries come from `src/model_factory/ISFM/isfm_components.csv`.
- Loss, metric, contrastive strategy, and regularization keys come from
  `src/task_factory/Components/README.md` and the corresponding factories.
- Baseline mapping is derived from model registry entries, Slice 2 task/data
  compatibility, maintained configs, and run evidence.

Do not maintain a second frozen inventory in prose.

## Support Status Contract

Every model, ISFM component, and selected baseline must resolve to exactly one
status:

- `smoke-tested`
- `dependency-blocked`
- `unverified`
- `unsupported`
- `failed`

Blocked, unverified, and failed entries must include a reason. They must not be
counted as completed comparisons.

## Model Smoke Contract

Focused model validation must identify these failure classes by registry row:

- missing module path;
- missing factory entry point;
- missing optional dependency;
- constructor argument mismatch;
- forward-output shape or type mismatch.

Passing smoke evidence does not imply full real-data benchmark evidence.

## ISFM Component Contract

Component validation must check:

- component ids referenced by maintained ISFM configs exist in the component
  registry;
- component module paths resolve for smoke-tested entries;
- required key args are documented or recorded as a gap.

## Loss And Contrastive Contract

Loss, metric, and contrastive strategy validation must check:

- documented keys are accepted by the relevant factory;
- unknown keys fail explicitly;
- contrastive losses that require positive pairs or two views fail explicitly when
  batches cannot satisfy those contracts;
- no invalid pairing may silently return zero loss or fall back to another loss.

## Baseline Mapping Contract

For each selected PHM task family, baseline evidence must record:

- baseline role: mandatory, optional, blocked, or unverified;
- registered model reference;
- compatible task/data entry from Slice 2;
- config and command when runnable;
- evidence result or blocker reason.

Paper-facing baseline claims are not complete until they map to run artifacts or a
documented unresolved blocker.

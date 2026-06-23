# Research: Model, Loss, And Baseline Registry

## Decision: Use model registry rows as the model support source

**Rationale**: `src/model_factory/model_registry.csv` is the documented model
factory source of truth and includes type, name, module path, arguments, notes, and
test status. It is the only stable way to distinguish public factory entries from
archived or utility code.

**Alternatives considered**:

- Scan model directories. Rejected because archived, utility, and experimental code
  may not be registered or factory-ready.
- Maintain a hand-written model list in the spec. Rejected because it duplicates the
  registry and will drift.

## Decision: Treat ISFM components as first-class registry entries

**Rationale**: ISFM models compose embeddings, backbones, and task heads. The
component registry captures this surface more precisely than model-level rows alone.

**Alternatives considered**:

- Validate only complete ISFM models. Rejected because component wiring bugs can be
  hidden until a specific config selects that component.

## Decision: Support statuses include dependency-blocked and failed

**Rationale**: Some registered entries require optional dependencies. Marking them
as skipped or passing would overstate benchmark support; marking them
dependency-blocked preserves traceability without forcing new dependencies into this
slice.

**Alternatives considered**:

- Install every optional dependency immediately. Rejected because the goal requires
  minimal changes and explicit dependency justification.
- Treat unavailable dependencies as unsupported. Rejected because the registry entry
  may be valid but blocked in the current environment.

## Decision: Model smoke tests should precede full training

**Rationale**: Import, constructor, dependency, and output-shape failures are cheaper
and more precise than failed full experiments. Full training evidence is still
needed for final baseline comparisons, but it should build on focused smoke health.

**Alternatives considered**:

- Rely only on demo/full matrix runs. Rejected because failures would be slower and
  harder to attribute to a specific model row.

## Decision: Loss and contrastive support is keyed by factories and documented contracts

**Rationale**: The task component README and loss/strategy factories define the
supported keys and pairing semantics. Tests already cover some failure modes such as
odd two-view batches and missing positive pairs; this slice should extend coverage
only where the contract is uncovered.

**Alternatives considered**:

- Allow fallback to CE or zero loss for invalid pairings. Rejected because it
  invalidates pretraining evidence.

## Decision: Baseline mapping is derived, not frozen in this spec

**Rationale**: Mandatory and optional baselines depend on registered models, Slice 2
task/data compatibility, and maintained configs. A frozen prose list in the spec
would drift as registries change.

**Alternatives considered**:

- Hard-code a paper baseline list now. Rejected because Slice 4 owns final paper
  claim alignment and this slice owns generic support contracts.

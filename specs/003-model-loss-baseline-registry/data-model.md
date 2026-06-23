# Data Model: Model, Loss, And Baseline Registry

## Model Registry Entry

Represents one public model factory entry.

**Fields**:

- `model_type`: registry model type.
- `model_name`: registry model name.
- `module_path`: import path to the model module.
- `args`: key constructor/config arguments.
- `notes`: intended usage or dependency notes.
- `test_status`: recorded status field from the registry.
- `support_status`: one Support Status value.

**Validation rules**:

- `(model_type, model_name)` must be unique.
- `module_path` must resolve for entries marked smoke-tested.
- A smoke-tested entry must expose the expected factory entry point and output shape
  for its selected task.

## ISFM Component Entry

Represents one registered ISFM embedding, backbone, or task head component.

**Fields**:

- `component_type`: embedding, backbone, task_head, or related category.
- `component_id`: registry component id.
- `module_path`: import path to the component module.
- `key_args`: required or important configuration keys.
- `notes`: intended usage.
- `support_status`: one Support Status value.

**Validation rules**:

- `(component_type, component_id)` must be unique.
- Component ids referenced by maintained ISFM configs must exist in the component
  registry.
- Missing key args or missing modules must be recorded as failed or unverified.

## Component Contract

Represents validation expectations for a model, loss, metric, regularizer, or
contrastive strategy.

**Fields**:

- `kind`: model, isfm_component, loss, metric, regularizer, or contrastive_strategy.
- `key`: registry id or factory key.
- `required_inputs`: constructor args, batch keys, labels, positive pairs, or
  two-view shape requirements.
- `expected_output`: tensor shape, scalar loss, metric mapping, or status result.
- `dependency`: optional dependency name when relevant.

**Validation rules**:

- Unknown keys fail explicitly.
- Impossible pairings fail explicitly with a reason.
- Optional dependencies must be recorded when absent.

## Support Status

Auditable support state for registry entries and baselines.

**Values**:

- `smoke-tested`: focused import/constructor/forward or component test passes.
- `dependency-blocked`: optional dependency is required and unavailable.
- `unverified`: source-of-truth entry exists but passing evidence is missing.
- `unsupported`: entry is absent from source-of-truth registries or intentionally
  out of scope.
- `failed`: validation ran and failed with a recorded reason.

**Validation rules**:

- Every source-of-truth entry has exactly one status.
- A blocked or failed entry cannot be counted as a completed baseline.

## Baseline Mapping

Represents a comparison baseline selected for a PHM task family.

**Fields**:

- `task_family`: Slice 2 task family.
- `baseline_role`: mandatory, optional, blocked, or unverified.
- `model_ref`: `(model_type, model_name)` registry reference.
- `config_path`: runnable config when available.
- `command`: smoke or full command when available.
- `evidence`: Validation Evidence.
- `blocker_reason`: required when role is blocked or unverified.

**Validation rules**:

- `model_ref` must exist in the model registry.
- `task_family` must be compatible according to Slice 2 evidence.
- Baseline claims require command/run evidence or a blocker.

## Validation Evidence

Represents checked support evidence.

**Fields**:

- `command`: exact command or test.
- `result`: pass, fail, blocked, or skipped.
- `reason`: required for fail, blocked, and skipped.
- `artifact_paths`: run artifacts or logs when produced.
- `timestamp`: time evidence was recorded.

**Validation rules**:

- Do not mark support as smoke-tested without a command/test result.
- Skipped full evidence must state which prerequisite is missing.

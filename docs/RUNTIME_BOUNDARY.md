# Runtime and research ownership

PHMFactory owns reusable models, operators, losses, readers, samplers, metrics and runtime interfaces. A downstream research repository owns its study, task binding, model-selection budget, comparison arms, orchestration, result directories, scientific tables, figures and manuscript. Existing model names may retain their published origin; a name is not a license to embed that publication's study inside the runtime.

A configurable algorithm with explicit inputs and outputs is not the same as an experiment controller. Move an entire paper-specific controller to its owner; extract a general primitive only when its reusable interface is already justified. Do not keep a forwarding stub or a fallback import to the old owner.

## Explicit downstream modules

The existing `phmfactory research` interface accepts an optional `task.module` containing the fully qualified module selected by the caller. The module must expose `execute(config, phase, output, ...)`. An explicit module must be importable in the caller's environment; an invalid or missing module fails without substitution. The default built-in path remains unchanged for reusable installed tasks.

`task.module` is configuration-selected code, not untrusted data: use only a trusted module and configuration. PHMFactory does not enumerate downstream repositories or know their experiment IDs, result layouts, tuning policies or target-release rules. Ordinary training rejects this execution form instead of treating it as a Lightning task.

Downstream programs continue to use the existing public configuration resolver and shared model/runtime interfaces. No second configuration system, trainer or evaluator is introduced by module selection.

Pipelines 01 and 06 also accept explicit Factory extensions. The following module names are
illustrative: the caller must provide importable modules in its environment.

```yaml
task:
  factory_module: example_extension.factory
model:
  module: example_extension.model
```

The Task factory module is imported before data construction, so it may use the
existing `register_task` and `register_dataset_adapter` interfaces. The configured
`task.type` and `task.name` still select the Task and dataset. The model module exports
`Model(args_model, metadata)`. Explicit imports fail if unavailable; omitting either
field retains the ordinary built-in resolution. These fields select trusted executable
code and are distinct from the scientific `task.module` entry above.

For Pipeline 06, the same module import runs before Data Factory construction in
each selected train, sample or eval invocation. A missing module or an exception
inside its registration code stops that invocation; it does not select a built-in
Task. See the [generative interface](../configs/demo/10_generative/README.md#factory-interface)
for the model and Task signatures and the current evaluation limits.

## Migration acceptance

Validate the receiving repository and its call paths first. Preserve the scientific configuration and comparison semantics. Compare the resulting commands, fixed-model outputs and relevant tests before removing the source implementation. Keep unassigned or unvalidated research artifacts intact and report them explicitly. Historical source references are provenance, not additional active entrypoints.

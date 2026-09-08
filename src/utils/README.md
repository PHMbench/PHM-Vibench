# Runtime utilities

This directory contains helpers used by the existing runtime and research components.
It is not an alternative public configuration, training, or evaluation entrypoint.

## Start with the owning interface

| Task | Interface or guide |
| --- | --- |
| Resolve a public experiment | `phmfactory.config.analyze_config()`; [configuration guide](../../configs/README.md) |
| Construct data | [Data Factory](../data_factory/README.md) |
| Construct a model | [Model Factory](../model_factory/README.md) |
| Define an objective and metrics | [Task Factory](../task_factory/README.md) |
| Select devices and callbacks | [Trainer Factory](../trainer_factory/README.md) |
| Inspect repeated-run output | [run_summary.py](run_summary.py) and the CLI's returned paths |
| Inspect checkpoint and logging helpers | [utils.py](utils.py) |

The historical loader is documented in [src/configs/README.md](../configs/README.md).
Do not use it to add another merge or validation path to a public command. Do not copy
old encoding fallback, automatic device selection, or result-directory scanning recipes
into new code.

## Existing registry interface

[registry.py](registry.py) exposes `Registry()`, `register(name)`, `get(name)`, and
`available()`. It does not expose `Registry("model")`, `register_module()`, or `build()`.
Use an existing Factory's registration interface rather than creating another registry.
At this source state repeated registration replaces a key; this is a known implementation
behavior, not a recommendation or a guarantee that duplicate-key protection is complete.

## Research helpers

A utility's presence does not make it a maintained workflow. Before using a historical
HSE, multi-stage, evaluation, or validation helper, inspect its implementation and actual
callers. Use the current [supported combinations](../../SUPPORTED_COMBINATIONS.md) to
check an experiment's scope, not the former utility version labels or feature lists.

## Changes and verification

Keep helpers local to the responsibility they serve. Preserve the requested experiment
and original failure; do not add retries, substitute data, silently choose a different
loss/device, or introduce another configuration or result manager.

For a documentation-only edit, run `python -m scripts.validate_docs`. For a helper change,
run its existing focused tests and the affected public path as described in
[testing](../../docs/testing.md). Do not download a dataset merely to validate this guide.

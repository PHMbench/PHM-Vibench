# Contributing Models

Use this page for model-factory-specific work. General issue, branch, evidence,
license, documentation, and pull-request rules are in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Runtime contract

The model factory resolves configuration as:

```text
model.type = <Family>
model.name = <ModelName>
→ src.model_factory.<Family>.<ModelName>
→ module.Model(args_model, metadata)
```

A model module must expose a `Model` class. Keep its public behavior explicit:

```python
class Model(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()

    def forward(self, x, data_id=None, task_id=None):
        ...
```

The exact `forward` arguments and output depend on the task, but the contribution
must document accepted input shape, output shape or keys, dtype, device, optional
metadata identifiers, and failure behavior.

Do not add a model-specific conditional to `main.py`. If a model family needs
reusable components, keep them under its family directory and expose them through
the family configuration.

## Choose the scope

Before implementation, state whether the contribution is:

- a new model family;
- a model within an existing family;
- a reusable embedding, backbone, head, operator, or layer;
- an experimental research implementation;
- a compatibility or correctness fix to an existing model.

Do not claim release support based only on a source file or registry row.

## Implement the model

1. Add `src/model_factory/<Family>/<ModelName>.py`.
2. Expose `Model(args_model, metadata)`.
3. Keep behavior-affecting parameters under `model.*` and document defaults and
   allowed values.
4. Make CPU/device placement explicit and avoid import-time `.cuda()` calls.
5. Fail early for unsupported shapes, modes, class-count representations, or
   optional features.
6. Isolate optional dependencies so unrelated maintained models do not import
   them unconditionally.
7. Record source and license for copied or adapted implementations.

The factory infers `num_classes` from metadata when the config does not provide it.
A model that requires an integer, per-dataset mapping, or another representation
must validate that contract itself and include regression tests.

Checkpoint behavior must be documented. The current shared loader can load a
matching subset with `strict=False`; a model contribution should not treat a
partial load as full compatibility without explicit evidence.

## Inventory and configuration

`src/model_factory/model_registry.csv` is the discovered model inventory. Update it
when adding a public model entry, but do not treat inventory as support evidence.

Create the first runnable configuration under `configs/experiments/`. Promote it
to `configs/demo/` only after focused tests and an applicable smoke run pass.

A maintained model promotion normally requires:

- model registry entry;
- config registry entry;
- regenerated `docs/CONFIG_ATLAS.md`;
- input/output and task compatibility documentation;
- update to `SUPPORTED_COMPONENTS.md` or `SUPPORTED_COMBINATIONS.md` when the
  release-supported surface changes;
- explicit known limitations.

For the ISFM family, read [ISFM/README.md](ISFM/README.md) and its component
inventory before adding an embedding, backbone, or task head.

## Tests

Use pytest under `test/`; do not rely on an `if __name__ == "__main__"` block as
the integration test.

Test at least:

- import and construction through the real model factory;
- required and optional parameter handling;
- input/output shape, dtype, and device;
- a representative forward pass;
- relevant `task_id` or `data_id` behavior;
- invalid shape or unsupported configuration rejection;
- CPU behavior and CUDA behavior when CUDA support is claimed;
- checkpoint save/load behavior when compatibility is claimed;
- optional dependency absence when the dependency is not needed by the selected
  model.

Example focused commands:

```bash
python -m py_compile src/model_factory/<Family>/<ModelName>.py
python -m pytest <focused-test-file> -q
python -m scripts.config_inspect --config <yaml> --dump targets --format yaml
```

Before merging shared model-factory changes, also run:

```bash
python -m pytest test/ -q
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Pull-request evidence

The PR should include:

- method and user/research purpose;
- source paper or specification and code license;
- constructor and forward contracts;
- supported tasks and known incompatible combinations;
- parameter table or authoritative config link;
- dependency and hardware requirements;
- focused tests and exact outcomes;
- smallest runnable config;
- checkpoint and migration notes;
- evidence level: import, assembly, smoke, mini-E2E, or benchmark protocol.

Synthetic forward tests establish software contracts, not industrial-dataset
accuracy or state-of-the-art performance.

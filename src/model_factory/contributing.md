# Contribute a Model

This page is the model-factory addendum to the repository-wide
[contributor guide](../../CONTRIBUTING.md). The current factory and model
selection contract are described in [`README.md`](README.md).

## Factory contract

A public model is selected by:

```yaml
model:
  type: "<family>"
  name: "<module>"
```

The factory imports:

```text
src.model_factory.<type>.<name>
```

The module must expose:

```python
class Model(torch.nn.Module):
    def __init__(self, args_model, metadata):
        ...

    def forward(self, x, data_id=None, task_id=None):
        ...
```

The exact `forward` arguments and output depend on the intended task. Document
that contract rather than assuming every model returns classification logits.

Do not add model-specific branches to `main.py`, a pipeline, or the generic
factory. If a model requires special composition, express it through the model
module, reusable components, and validated config fields.

## Choose the correct family

Place the implementation under the nearest existing family, for example:

```text
src/model_factory/ISFM/
src/model_factory/CNN/
src/model_factory/RNN/
src/model_factory/Transformer/
src/model_factory/MLP/
src/model_factory/NO/
src/model_factory/X_model/
```

Create a new family only when the existing layout cannot represent a coherent
set of models. Do not copy a full existing model merely to change defaults.

## Document the model contract

The pull request must state:

- expected input shape, dtype, device, and required batch metadata;
- output shape and semantic meaning for every supported `task_id`;
- constructor/config parameters, defaults, and allowed values;
- required and optional dependencies;
- supported tasks/losses and explicitly rejected combinations;
- CPU/GPU behavior and precision constraints;
- checkpoint save/load behavior and compatibility boundary;
- source paper, upstream code, and license when porting an implementation;
- known numerical, memory, sequence-length, or data-shape limits.

A parameter that is parsed but not consumed must not be documented as active.

## Optional dependencies

Keep optional model-family dependencies behind the selected module boundary.
Selecting a lightweight model should not import an unrelated optional backbone.
When a selected model needs an unavailable package, raise an actionable error
that names the component and installation requirement.

Avoid import-time downloads, global device allocation, and unconditional `.cuda()`.
Parameters and buffers should follow normal PyTorch device movement.

## Register and configure

Add or update the row in:

```text
src/model_factory/model_registry.csv
```

The registry records discoverability and the module path. It does not establish
release support.

Start with a portable config under `configs/experiments/`, usually by copying the
nearest maintained demo and changing only the model block. Do not commit an
external checkpoint or personal path.

```bash
python -m scripts.config_inspect \
  --config configs/experiments/<model>_smoke.yaml \
  --dump targets \
  --override trainer.num_epochs=1
```

## Add focused tests

Tests belong under `test/`, not under an `if __name__ == '__main__'` block in the
model source.

Cover at least:

- dynamic import and construction from an `args_model` fixture;
- valid forward shape, dtype, device, and finite output;
- configuration defaults and invalid values;
- task-specific outputs or dispatch;
- missing metadata/data-ID errors when applicable;
- optional module enabled/disabled behavior;
- CPU path and CUDA path when the model requires CUDA;
- checkpoint round-trip or migration when persistence changes;
- missing optional dependency behavior.

Example assertions:

```python
assert output.shape == expected_shape
assert output.device == input_tensor.device
assert torch.isfinite(output).all()
```

## Run the integration path

After focused tests, inspect and run the smallest applicable experiment:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/experiments/<model>_smoke.yaml \
  --override trainer.num_epochs=1
python main.py \
  --config configs/experiments/<model>_smoke.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m pytest test/ -q
```

Use a legal tiny fixture or the repository Dummy path for software integration.
Do not describe a synthetic or one-epoch result as benchmark evidence.

## Promote a model to release support

Promotion requires more than a registry row:

- maintained config and config-registry entry;
- passing factory/contract tests;
- a passing stated smoke command;
- task/loss compatibility evidence;
- documentation of dependencies and limitations;
- generated `docs/CONFIG_ATLAS.md` when a shipped config changes;
- updates to support documents only when the public boundary changes.

## Checklist

- [ ] Correct model family and dynamic module path used.
- [ ] `Model(args_model, metadata)` contract implemented.
- [ ] Input/output, metadata, task, dtype, and device behavior documented.
- [ ] Optional dependencies are lazy and errors are actionable.
- [ ] Registry row matches the actual module.
- [ ] Portable experimental config added.
- [ ] Focused construction/forward/error tests added under `test/`.
- [ ] Checkpoint compatibility considered.
- [ ] Minimal config-first smoke executed or marked `NOT_EXECUTED`.
- [ ] No unsupported performance or compatibility claim added.

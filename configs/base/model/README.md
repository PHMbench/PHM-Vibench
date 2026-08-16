# `configs/base/model/`

The `model` block selects the implementation constructed by Model Factory. Model Factory
owns model identity, construction, and explicitly requested external weights. It does not
read data, select a task, move a model to a device, or repair incompatible input shapes.

## Transparent baseline

Use the global-average linear classifier when checking data, label, split, metric, and
Factory wiring before introducing a more complex representation:

```yaml
base_configs:
  model: "configs/base/model/global_average_linear.yaml"

model:
  input_dim: 2
```

The implementation computes:

```text
[B, L, C] --mean over L--> [B, C] --linear--> [B, K]
```

`K` is derived from the validated metadata label ontology unless `model.num_classes` is
explicitly supplied. A channel mismatch fails; the model does not pad, truncate, repeat,
or select a different channel count.

Runnable example:

```bash
phmfactory preflight \
  --config configs/demo/00_smoke/dummy_global_average_linear.yaml

phmfactory \
  --config configs/demo/00_smoke/dummy_global_average_linear.yaml
```

This example changes only the model base relative to the packaged Dummy ISFM smoke.

## ISFM example

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
```

ISFM components must be shape-compatible. HSE patch sizes must fit the actual input; the
embedding does not repeat time or channels to satisfy an invalid configuration.

## Core fields

| Field | Meaning |
| --- | --- |
| `model.type` | Top-level family used by Model Factory. |
| `model.name` | Concrete implementation under that family. |
| `model.input_dim` | Declared input channel count where consumed. |
| `model.num_classes` | Optional explicit output classes; otherwise derived from valid metadata. |
| `model.weights_path` | Optional external checkpoint; missing or incompatible files fail. |
| `model.weights_strict` | Defaults to strict loading; set false only for explicit transfer learning. |
| `model.embedding/backbone/task_head` | Required component IDs for `model.type=ISFM`. |

## Extension rule

A new model should require only:

1. one implementation under `src/model_factory/<TYPE>/`;
2. one configuration under this directory or an experiment directory;
3. one row in `src/model_factory/model_registry.csv`;
4. a focused forward/backward contract test.

Do not modify `main.py`, Data Factory, Task Factory, Trainer Factory, or Pipeline code merely
to select the model. A valid replacement changes the model block and nothing else.

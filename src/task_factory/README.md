# Task Factory

The Task Factory wraps the selected model in the Lightning task identified by
`task.type` and `task.name`.

```python
from src.task_factory import build_task

task = build_task(
    args_task=args_task,
    network=model,
    args_data=args_data,
    args_model=args_model,
    args_trainer=args_trainer,
    args_environment=args_environment,
    metadata=metadata,
)
```

This is a construction fragment using already resolved configuration and metadata.
Tasks own objectives, estimator lifecycle and optimization, not device fallback or data
replacement. Failures must not substitute another task, zero loss or an easier objective.

## Configuration and resolution

A task fragment is:

```yaml
task:
  type: DG
  name: classification
  loss: CE
  metrics: [acc, f1]
  optimizer: adamw
  lr: 0.001
```

For `DG.classification`, the Factory checks the existing `TASK_REGISTRY`, imports
`src.task_factory.task.DG.classification`, then checks for decorator registration. The
historical exported class `task` remains a compatibility path. Do not add another class
name guessing strategy or implement both paths without a real compatibility need.

A new task can use `@register_task("MyTaskType", "my_task")` on its class in
`src/task_factory/task/MyTaskType/my_task.py`. See [Default_task.py](Default_task.py) for
the actual constructor and lifecycle, and [components](Components/README.md) for loss and
metric inputs. Do not copy an illustrative fragment as a complete runnable task.

## Batch and metadata

`DG.tii_joint` accepts the projected common/increment inputs for the matched
support-conditioned tokenizer. It also accepts `E_03_Patch` and `E_01_HSE` on
the same source-normalized admitted union. HSE is returned to its effective
sampling grid only when the admitted signal survives the grid round trip.
`task.frozen_features` mean-pools the encoder output without invoking a source
head. The existing `scripts.tii_one_model` exposes frozen-episode export,
source-only alpha scoring, and target readout on saved features; the latter
requires an explicit `--output_dir` and emits probabilities alongside NLL,
macro-F1, accuracy and predictive entropy for raw and source-feature-RMS
scales. Query metrics weight physical groups equally. Source task and encoder
identities are checked, but caller-supplied provenance is not independent proof
that a checkpoint or feature file was produced with those identities.

The same frozen-episode entrypoint also accepts `--fisher-episode` and an
explicit checkpoint from the official FISHER inference implementation. Its
feature specification declares whether the checkpoint is a random fixture or
externally pretrained, whether incomplete time/frequency coordinates may be
discarded, and the pretraining provenance and excluded episode identities for
external weights. The supplied YAML owns DataFactory inputs; this mode does not
fit the FISHER encoder or use the YAML model section. The output records the
official log-STFT geometry and discarded coordinates. It uses the same source
RMS, episode loader, source-only readout selection and target evaluator as the
other frozen features. Source selection and target features must declare the
same checkpoint path and kind; callers must preserve the file and independently
verify any claimed pretraining exclusions.
The optional dependency is `omegaconf`; the official MIT license accompanies
the source. Random-fixture execution is not a pretrained FISHER result.

Document the dictionary fields actually consumed, such as `x`, `y`, `file_id`,
`domain_id`, and `mask`. Preserve per-sample identity and metadata through the batch;
`x, y = batch` is not a replacement for that contract. Configure the correct explicit
dataset adapter when needed; do not catch an import failure and use `Default_dataset`.

Metric construction requires validated metadata and, on maintained paths, `loss_name`.
Aliases and logged estimator names belong to the existing metric helpers. Do not invent
missing values, average batch-level F1, or use argmax labels as AUROC scores. Code and tests
must determine whether every declared estimator and expected population was evaluated.

`DG.classification` with `TSPN_fusion` uses
`network.forward_details(batch["x"])["candidate_logits"]` for training, validation,
and test. The model's separate deployment mixture (including its `alpha=0` reference
default) is not the classifier evaluated by this task. A restored task uses the same
candidate path, while the reference parameters and running statistics stay frozen.

## Verification and support

Inspect and run the same complete configuration:

```bash
phmfactory preflight --config <yaml>
phmfactory --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Run the relevant objective, estimator and Task tests. A registry row or import is only
implementation discovery; `sanity_ok` software execution is not a stable release or
`baseline_valid` scientific claim. Use the existing [supported combinations](../../SUPPORTED_COMBINATIONS.md)
and [contribution guide](contributing.md), not a second support table here.

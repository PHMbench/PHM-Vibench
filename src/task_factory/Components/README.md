# Task components

These modules supply mathematical operations used by Tasks. They do not own data
selection, device fallback, experiment orchestration, or support promotion. Read the
[Task Factory guide](../README.md) before adding a Task.

## Loss interfaces

[loss.py](loss.py) provides:

```python
get_loss_fn(loss_name: str)
prepare_loss_inputs(loss_name, predictions, target)
```

For the maintained supervised path, select `task.loss` explicitly. CE and NLL consume
class-index targets; BCE consumes one score and one target per sample; regression losses
consume matching continuous predictions and targets. Use `prepare_loss_inputs` for the
implemented shape and dtype contract, not a blanket conversion of every target to long.

Inside a Task, with model outputs and targets already obtained from the declared batch:

```python
from src.task_factory.Components.loss import get_loss_fn, prepare_loss_inputs

loss_fn = get_loss_fn("CE")
logits, labels = prepare_loss_inputs("CE", predictions, target)
loss = loss_fn(logits, labels)
```

This is a component fragment, not a complete training script. The enclosing Task defines
how `predictions` and `target` are produced. Do not replace its dictionary batch contract
with `x, y = batch`, pass mixup arguments to plain CrossEntropyLoss, or catch TypeError to
retry another loss signature. A failed objective call must not select a different loss.

## Metric interfaces

[metrics.py](metrics.py) provides:

```python
get_metrics(metric_names, metadata, *, loss_name=None)
prepare_metric_inputs(metric_name, predictions, target, *, loss_name)
```

Metadata is required. The maintained Task supplies `loss_name`; omission exists only for
low-level compatibility. With the Task's validated metadata:

```python
from src.task_factory.Components.metrics import get_metrics

metrics = get_metrics(["acc", "f1"], metadata, loss_name="CE")
```

The result is a ModuleDict keyed by metadata `Name`, with stage-specific entries such as
`train_acc`, `val_acc`, and `test_f1`. Classification construction validates the label
ontology; do not infer missing classes from a single evaluation batch.

Use separate stage states: update on batches, compute over the evaluation population,
then reset at the stage boundary. AUROC needs continuous scores, not argmax labels. Use
`prepare_metric_inputs` for the selected estimator. Do not average batch F1 values or fill
missing metrics with zero. See [Default_task.py](../Default_task.py) for the implemented
lifecycle. This guide does not claim that namespace collisions or declared-metric
publication checks are already resolved.

## Contrastive and generative components

- [contrastive_losses.py](contrastive_losses.py) contains InfoNCE, SupCon, Triplet,
  Prototypical, BarlowTwins, and VICReg implementations. Their input contracts differ;
  inspect the selected implementation and its Task caller before use.
- [contrastive strategy notes](README_CONTRASTIVE_STRATEGIES.md) describe research
  composition. They are not evidence that every loss combination is maintained.
- `flow.py` / `FlowLoss` and `mean_flow_loss.py` / `MeanFlow` are experimental helpers.
  A helper test does not establish a complete Pipeline 06 method or benchmark result.
- Model-side masked reconstruction is documented with its model implementation, not
  treated as a generic supervised-loss substitute here.

No universal argument retry, zero-loss repair, new component manager, or extra registry
is required to document these boundaries.

## Verification

For documentation, run `python -m scripts.validate_docs`. For changes to supervised loss
or metric behavior, use the existing `test/test_task_estimator_truth.py` and the affected
Task tests. Select additional contrastive/generative tests only for the component changed.
A real-data rerun is not a documentation-edit requirement.

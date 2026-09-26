# Adaptation protocol

This document freezes the scientific protocol for test-time adaptation (TTA), continual
TTA, source-free domain adaptation (SFDA), delayed-label adaptation and supervised
continual controls. It does not claim that Tent, SAR, CoTTA, SHOT or another adaptation
algorithm is implemented.

The experiment object is

\[
\mathcal E_{\mathrm{adapt}} =
(\mathcal D_s,\mathcal S_t,f_{\theta_0},\Pi,\mathcal A,\mathcal U,\widehat R).
\]

The source information, ordered target stream, source model, protocol, adaptation
objective, state update and estimator must all match the visible request.

## Regimes and labels

The protocol schema distinguishes:

- `source_only`: frozen control, no adaptation update;
- `episodic_tta`: target-label-free adaptation with reset per episode;
- `online_tta`: target-label-free ordered stream without replay;
- `continual_tta`: persistent target-label-free non-stationary stream;
- `offline_sfda`: unlabeled adaptation population, then a distinct evaluation population;
- `continual_sfda`: persistent source-free adaptation across target populations;
- `delayed_label_adaptation`: labels become available only after their declared delay;
- `online_supervised_continual`: labels are available online and must not be reported as TTA.

A TTA/SFDA adaptation view never contains `y`. Delayed or online-supervised labels are
delivered through a separate label event rather than added to the ordinary adaptation
view.

## Required protocol dimensions

| Field | Values | Meaning |
| --- | --- | --- |
| `source_access` | `checkpoint_only`, `checkpoint_plus_artifact`, `source_data_available` | source information allowed at deployment |
| `target_label_access` | `none`, `delayed`, `online_supervised` | when target labels may affect updates |
| `timing` | `predict_then_update`, `update_then_predict` | whether the evaluated prediction is before or after the current unlabeled update |
| `state_persistence` | `episodic_reset`, `domain_reset`, `persistent` | when adaptation state is restored |
| `domain_boundary` | `hidden`, `known` | whether domain identity may be observed |
| `label_space` | `closed_set`, `partial_set`, `open_set` | relationship between source and target labels |
| `passes` | positive integer | number of passes over an offline adaptation population |

Online regimes require one pass. `domain_reset` requires a known boundary. Continual
regimes require persistent state. Offline/continual SFDA require distinct
`adapt_population` and `evaluation_population`. `checkpoint_plus_artifact` must name
the source artifacts explicitly.

## Causal prequential default

For online PHM diagnosis the default scientific estimator is:

\[
\hat y_t=f_{\theta_{t-1}}(x_t),\qquad
\widehat R_t=\operatorname{Eval}(\hat y_t,y_t),\qquad
\theta_t=\mathcal U(\theta_{t-1},x_t).
\]

This is `predict_then_update`. The evaluator may read `y`; the adapter may not.

`update_then_predict` is a separate transductive protocol:

\[
\theta_t=\mathcal U(\theta_{t-1},x_t),\qquad
\hat y_t=f_{\theta_t}(x_t).
\]

The two estimators must not be pooled.

## Views and ownership

Data Factory owns stream order and traceability. An adaptation batch may expose
`x`, optional `mask`, `sample_id`, `timestamp`, `sequence_id` and explicitly allowed
physical metadata. `file_id` is evaluator-only in B00 because repository file-number
ranges can encode the target class; adapters must not receive it as an identity shortcut. B00 uses a positive metadata schema rather than an
arbitrary pass-through: normalized keys are limited to physical quantities such as
`speed_rpm`, `shaft_rate_hz`, `load_hp`, `torque`, `temperature`, and
`sample_rate`. Target-derived aliases, including `Label`, `Label_Description`,
`fault_label`, `fault_type`, `condition_id`, and class/target label variants are
rejected. An unrecognized metadata key fails closed and must be reviewed before entering
the allowlist. `domain_id` is available only when `domain_boundary=known`.

The evaluator receives `y` separately. Model Factory still owns only the backbone and
explicit source checkpoint. Task Factory owns the adaptation objective and update
semantics once a concrete method exists. Trainer Factory will own device and state
checkpoint/resume. B00 does not add an adaptation Trainer.

## Source artifacts

Examples include Fisher information, source prototypes, source entropy distributions,
normalization statistics and source feature moments. An experiment that needs them must
declare `source_access=checkpoint_plus_artifact` and list them in `source_artifacts`.
Such a method is not equivalent to a pure checkpoint-only method.

## Delayed labels and SFDA

A delayed-label event must include `label_available_step`. Before that step the helper
raises rather than releasing the label. This prevents forecasting or delayed-supervision
experiments from reading future truth at prediction time.

B00 freezes the schemas for delayed-label, online-supervised and SFDA regimes but does
not execute those lifecycles. The dependency-light `execute_protocol_step()` is only a timing/isolation probe for
persistent `online_tta` / `continual_tta`. `source_only` is executed separately by the bounded B01 runtime described below,
not by this protocol helper;
`episodic_tta` and `domain_reset` likewise remain schema-level until an explicit reset
lifecycle exists. Label-bearing regimes need an explicit
label-event runtime; SFDA needs separate adaptation and evaluation populations. Passing
any unsupported lifecycle through the B00 single-stream executor fails rather than silently
running a different experiment.

## What is runnable after B00?

Nothing new. `configs/base/task/tta_protocol.yaml` is a validated protocol fragment, not
a registered Task. No TTA dataset adapter or algorithm is registered. The B01 Python runtime below adds a frozen control without registering a TTA Task
or changing public CLI dispatch. A protocol fragment is still not a runnable experiment.

## Evidence required before an algorithm is supported

A later algorithm PR must record paper, official repository, fixed source revision,
license, exact updated parameter subset, fixed input/source weights, a reference update
and numerical tolerance. Import, registry presence or shape-only forward evidence is not
algorithm fidelity.


## B01: source-only ordered-stream runtime

`phmfactory.source_stream.run_source_only_stream` accepts an already constructed model,
an explicitly prepared target-only DataLoader, the source-only protocol, an explicit
source checkpoint and the caller's evaluator. It strictly restores weights through the
existing Model Factory loader and performs **no fit, backward, optimizer step or update**.

The supported protocol is `source_only / checkpoint_only / none / predict_then_update /
persistent / closed_set / passes=1`. Either hidden or known domain boundaries may be
declared, but the B01 model receives only `x`. File IDs, labels, domain IDs and all
traceability fields remain evaluator-only; no per-file head or metadata-based model
forward is claimed. Other protocols, masked inputs and unsupported loaders fail closed.

The caller explicitly calls `model.eval()` before execution and owns model/input device
placement. B01 rejects training-mode submodules or existing gradients instead of fixing
them. After strict checkpoint loading, each batch is checked against one copy of all
registered parameters and buffers, including non-persistent buffers. Changes in values,
identities, modes, gradient flags or registered modules cause failure before the next
prediction. A mutation during forward fails before that prediction reaches the evaluator.
There is no automatic restoration that could conceal a failed source-only experiment.

The accepted loaders are synchronous (`num_workers=0`) with either the ordinary
SequentialSampler/BatchSampler or the existing unshuffled, non-dropping
Same_system_Sampler using `Dataset_id`. The latter's complete index coverage is checked;
its system-grouped order is retained, **not relabelled chronological order**. B01 does not
construct populations, shuffle, sort timestamps, change split/windowing, cast inputs,
move devices or fit normalization. An upstream producer must supply the intended order
and frozen preprocessing. A loader that drops samples fails the complete-population gate.

### Equivalence contract and usage

The reference is ordinary `model.eval()` plus `torch.no_grad()` inference, using the same
checkpoint, initial non-persistent buffers, inputs, preprocessing, order **and batch
partition**. TimesNet selects periods using a batch aggregate, so changing batch size is
not part of the equality claim. The CPU BatchNorm/Dropout fixture is checked exactly;
actual GlobalAverageLinear and TimesNet Model Factory paths use `rtol=1e-6, atol=1e-7`.
No arbitrary hardware/precision or batch-size invariance is claimed.

The following is the integration seam; `target_loader`, `model`, `source_metadata`,
`source_checkpoint` and `result_dir` are explicit caller-owned inputs. The model has a
single K>=2 CE logit head fixed by the source ontology, not inferred
from target labels. For brevity this example assumes one dataset metric namespace.

```python
from pathlib import Path
import pandas as pd
from phmfactory.source_stream import run_source_only_stream
from src.config_schema import AdaptationProtocolConfig
from src.task_factory.Components.metrics import get_metrics, prepare_metric_inputs
from src.utils.run_summary import write_run_summary

protocol = AdaptationProtocolConfig(
    regime="source_only", source_access="checkpoint_only", target_label_access="none",
    timing="predict_then_update", state_persistence="persistent",
    domain_boundary="hidden", label_space="closed_set",
)
model.eval()  # explicit; architecture/device/source class ontology are already fixed
metrics = get_metrics(["acc", "f1"], source_metadata, loss_name="CE")[dataset_name]

def evaluate(logits, view):
    for name in ("acc", "f1"):
        pred, target = prepare_metric_inputs(name, logits, view["y"], loss_name="CE")
        metrics[f"test_{name}"].update(pred, target)

sample_count = run_source_only_stream(
    model, target_loader, protocol, checkpoint_path=source_checkpoint, evaluate=evaluate,
)
# Only after complete success: aggregate over the population, never average batch F1.
result = {f"test_{name}": float(metrics[f"test_{name}"].compute()) for name in ("acc", "f1")}
result_dir = Path(result_dir)
result_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame([result]).to_csv(result_dir / "all_results.csv", index=False)
write_run_summary(result_dir / "run_summary.json", [result], [source_seed])
```

No new result schema, registry, Trainer or CLI is introduced. The Data Factory owns data
preparation; **do not run an ordinary training Data Factory over source records inside a
checkpoint-only deployment**. The integration tests construct the existing ordinary
Dummy factory as a test fixture and pass only its test loader into B01. They are not
claims about source-data access during real deployment. No training runs in these tests.

### Evidence and remaining boundaries

The focused tests execute real torch modules, the native Data/Model factories, canonical
raw and Lightning-prefixed checkpoints, existing Task metrics and existing result summary
writing. True/permuted/zero labels yield identical predictions, model state and RNG state
in the fixed fixture. Deliberate BatchNorm, parameter, non-persistent-buffer, mode and
evaluator-induced mutations fail. Empty, randomized, dropping or incomplete streams,
wrong checkpoints, invalid protocols and non-finite outputs also fail. The existing
public-package CI runs these tests again from the installed wheel outside the checkout.

The state checks have one full-state-copy memory cost and O(model-state-size) comparison
cost each batch. They are correctness checks, not a low-latency performance claim. The
runner uses trusted datasets, collators, model modules and evaluator code; it is not a
sandbox against malicious Python closures or an audit of arbitrary unregistered Python
state. Stochastic transforms and data-dependent mutable Python caches require their own
qualification. Discard partial evaluator state after any exception; do not publish it as
a completed population result.

These tests establish the bounded frozen control, not industrial PHM usefulness or TTA
algorithm fidelity. No TTA Task/CLI, adaptive optimizer, EMA, replay, SFDA, reset or resume
lifecycle is enabled. B01 requires exact-head CI and independent review before merge;
Tent remains a separate decision after that gate.

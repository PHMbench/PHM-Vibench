# Adaptation protocol

This document freezes the scientific protocol for test-time adaptation (TTA), continual
TTA, source-free domain adaptation (SFDA), delayed-label adaptation and supervised
continual controls. B01 supplies a frozen Python control and B02 supplies the bounded
Tent Python path below; SAR, CoTTA, SHOT and other algorithms remain outside this scope.

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
The separately bounded B02 implementation follows below.


## B02: one-step prequential Tent

`phmfactory.tent_stream.run_tent_stream` reuses B01's complete ordered loader contract,
canonical strict source-checkpoint loading and the caller's existing Task metric/output
path. `src.task_factory.Components.tent.Tent` owns entropy, the BN-affine parameter
subset and Adam. No TTA Task/CLI, dataset adapter, Trainer or registry is added.

The fixed source is [official Tent](https://github.com/DequanWang/tent) at
`e9e926a668d85244c66a6d5c006efbd2b82e83e8` (MIT; attribution is retained in the component
and [test fixture](../../test/fixtures/tent_upstream/README.md)). The production extension
admits affine BatchNorm1d as well as the original BatchNorm2d. Other normalization
families, non-affine BN and BN-free models fail; they are not modified to fit Tent.

### The exact update and estimator

B02 accepts only `online_tta` or `continual_tta`, `checkpoint_only`, no target labels,
`predict_then_update`, persistent state, closed-set classification and one pass. Each
batch makes exactly one model forward and one Adam step. The learning rate must be
explicit; betas=(0.9,0.999), eps=1e-8 and weight_decay=0. No scheduler, multi-step update,
pseudo-label loss, confidence filter, EMA, replay, reset, SFDA or supervised update is
included. Freeze the learning rate using source validation or a declared development
shift before observing target labels; there is no target-labelled selector in this API.

For an ordered batch $B_t$, with current-batch BN statistics and train-mode dropout:

$$
p_t=\operatorname{softmax}(f_{\theta_{t-1};\,\mathrm{BN}(B_t)}(B_t)),\qquad
L_t=-\frac{1}{|B_t|}\sum_{i\in B_t}\sum_k p_{ik}\log p_{ik}.
$$

The runtime evaluates the isolated pre-update prediction, then applies the entropy
gradient to BN scale/shift only. It reuses that exact forward graph, rather than
performing a second stochastic forward. The upstream one-step function likewise returns
pre-update logits. `update_then_predict` and more than one step are not silently mapped
onto this estimator.

This is **batch-prequential in the parameter update**, not strict sample-causal
inference: every prediction can use other inputs in its current batch through BN.
The first Tent prediction already uses batch statistics and training-mode dropout;
it is not the B01 frozen-source prediction even before the first Adam step. A later
scientific comparison should keep the frozen source and a separately declared BN-only
control distinct. Identical batch partitions and stochastic state are part of fidelity.

### Calling the bounded runtime

This extends the B01 construction fragment; `model`, `target_loader`, source checkpoint,
source-selected learning rate and evaluator already exist. It is not a runnable YAML
preset, a data-download command or a training command.

```python
from phmfactory.tent_stream import run_tent_stream

protocol = AdaptationProtocolConfig(
    regime="online_tta", source_access="checkpoint_only", target_label_access="none",
    timing="predict_then_update", state_persistence="persistent",
    domain_boundary="hidden", label_space="closed_set",
)
model.eval()
adapter = run_tent_stream(
    model, target_loader, protocol, checkpoint_path=source_checkpoint,
    learning_rate=source_selected_lr, evaluate=evaluate,
)
assert adapter.num_samples == len(target_loader.dataset)
# Now compute population metrics and use the same all_results.csv/run_summary writer.
```

Only `x` reaches `Tent.predict`; `Tent.adapt()` accepts no external labels or loss. IDs,
label aliases and even known domain IDs remain evaluator-only for this x-only method.
All registered state is checked before/after prediction and before/after update; only
BN affine parameters may change at the update. Failures keep the original exception,
produce no success return, and require discarding the partial model, optimizer and
metric state. There is no rollback-and-continue path. Stateful Python caches and hostile
closures remain outside the trusted-module contract described under B01.

### Saved algorithm state versus whole-experiment resume

At a completed batch boundary `Tent.state_dict()` supplies ordinary model/optimizer
state, non-persistent buffers, counters, last entropy and CPU/relevant CUDA torch RNG.
`Tent.load_state_dict()` requires compatible architecture, Adam settings and device kind.
These are payload methods for an enclosing checkpoint owner, not a second checkpoint
writer or Trainer. In the CPU dropout fixture, ordinary `torch.save` followed by a new
Python process and `torch.load` reproduces the next prediction, model/Adam state and RNG
exactly. Saving a pending, not-yet-adapted batch is rejected.

The full-pass `run_tent_stream` entrypoint does **not** resume an entire experiment.
A future Trainer-owned integration must also restore the ordered stream cursor,
preprocessing RNG and evaluator accumulator. Algorithm-state continuation alone must not
be advertised as resumable end-to-end execution. CUDA execution/resume is not qualified
by CPU tests; no GPU fallback is made.

### Evidence boundary

The unmodified upstream oracle covers BN2d and a controlled BN1d operator extension.
Tests compare logits, losses, gradients, Adam moments, model parameters and RNG, and
replace labels with permuted/zero values while preserving all adaptation trajectories.
They also execute the existing ResNet1D Model Factory and native Data Factory test loader
on bundled Dummy, reuse B01 frozen inference, and check population F1 and result output.
The standard public-package workflow repeats B02 from an installed wheel outside the
checkout. Initial failed oracle-layout checks are retained as diagnostics, not passes.

No industrial dataset or trained industrial checkpoint has been evaluated for B02.
This is algorithm/framework verification, not E3 utility, safety, robustness, CTTA
non-forgetting or `benchmark_ready` evidence. O(model-state) checks and the source-state
copy are not an adaptation latency or peak-memory benchmark. Source weights, target
order and target preprocessing remain caller-owned and must be fixed in a real study.


## B03: Sharpness-Aware and Reliable Entropy Minimization (SAR)

`phmfactory.sar_stream.run_sar_stream` reuses the B01 ordered-loader contract, strict
source checkpoint loading and the caller's existing population evaluator. The algorithm
component is `src.task_factory.Components.sar.SAR`. No TTA Task registration, second
Trainer, dataset adapter, result system or public CLI is introduced.

The fixed reference is the official `mr-eggplant/SAR` repository at
`20f6e24b17525f34503510afccedc0629b67b7c4` (BSD-3-Clause). The source-equivalent `sar.py`, `sam.py` and license are retained as test-only fixtures; only repository-required trailing whitespace normalization is applied. B03 preserves the official
BatchNorm2d, GroupNorm and LayerNorm update and top-layer exclusion rules, and adds an
operator-equivalent affine BatchNorm1d path for PHM 1-D models. No normalization layer is
inserted or replaced to make an incompatible model pass.

### Exact B03 transaction

B03 accepts only `online_tta` or `continual_tta` with `checkpoint_only`, no target
labels, `predict_then_update`, persistent state, closed-set classification and one pass.
The caller supplies an explicit learning rate and reliable-entropy margin before target
labels are observed. The SAM radius is fixed by default to the official `rho=0.05`; the
base optimizer is SGD with momentum 0.9 and no weight decay. The official recovery
threshold is fixed to 0.2. There is no target-labelled selector, scheduler, replay,
teacher, pseudo-label objective or extra target pass.

For current batch $B_t$, first compute logits and reliable-sample entropy

$$
z_t=f_{\theta_{t-1}}(B_t),\qquad
H_i(z_t)=-\sum_k p_{ik}\log p_{ik},\qquad
I_1=\{i:H_i(z_t)<E_0\}.
$$

The evaluator receives an isolated copy of $z_t$ **before** the current update. SAR then
uses the same first-forward graph to obtain the reliable entropy gradient, performs the
SAM ascent step

$$
\hat\epsilon(\theta)=\rho\frac{g}{\lVert g\rVert_2+10^{-12}},
$$

runs the method's required second forward at $\theta+\hat\epsilon$, applies the reliability
filter again, restores $\theta$, and performs the SGD-momentum descent step using the
second gradient. Thus B03 makes **two method forwards per updated batch** while exposing
only the first pre-update logits to evaluation. A random-number change by the evaluator
between the two forwards is rejected, rather than silently changing the dropout mask and
therefore the adaptation trajectory.

Like Tent, this is batch-prequential in parameter updates rather than strict sample-causal
inference: BatchNorm can couple examples inside the current batch. Batch partition is part
of the protocol and no batch-size invariance is claimed.

### Reliable filtering and recovery edge semantics

The official code takes a mean over an empty reliable set. In modern PyTorch that value is
NaN while its gradient can be zero, which can still interact with accumulated SGD
momentum. B03 does not treat that artifact as a scientific update. If either reliability
stage selects zero samples, the current batch is evaluated but the parameter update is
explicitly skipped; the SAM perturbation, if already created, is restored without a base
optimizer step. This deviation is stated and tested rather than hidden behind a warning
or fallback.

After a successful second-stage update, B03 updates the official entropy EMA

$$
\bar H_t=0.9\bar H_{t-1}+0.1H_t.
$$

When the EMA falls below 0.2, the configured source model and source optimizer state are
restored as the SAR recovery action. This is **algorithm-internal recovery**, not the
protocol's `episodic_reset` or `domain_reset`; the ordered stream continues and the method
remains a persistent TTA process. The just-computed EMA is retained after recovery to
match the official forward ordering. The implementation resets SGD momentum together
with the source model, matching the intended source-state recovery rather than relying on
the upstream SAM wrapper's incomplete optimizer serialization.

### State and leakage boundary

Only `x` reaches `SAR.predict`. Labels, `file_id`, label-derived aliases and domain IDs are
kept in the evaluator view. `SAR.adapt()` accepts no label, external loss or new input.
True, permuted and all-zero target labels must produce identical predictions and complete
model/optimizer/EMA/RNG trajectories when source checkpoint, stream, batch partition and
seed are fixed.

At a completed batch boundary `SAR.state_dict()` contains current model state, explicit
non-persistent buffers, SGD momentum, the recovery anchor, EMA/counters and the relevant
torch RNG. `load_state_dict()` rejects changed hyperparameters, malformed counters,
non-finite or missing momentum, incompatible recovery anchors and changed device kind.
This is an **algorithm-state payload**, not end-to-end experiment resume. Stream cursor,
preprocessing RNG and evaluator state still belong to a future Trainer-owned resume
integration.

### Evidence boundary

The numerical oracle checks the official BN2d/GN/LN implementation over multiple batches,
including logits, adapted parameters, optimizer trajectory and EMA. A paired BN1d/BN2d
fixture verifies the declared 1-D operator extension. Tests also cover source-top-layer
exclusion, recovery, empty reliable sets, evaluator RNG/state mutation, new-process
continuation, label permutation, population Macro-F1, existing result writing, and the
actual ResNet1D Model Factory plus native Dummy target loader. The public-package workflow
repeats B03 from the installed wheel outside the repository checkout.

These gates establish implementation fidelity for the declared bounded path, not E3 PHM
utility. No industrial checkpoint/data, GPU efficiency, A→B→A forgetting, small-batch
superiority, safety guarantee or `benchmark_ready` status is claimed. SAR's motivating
wild-stream benefits must be tested separately on fixed PHM shifts before any such claim.

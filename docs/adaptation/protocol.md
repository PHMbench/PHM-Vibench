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

- \`source_only\`: frozen control, no adaptation update;
- \`episodic_tta\`: target-label-free adaptation with reset per episode;
- \`online_tta\`: target-label-free ordered stream without replay;
- \`continual_tta\`: persistent target-label-free non-stationary stream;
- \`offline_sfda\`: unlabeled adaptation population, then a distinct evaluation population;
- \`continual_sfda\`: persistent source-free adaptation across target populations;
- \`delayed_label_adaptation\`: labels become available only after their declared delay;
- \`online_supervised_continual\`: labels are available online and must not be reported as TTA.

A TTA/SFDA adaptation view never contains \`y\`. Delayed or online-supervised labels are
delivered through a separate label event rather than added to the ordinary adaptation
view.

## Required protocol dimensions

| Field | Values | Meaning |
| --- | --- | --- |
| \`source_access\` | \`checkpoint_only\`, \`checkpoint_plus_artifact\`, \`source_data_available\` | source information allowed at deployment |
| \`target_label_access\` | \`none\`, \`delayed\`, \`online_supervised\` | when target labels may affect updates |
| \`timing\` | \`predict_then_update\`, \`update_then_predict\` | whether the evaluated prediction is before or after the current unlabeled update |
| \`state_persistence\` | \`episodic_reset\`, \`domain_reset\`, \`persistent\` | when adaptation state is restored |
| \`domain_boundary\` | \`hidden\`, \`known\` | whether domain identity may be observed |
| \`label_space\` | \`closed_set\`, \`partial_set\`, \`open_set\` | relationship between source and target labels |
| \`passes\` | positive integer | number of passes over an offline adaptation population |

Online regimes require one pass. \`domain_reset\` requires a known boundary. Continual
regimes require persistent state. Offline/continual SFDA require distinct
\`adapt_population\` and \`evaluation_population\`. \`checkpoint_plus_artifact\` must name
the source artifacts explicitly.

## Causal prequential default

For online PHM diagnosis the default scientific estimator is:

\[
\hat y_t=f_{\theta_{t-1}}(x_t),\qquad
\widehat R_t=\operatorname{Eval}(\hat y_t,y_t),\qquad
\theta_t=\mathcal U(\theta_{t-1},x_t).
\]

This is \`predict_then_update\`. The evaluator may read \`y\`; the adapter may not.

\`update_then_predict\` is a separate transductive protocol:

\[
\theta_t=\mathcal U(\theta_{t-1},x_t),\qquad
\hat y_t=f_{\theta_t}(x_t).
\]

The two estimators must not be pooled.

## Views and ownership

Data Factory owns stream order and traceability. An adaptation batch may expose
\`x\`, optional \`mask\`, \`file_id\`, \`sample_id\`, \`timestamp\`, \`sequence_id\` and
explicitly allowed physical metadata. Target-label aliases, including case or separator
variants such as \`Label\` and \`fault_label\`, are rejected from that metadata surface.
\`domain_id\` is available only when \`domain_boundary=known\`.

The evaluator receives \`y\` separately. Model Factory still owns only the backbone and
explicit source checkpoint. Task Factory owns the adaptation objective and update
semantics once a concrete method exists. Trainer Factory will own device and state
checkpoint/resume. B00 does not add an adaptation Trainer.

## Source artifacts

Examples include Fisher information, source prototypes, source entropy distributions,
normalization statistics and source feature moments. An experiment that needs them must
declare \`source_access=checkpoint_plus_artifact\` and list them in \`source_artifacts\`.
Such a method is not equivalent to a pure checkpoint-only method.

## Delayed labels and SFDA

A delayed-label event must include \`label_available_step\`. Before that step the helper
raises rather than releasing the label. This prevents forecasting or delayed-supervision
experiments from reading future truth at prediction time.

B00 freezes the schemas for delayed-label, online-supervised and SFDA regimes but does
not execute those lifecycles. The dependency-light \`execute_protocol_step()\` accepts
only \`source_only\`, \`episodic_tta\`, \`online_tta\` and \`continual_tta\`. Label-bearing
regimes need an explicit label-event runtime; SFDA needs separate adaptation and
evaluation populations. Passing either through the B00 single-stream executor fails
rather than silently running a different experiment.

## What is runnable after B00?

Nothing new. \`configs/base/task/tta_protocol.yaml\` is a validated protocol fragment, not
a registered Task. No TTA dataset adapter or algorithm is registered. The next bounded
change is a source-only ordered-stream runtime that must reproduce ordinary frozen
prediction under the same checkpoint, inputs and order before any adaptive method is
added.

## Evidence required before an algorithm is supported

A later algorithm PR must record paper, official repository, fixed source revision,
license, exact updated parameter subset, fixed input/source weights, a reference update
and numerical tolerance. Import, registry presence or shape-only forward evidence is not
algorithm fidelity.

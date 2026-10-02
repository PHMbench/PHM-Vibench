# Shared implementation for six paper experiments

The paper repositories reuse the following implementations from PHMFactory.
Their datasets, manuscript claims, historical checkpoints, and experiment results
remain separate. These additions do not promote any real-data combination to
`baseline-valid`.

| Paper | Implementation owner and boundary |
| --- | --- |
| P04 | `src/model_factory/MoE/M_05_FixedRouteMoE.py`: compact matched expert controls and fixed-route reference replacement; callers provide physical views and compatibility scores. |
| P05 | `src/model_factory/X_model/P4JointFuzzy.py`: normalized Gaussian feature classifier, initialized only with training features; the paper owns box certification and calibration. |
| P06 | `phmfactory/p06.py`: repeat-calibrated symbols, prototype decisions, and exact stored-affine certificates; caller-supplied pairing and physical transformation validity are not certified. |
| P07 | `src/model_factory/X_model/P07OperatorPath.py`: bounded operator DAG and selective extraction, with the same fidelity threshold for checked argmax and searched paths. |
| P08 | `src/model_factory/ISFM/M_P08_PhysicalConditioning.py` and `src/data_factory/p08_data.py`: actual HSE, shared head, same-information fusion controls, and explicit source-only condition encoding. |
| P09 | `src/task_factory/task/GFS/physical_prior_core.py` and `physical_prior.py`: support-only adaptation and explicit native/TorchScript source qualification; no query-driven fitting or model selection. |

P08's dedicated `scripts.p08_physical` entrypoint uses the public configuration
resolver and the existing Model Factory. It executes source-only tuning,
source-selected comparison, and completed-checkpoint mechanism experiments;
failed or mismatched comparison outputs cannot feed the mechanism stage.

P07 charges reference, argmax, ranking, candidate, and replay calls to one budget.
An already verified argmax path remains available when that budget expires, but
does not imply a minimum-cost result if cheaper paths remain unchecked. A focused
test counts executed methods independently of the returned counters.

The integration also retains four previously separate P09 runtime corrections
from `bac5668`, `ece8b1e`, `75c2e4a`, and `b064e44`: explicit HSE prompt inputs,
failure on invalid metadata, lazy selected-component loading, preservation of
task-module execution errors, and removal of unused prompt wrappers. Those
corrections were absent from the starting integration line `07a0355`.

Validation uses `LQ_signal` on CPU with synthetic fixtures. Focused tests cover
model outputs and gradients, exact certificate arithmetic, source export state,
independent unit boundaries, and extraction budgets. Public CLI help, doctor,
smoke preflight, and the bundled Dummy train/checkpoint/test lifecycle pass.
`scripts.validate_configs` and `scripts.validate_docs` pass. Import checks from
each paper working directory verify that the explicitly selected engine modules
resolve to this checkout, rather than a paper-local `src` package.

No industrial data, GPU benchmark, baseline tuning, real HSE source fitting, or
scientific performance comparison was executed for this integration. Historical
results remain attached to their original code and protocol versions.

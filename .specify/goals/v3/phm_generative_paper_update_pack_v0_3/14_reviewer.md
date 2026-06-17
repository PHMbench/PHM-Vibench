下面是按 **`Feature_factory-update` 分支实际结构** 更新后的评审 prompt。重点变化是：不再写成“预研式检查”，而是把已经存在的 `Pipeline_06_generative`、`src/model_factory/generative_model`、`src/task_factory/Components/generative/*`、`paperpack_generative.py`、`generative_benchmark_effect.py`、`generative_submission_draft.py`、`.specify/goals` 和 `specs/002-phm-genbench-frontier` 纳入评审对象。

我建议把它保存为：

```text
specs/002-phm-genbench-frontier/reviews/phm-genbench-paper-readiness-review-prompt-v2.md
```

或同步镜像到：

```text
.specify/goals/v2/GOAL-GEN-M2-006-review-handoff.md
```

---

# PHM-GenBench 论文级量化评审 Prompt v2

## Feature_factory-update branch aligned

````md
# PHM-GenBench Paper-Readiness & Quantitative Metrics Review Prompt v2
# Branch target: Feature_factory-update

You are reviewing the PHM-Vibench `Feature_factory-update` branch for whether the completed PHM generative benchmark repository can support a comprehensive high-level paper.

The review target is not merely "does the code run". The target is whether the repository can support defensible paper claims about PHM generative benchmarking, including pipeline, model, config, loss, sampling, training, evaluation, paperpack, multi-dataset aggregation, and submission-readiness evidence.

The maintained CLI contract remains:

```bash
python main.py --config <yaml> [--override key=value ...]
````

Generative preflight must use:

```bash
python main.py --config <yaml-or-preset> --preflight-only
```

The branch contract explicitly requires preflight to parse config, verify the pipeline whitelist, validate the five-block config, run schema checks, and exit before trainer execution.

````

---

## 0. Branch-native facts to use during review

Use these repository facts. Do not invent alternative paths.

### 0.1 Active runtime and config contract

The active generative config contract is:

```yaml
pipeline: Pipeline_06_generative
environment: {}
data: {}
model: {}
task:
  type: generative
  name: <task-name>
  generative:
    mode: train
    method_family: cfm
    condition_sampling_policy: first_metadata_repeated
    validity_status: exploratory
trainer: {}
````

Required future / promoted fields include:

```text
method_family
condition_sampling_policy
experimental
validity_status
```

The branch contract also requires sample/eval/paperpack runs to produce artifacts including `synthetic_data_manifest.json`, `normalization_params.json`, `normalization_params.sha256`, `generative_eval_metrics.csv`, `paperpack/reproducibility_statement.md`, table CSVs, appendix files, and figure-source CSVs. 

### 0.2 Validity contract

A synthetic result can be `benchmark-valid` only if it has:

```text
non-test source split
config hash
protocol hash
normalization artifact and hash
condition counts
leakage checks
metric status/reason reporting
paperpack traceability
```

If any item is missing, the result must not remain `benchmark-valid`. 

### 0.3 Feature specification priorities

The branch spec defines four user stories:

```text
P1: Govern benchmark validity
P2: strict train/sample/eval/paperpack evidence loop
P3: integrate frontier model families through existing factories
P4: produce paper-grade review artifacts
```

The spec requires that generative experiments cannot become benchmark-valid without config, protocol, normalization, condition, leakage, and metric evidence, and that paperpack must preserve source paths, table CSVs, figure sources, missing metrics, and reproducibility statements. 

### 0.4 Process artifact rule

Process artifacts should live under the active Speckit feature directory. The branch states that `.codex/` and `.claude/` are tool scratch or mirrors, while process, review, handoff, validation, and paper-readiness artifacts should remain under the active feature directory. 

### 0.5 Existing branch implementation areas

Review these branch-native paths:

```text
src/Pipeline_06_generative.py

src/model_factory/generative_model/
  condition_encoder.py
  film.py
  phm_cfm_mlp1d.py
  phm_unet1d.py
  phm_dit1d.py
  mamba1d_backbone.py

src/task_factory/task/generative/
  conditional_flow_matching.py
  generative_eval.py

src/task_factory/Components/generative/
  losses/
    flow_matching.py
    rectified_flow.py
    ddpm.py
    score_sde.py
  metrics/
    temporal.py
    spectral.py
    distribution.py
    diversity.py
    leakage.py
    tstr.py
  manifests/
    synthetic_data_manifest.py

configs/base/model/
  generative_cfm.yaml
  generative_unet1d.yaml
  generative_dit1d.yaml
  generative_ssm1d.yaml

configs/base/task/
  generative_cfm.yaml
  generative_rectified_flow.yaml
  generative_ddpm.yaml
  generative_score_sde.yaml
  generative_meanflow.yaml
  generative_drifting_flow.yaml
  generative_ot_nfm.yaml
  generative_transition_flow_matching.yaml

configs/demo/10_generative/
  dummy_generative_cfm.yaml
  dummy_generative_ddpm.yaml
  dummy_generative_rectified_flow.yaml
  dummy_generative_score_sde.yaml
  ...

configs/paper/phm_generative/
  six_dataset_benchmark_matrix.yaml
  benchmark_effect_matrix.yaml
  cfm_train_grid_seed0.yaml
  cfm_train_grid_seed1.yaml
  ...

scripts/
  paperpack_generative.py
  generative_benchmark_effect.py
  generative_submission_draft.py
  generative_sweep.py
```

---

# 1. Required context injection

Before reviewing, read or inspect:

```bash
cat specs/002-phm-genbench-frontier/spec.md
cat specs/002-phm-genbench-frontier/contracts/generative-benchmark-contract.md
cat specs/002-phm-genbench-frontier/data-model.md
cat specs/002-phm-genbench-frontier/checklists/benchmark-readiness.md

cat configs/demo/10_generative/dummy_generative_cfm.yaml
cat configs/base/model/generative_cfm.yaml
cat configs/base/task/generative_cfm.yaml

sed -n '1,240p' src/Pipeline_06_generative.py
sed -n '1,220p' src/task_factory/task/generative/conditional_flow_matching.py
sed -n '1,180p' src/task_factory/Components/generative/losses/flow_matching.py
sed -n '1,220p' src/task_factory/task/generative/generative_eval.py

find src/task_factory/Components/generative/metrics -maxdepth 1 -type f -name '*.py' -print
sed -n '1,260p' scripts/paperpack_generative.py
sed -n '1,360p' scripts/generative_benchmark_effect.py
sed -n '1,260p' scripts/generative_submission_draft.py
```

Do not evaluate from memory. Use branch code and branch contracts.

---

# 2. Review objective

Evaluate whether `Feature_factory-update` can support a full high-level PHM generative benchmark paper.

The review must cover:

```text
1. CLI and preflight contract
2. config completeness
3. pipeline completeness: train/sample/eval
4. model factory completeness
5. loss correctness
6. training-loop completeness
7. sampler completeness
8. normalization and domain evidence
9. metric-suite completeness
10. missing-metric status/reason handling
11. leakage and validity gates
12. paperpack completeness
13. benchmark-effect aggregation
14. multi-dataset / six-dataset readiness
15. submission-draft gating
16. branch-native goal and handoff hygiene
```

The review must output:

```text
paper_readiness_score
blocking issues
non-blocking issues
metric gap matrix
evidence gap matrix
paper artifact matrix
Codex-ready /goal backlog
```

---

# 3. Branch-specific scoring rubric

Score each axis from 0 to 5.

```text
0 = absent
1 = placeholder only
2 = partial implementation
3 = runnable smoke implementation
4 = benchmark-usable implementation
5 = paper-ready implementation with tests, manifests, configs, source paths, and clear failure gates
```

Compute:

```text
paper_readiness_score = 100 * sum(axis_scores) / (5 * number_of_axes)
```

Decision levels:

```text
0-39   NOT_PAPER_READY
40-59  PROTOTYPE_ONLY
60-74  WORKSHOP_LEVEL
75-89  CONFERENCE_CANDIDATE
90-100 TOP_TIER_CANDIDATE
```

A score above 75 is invalid if any blocking issue exists.

---

# 4. Score axes

## A. CLI and preflight

Check:

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
```

Required:

```text
- preflight parses config
- validates pipeline whitelist
- validates five-block config
- validates schema checks
- exits before trainer execution
- malformed YAML fails fast
- unsupported pipeline fails fast
```

Blocking if:

```text
- preflight starts training
- YAML parse failure silently falls back
- invalid pipeline silently uses another pipeline
```

## B. Config completeness

Check:

```text
configs/demo/10_generative/*.yaml
configs/base/model/generative_*.yaml
configs/base/task/generative_*.yaml
configs/paper/phm_generative/*.yaml
```

Required:

```text
pipeline: Pipeline_06_generative
task.type: generative
task.generative.mode: train | sample | eval
task.generative.method_family
task.generative.condition_sampling_policy
task.generative.validity_status
data.normalization
model.type: generative_model
model.name
trainer settings
```

The existing dummy CFM config already uses `Pipeline_06_generative`, base model/task configs, `standardization`, and `task.generative.mode: train`.  The base CFM task currently includes `source_split`, `eval_split`, `domain_map_path`, `num_steps`, `num_samples`, `length`, `validity_status`, and `allow_untrained_smoke`.  The base CFM model uses `type: generative_model`, `name: phm_cfm_mlp1d`, input channels, hidden dimension, condition dimension, class count, and domain count. 

Score 5 only if configs cover:

```text
train config
sample config
eval config
paper matrix config
multi-seed config
at least one non-dummy paper matrix with required dataset coverage
```

## C. Pipeline completeness

Check `src/Pipeline_06_generative.py`.

Required:

```text
mode=train
mode=sample
mode=eval
normalization artifacts
condition policies
condition counts
synthetic sample payload
synthetic manifest
leakage checks
efficiency fields
metric CSV
```

The current pipeline loads five config sections, builds data/model/task through existing factories, supports condition policies including first-metadata repeated, grid, train-distribution, and explicit conditions, and writes normalization artifacts from the train split.  It also samples to `synthetic/samples.pt`, records `fault_label`, `domain_id`, condition policy, condition counts, sampler metadata, sampling wall-clock, NFE, samples/s, parameter count, and peak memory.  Eval mode writes `generative_eval_metrics.csv` and records metric compute time, parameter count, NFE, eval split status, and sample counts. 

Blocking if:

```text
- train/sample/eval do not share config-first path
- sample does not write synthetic_data_manifest.json
- eval does not write generative_eval_metrics.csv
- condition sampling can use non-train/test metadata without guard
```

## D. ModelFactory completeness

Check:

```text
src/model_factory/generative_model/
src/model_factory/model_registry.csv
```

Required:

```text
- models live under src/model_factory/generative_model
- no src/phm_factory runtime
- model output shape [N,C,L]
- condition input only fault_label + domain_id
- load/rpm are represented through domain_id mapping, not direct model input
- smoke forward exists or is indirectly covered
- model cards/README explain prediction_type
```

The CFM MLP model uses `ConditionEncoder`, FiLM, Conv1D projection, and returns `[N,C,L]` velocity predictions after validating rank and channel count.  The condition encoder explicitly encodes V0 conditions `fault_label` and `domain_id`, using embeddings and scalar time `t`.  The Mamba/SSM placeholder is explicitly stateless and avoids mandatory CUDA dependencies while preserving `[N,C,L]` contract and no sampler-managed hidden cache. 

Score 5 only if every promoted model has:

```text
registry row
config
smoke forward test
model card or README
shape contract
prediction type
dependency guard
```

## E. Loss correctness

Check:

```text
src/task_factory/Components/generative/losses/
```

Required CFM:

```text
x1: [N,C,L]
z: [N,C,L]
t: [N] or [N,1,1]
pred_velocity: [N,C,L]
target_velocity = x1 - z
loss = MSE(pred_velocity, target_velocity)
NaN/Inf guard
shape guard
```

The current CFM loss samples `t`, builds `x_t = (1-t)z + t*x1`, uses `target_velocity = x1 - z`, checks finite tensors, and returns `loss` plus `mse_v`. 

Required DDPM:

```text
target = epsilon
prediction = epsilon_theta
loss = MSE(epsilon_theta, epsilon)
```

Required Score SDE:

```text
target = conditional score
prediction = score_theta
loss = denoising score matching
```

Blocking if:

```text
- CFM target differs from x1 - z
- DDPM target differs from epsilon
- Score SDE is implemented as velocity regression
- losses placed outside src/task_factory/Components/generative/losses
- FFT/envelope spectrum enters V0 training loss
```

## F. Task and training completeness

Check:

```text
src/task_factory/task/generative/conditional_flow_matching.py
```

Required:

```text
LightningModule
training_step
validation_step
test_step
optimizer
loss logging
condition extraction from batch/metadata
file_id traceability
sample method
manifest writer bridge
```

The current CFM task wraps a network with Lightning, extracts `fault_label` and `domain_id` from batch or metadata, samples Gaussian noise and `t`, computes CFM loss, logs train/val/test losses, configures Adam/AdamW, samples through Euler ODE, and delegates manifest writing.  It also builds synthetic manifests with model name, loss ID, checkpoint path, source split, domain map hash, normalization evidence, seed, shape, config/protocol hashes, dependency lock hash, leakage checks, condition policy, and condition counts. 

Score 5 only if training evidence includes:

```text
one-batch forward/loss test
one optimizer-step test
one epoch smoke run
checkpoint existence test
finite-loss guard over first K steps
failed-run evidence behavior
```

## G. Sampler completeness

Check:

```text
src/task_factory/Components/generative/samplers/
```

Required for flow models:

```text
Euler ODE sampler
num_steps / NFE recorded
stateless model call per step
NaN/Inf guard
output shape [N,C,L]
```

Required for DDPM:

```text
scheduler
noise schedule recorded
sampler
NFE/step count recorded
```

Blocking if:

```text
- sampler can silently return NaN
- sampler does not record NFE
- Mamba hidden cache is carried across diffusion/flow steps
```

## H. Normalization and domain evidence

Check:

```text
src/data_factory/data_utils.py
src/data_factory/ID/domain_map.py
configs/domain_maps/dummy_domain_map.csv
synthetic_data_manifest.json
normalization_params.json
normalization_params.sha256
```

Required:

```text
- standardization or robust_scaler
- per_channel scope
- source_split=train
- params artifact
- params hash
- domain_map path
- domain_map hash
- condition_counts keyed by fault/domain
```

The branch data model requires normalization evidence with artifact path, artifact hash, method, per-channel scope, and train/source split; benchmark-valid runs require the artifact and hash, and statistics cannot be computed from validation or test splits. 

Blocking if:

```text
- normalization stats computed from val/test
- domain_map hash missing
- load/rpm used as direct model condition
- benchmark-valid without normalization_params hash
```

## I. Metric-suite completeness

Review all metric families.

### I.1 Temporal metrics

Current temporal metrics include:

```text
mean abs error
std abs error
skew abs error
kurtosis abs error
L1/L2
RMS error
crest factor error
zero crossing rate error
autocorr RMSE
cross-channel correlation error
status codes
```

These are implemented for `[N,C,L]` tensors and return NaN metrics on invalid shapes. 

Paper-ready additions to check:

```text
per-domain aggregation
per-fault aggregation
confidence interval
window-length sensitivity
impulse interval / peak interval statistics
```

### I.2 Spectral metrics

Current spectral metrics include:

```text
FFT L1
log FFT L1
PSD L2
band energy error
spectral angle
fault-frequency preservation proxy
status codes
```

They run under `torch.no_grad()` and are eval-only FFT amplitude metrics for `[N,C,L]`. 

Paper-ready additions to check:

```text
sampling_rate-aware frequency axes
Welch PSD option
envelope spectrum / Hilbert peak metrics
fault-characteristic-frequency table per dataset/domain
inverse-normalization support for physical units
```

Blocking if:

```text
- spectral metrics are absent
- spectral metrics become V0 training loss
- no status/reason when not computable
```

### I.3 Distribution metrics

Current distribution metrics include:

```text
mean distance
variance distance
MMD-RBF
sliced Wasserstein
energy distance
```

They use flattened `[N,C,L]` tensors and lightweight torch operations. 

Paper-ready additions to check:

```text
feature-space FID-like PHM metric
fixed feature extractor provenance
discriminative real-vs-synthetic score
per-domain / per-class aggregation
```

### I.4 Leakage metrics

Current leakage metrics include:

```text
nearest-neighbor L2
duplicate rate
nearest-neighbor pass flag
```

They compare synthetic windows to real windows by nearest neighbor distance. 

Paper-ready additions to check:

```text
real-train vs synthetic
real-test vs synthetic distance ratio
train-member AUC if feasible
near-duplicate threshold sensitivity
split provenance proof
```

Blocking if:

```text
- benchmark-valid rows lack leakage checks
- synthetic source split can be test or target_test
- nearest-neighbor leakage result is missing without reason
```

### I.5 Diversity metrics

Current diversity metrics include:

```text
PRDC precision
PRDC recall
PRDC density
PRDC coverage
intra-class variance ratio
```

They support label-conditioned intra-class variance when labels are available. 

Paper-ready additions to check:

```text
condition consistency classifier
coverage by fault_label/domain_id
mode-collapse alert thresholds
```

### I.6 Utility metrics

Current TSTR/TRTS uses a lightweight nearest-centroid probe. It returns TSTR and TRTS accuracy when labels are available, otherwise a placeholder. 

Paper-ready additions required:

```text
real downstream model protocol
real-only vs real+synthetic augmentation gain
few-shot gain
cross-domain transfer gain
RUL/anomaly/fault classification task-specific adapters
macro-F1 / AUROC / AUPRC where applicable
confidence intervals over seeds
```

Blocking for top-tier paper if:

```text
- no utility metric beyond nearest-centroid smoke proxy
- no real-only baseline
- no real+synthetic augmentation comparison
- no multi-seed confidence reporting
```

## J. Missing metric status/reason completeness

The current `generative_eval.py` imports temporal, spectral, distribution, diversity, leakage, and TSTR metrics, computes them, adds per-fault and per-domain group metrics, and annotates status/reason fields for missing or non-finite values. 

Required review checks:

```text
- every metric has value/status/reason
- non-computable metrics are not silently dropped
- paperpack preserves missing reasons
- benchmark-effect aggregation preserves missing reasons
```

Blocking if:

```text
- missing values are silently omitted
- NaN is accepted as valid score
- missing reasons do not propagate to paperpack
```

## K. Paperpack completeness

Check:

```bash
python -m scripts.paperpack_generative --run_dir <run_dir>
```

The current paperpack script reads `generative_eval_metrics.csv`, groups quality, utility, efficiency, and leakage prefixes, aggregates mean/std/n, preserves source paths, writes missing-metric reports, manifest completeness tables, run index, table CSVs, and figure sources.  It writes `reproducibility_statement.md`, `table_quality.csv`, `table_utility.csv`, `table_efficiency.csv`, `table_leakage.csv`, mean/std tables, ablation table, `run_index.csv`, `manifest_completeness.csv`, `missing_metrics.csv/md`, and figure-source CSVs for spectral, temporal, metric barplots, dataset-method heatmaps, and missing-metric audit. 

Paper-ready checks:

```text
- figure_sources contain enough raw data for actual plotting, not only metric summaries
- tables are manuscript-ready
- seed aggregation is explicit
- source paths preserved for every row
- manifest completeness table aligns with validity contract
```

## L. Benchmark-effect aggregation completeness

Check:

```bash
python -m scripts.generative_benchmark_effect --dry-run --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml
python -m scripts.generative_benchmark_effect --from-runs <run_dirs> --matrix <matrix>
```

The current script has separate dry-run and from-runs modes, builds train/sample/eval/paperpack command plans, and keeps long training out of CI while preserving a testable benchmark contract.  It generates stage commands for train, sample, eval, and paperpack with overrides for seed, output dir, synthetic dataset ID, checkpoint path, generated sample path, and condition sampling policy.  It aggregates metric records with dataset/method/seed, status, manifest path, metric source path, computes mean/std/rank/delta vs baseline, writes summary CSV, missing metrics, benchmark effect report, and manifest. 

Paper-ready checks:

```text
- observed_configured_dataset_count used for six-dataset claim
- configured_dataset_count alone cannot satisfy paper claim
- missing datasets and unexpected datasets create input_gaps
- every summary row has metric_source_paths and manifest_paths
- benchmark_status downgraded to exploratory unless all contributing manifests are benchmark-valid
```

## M. Submission draft gating

Check:

```bash
python -m scripts.generative_submission_draft \
  --summary <benchmark_effect_summary.csv> \
  --manifest <benchmark_effect_manifest.json> \
  --output <paper.md> \
  --require-submission-ready
```

The current submission draft generator is conservative: it writes submission-ready status only when the input summary covers the required dataset count and all contributing rows are benchmark-valid, and it does not invent missing results.  It blocks readiness if input gaps, missing configured datasets, unexpected datasets, insufficient observed configured dataset count, missing metric source paths, missing manifest paths, missing quality evidence, missing utility evidence, or non-benchmark-valid rows are present.  The generated draft records evidence gaps and submission readiness sidecars, and `--require-submission-ready` exits nonzero if readiness gates fail. 

Score 5 only if:

```text
- markdown draft can be generated from real evidence
- readiness sidecars are generated
- failure mode is nonzero when submission-ready is required
- no placeholder tokens are permitted
```

---

# 5. Blocking issue rules

Mark the review as `BLOCKING` if any condition is true.

```text
1. `python main.py --config <yaml>` path is broken.
2. `Pipeline_06_generative` cannot execute train/sample/eval modes.
3. Preflight starts trainer execution.
4. Invalid YAML or invalid pipeline silently falls back.
5. Runtime creates or depends on `src/phm_factory/` for generative work.
6. Generative losses are not under `src/task_factory/Components/generative/losses/`.
7. CFM target is not exactly `x1 - z`.
8. DDPM target is not epsilon.
9. Score SDE target is not score / DSM-compatible target.
10. Model condition uses direct load/rpm instead of `fault_label` + `domain_id`.
11. Synthetic source split can be `test` or `target_test`.
12. Benchmark-valid manifests lack config hash, protocol hash, normalization artifact/hash, condition counts, leakage checks, or metric status/reason fields.
13. Spectral metrics are absent for vibration generation.
14. FFT/envelope metrics are used as V0 training loss.
15. Missing metrics are silently dropped instead of status/reason reported.
16. Paperpack rows do not preserve metric source paths.
17. Benchmark-effect summary rows do not preserve manifest paths.
18. Submission draft can be marked ready without observed configured dataset coverage.
19. Six-dataset claim uses configured dataset count rather than observed configured dataset count.
20. Utility evidence is absent or only non-labeled placeholder while paper claims downstream benefit.
```

---

# 6. Quantitative metric review matrix

Return a table using this schema:

```text
Metric family | Current implementation | Paper-level requirement | Gap | Blocking? | Recommended goal
```

Use these families:

```text
temporal_fidelity
spectral_fidelity
distribution_fidelity
diversity_coverage
leakage_memorization
utility_TSTR_TRTS
augmentation_gain
cross_domain_transfer
few_shot_gain
efficiency
robustness_seed_statistics
condition_coverage
missing_metric_reasons
benchmark_effect_aggregation
```

Recommended branch-specific gap interpretation:

```text
temporal_fidelity:
  current: good smoke-to-benchmark baseline
  likely gap: confidence intervals, impulse interval stats, paper grouping

spectral_fidelity:
  current: FFT/PSD/band/spectral-angle proxy
  likely gap: sampling_rate-aware frequency axis, envelope spectrum, fault-frequency table

distribution_fidelity:
  current: MMD/SWD/energy lightweight implementation
  likely gap: PHM feature-FID and discriminative score

diversity:
  current: PRDC and intra-class variance ratio
  likely gap: condition consistency and mode-collapse thresholds

leakage:
  current: train nearest-neighbor proxy
  likely gap: train-vs-test distance ratio and threshold sensitivity

utility:
  current: nearest-centroid TSTR/TRTS proxy
  likely gap: true downstream classifier/RUL/anomaly protocol, real-only vs real+synthetic augmentation gain

efficiency:
  current: parameter_count, sampling_nfe, wall-clock, samples/s, memory fields
  likely gap: training GPU memory and full seed/method aggregation

robustness:
  current: matrix/seeds planned
  likely gap: actual multi-seed evidence and CI/non-CI separation

condition_coverage:
  current: condition_counts and condition sampling policies
  likely gap: condition-grid coverage assertions in tests
```

---

# 7. Required output format

Return exactly these sections.

## 1. Executive decision

```xml
<REVIEW_DECISION>APPROVE | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<PAPER_READINESS_SCORE>0-100</PAPER_READINESS_SCORE>
<PAPER_READINESS_LEVEL>NOT_PAPER_READY | PROTOTYPE_ONLY | WORKSHOP_LEVEL | CONFERENCE_CANDIDATE | TOP_TIER_CANDIDATE</PAPER_READINESS_LEVEL>
<BRANCH>Feature_factory-update</BRANCH>
```

## 2. Scorecard

Return a Markdown table:

```text
Axis | Score 0-5 | Evidence paths | Evidence summary | Missing pieces | Blocking?
```

Axes:

```text
A CLI/preflight
B config completeness
C pipeline train/sample/eval
D model_factory/generative_model
E loss correctness
F task/training completeness
G sampler completeness
H normalization/domain evidence
I temporal metrics
J spectral metrics
K distribution metrics
L diversity metrics
M leakage metrics
N utility metrics
O missing-metric status/reason
P paperpack
Q benchmark-effect aggregation
R submission-draft gating
S tests/validation coverage
T branch goal/handoff hygiene
```

## 3. Blocking issues

Each item must include:

```text
- issue
- why it blocks high-level paper claims
- affected files
- required fix
- validation command
```

## 4. Metric gap matrix

Use the matrix specified above.

## 5. Paper evidence matrix

Return a table:

```text
Paper artifact | Required evidence | Exists? | Source path | Missing action | Benchmark-valid gate
```

Artifacts:

```text
Table 1 dataset/domain summary
Table 2 model-family summary
Table 3 fidelity metrics
Table 4 utility metrics
Table 5 leakage/validity
Table 6 efficiency
Table 7 ablation
Figure 1 pipeline architecture
Figure 2 model/loss taxonomy
Figure 3 signal gallery
Figure 4 temporal/spectral overlays
Figure 5 distribution/diversity plots
Figure 6 utility plots
Figure 7 leakage plots
Figure 8 efficiency/quality trade-off
Reproducibility statement
Missing metric appendix
Benchmark effect manifest
Submission readiness report
```

## 6. Branch-native validation commands

Return commands that should be run now:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override environment.iterations=1 \
  --override trainer.num_epochs=1 \
  --override trainer.gpus=0 \
  --override trainer.device=cpu

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=sample \
  --override task.generative.allow_untrained_smoke=true \
  --override task.generative.condition_sampling_policy=grid \
  --override 'task.generative.condition_grid.fault_label=[0,1]' \
  --override 'task.generative.condition_grid.domain_id=[0,1]' \
  --override task.generative.condition_grid.samples_per_condition=1

python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=eval \
  --override task.generative.generated_path=<path-to-synthetic-samples.pt>

python -m scripts.paperpack_generative --run_dir <run-dir>

python -m scripts.generative_benchmark_effect \
  --dry-run \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --allow-missing-data

python -m scripts.generative_submission_draft \
  --summary <benchmark_effect_summary.csv> \
  --manifest <benchmark_effect_manifest.json> \
  --output <paper.md> \
  --require-submission-ready
```

If a command cannot be run because paths are unavailable, report:

```text
status: not_run
reason: <specific missing path or dependency>
```

Do not silently skip.

## 7. Codex-ready `/goal` backlog

Generate small PR-sized goals only. Each goal must use this format:

```md
/goal

## Goal ID
GOAL-FFU-PAPER-XXX

## Objective
...

## Why
...

## Current facts
...

## Scope
...

## Out of scope
...

## Required behavior
...

## Acceptance criteria
...

## Validation commands
...
```

Prioritize these if gaps exist:

```text
GOAL-FFU-PAPER-000
Add branch-native paper-readiness review prompt v2 under specs/002-phm-genbench-frontier/reviews/

GOAL-FFU-PAPER-001
Add metric contract schema for generative_eval_metrics.csv and paperpack tables

GOAL-FFU-PAPER-002
Add sampling_rate-aware spectral metrics and envelope-spectrum missing-reason handling

GOAL-FFU-PAPER-003
Add true downstream utility protocol skeleton: real-only vs real+synthetic augmentation gain

GOAL-FFU-PAPER-004
Add train-vs-test leakage ratio and threshold sensitivity metrics

GOAL-FFU-PAPER-005
Add condition-grid coverage test for fault_label/domain_id manifest counts

GOAL-FFU-PAPER-006
Add multi-seed aggregation test for paperpack mean/std tables

GOAL-FFU-PAPER-007
Add benchmark-effect manifest gate test for observed_configured_dataset_count

GOAL-FFU-PAPER-008
Add submission draft negative test: require-submission-ready fails on exploratory rows

GOAL-FFU-PAPER-009
Add figure-source data contract for temporal/spectral overlays and signal gallery

GOAL-FFU-PAPER-010
Add PHM feature-FID or discriminative real-vs-synthetic score adapter
```

## 8. Final recommendation

State whether the branch is ready for:

```text
internal experiment only
workshop paper
full conference paper
Nature-family / high-level journal submission
```

Then list the top 5 actions needed to reach the next level.

````

---

# 2. 给 Codex 的第一条具体 `/goal`

这条是把上面的 prompt 落进 `Feature_factory-update` 分支，而不是实现新指标。

```md
/goal

## Goal ID
GOAL-FFU-PAPER-000

## Objective
Add a branch-native PHM-GenBench paper-readiness and quantitative metric review prompt v2 under the active Speckit feature directory.

## Why
`Feature_factory-update` already contains `Pipeline_06_generative`, generative model factory modules, TaskFactory generative losses/metrics, paperpack generation, benchmark-effect aggregation, and submission-draft gating. The review prompt must now evaluate the actual branch state rather than the earlier pre-implementation plan.

## Current facts
- The maintained CLI contract is `python main.py --config <yaml> [--override key=value ...]`.
- Generative configs use `pipeline: Pipeline_06_generative`.
- The active feature directory is `specs/002-phm-genbench-frontier/`.
- Generative models live in `src/model_factory/generative_model/`.
- Generative losses live in `src/task_factory/Components/generative/losses/`.
- Generative metrics live in `src/task_factory/Components/generative/metrics/`.
- Paperpack tooling exists in `scripts/paperpack_generative.py`.
- Benchmark-effect tooling exists in `scripts/generative_benchmark_effect.py`.
- Submission draft gating exists in `scripts/generative_submission_draft.py`.

## Scope
Allowed to add:
- `specs/002-phm-genbench-frontier/reviews/phm-genbench-paper-readiness-review-prompt-v2.md`
- `specs/002-phm-genbench-frontier/reviews/README.md` only if it needs an index update

Allowed to modify:
- `specs/002-phm-genbench-frontier/checklists/benchmark-readiness.md` only to link the new review prompt

## Out of scope
- Do not implement new metrics.
- Do not modify `src/Pipeline_06_generative.py`.
- Do not modify model, loss, sampler, trainer, or paperpack runtime code.
- Do not create `src/phm_factory/`.
- Do not change configs.
- Do not run long training.

## Required behavior
The review prompt must evaluate:
1. CLI and preflight
2. generative config completeness
3. train/sample/eval pipeline
4. model_factory/generative_model integration
5. TaskFactory generative loss correctness
6. training-loop completeness
7. sampler completeness
8. normalization and domain evidence
9. temporal/spectral/distribution/diversity/leakage/utility metrics
10. missing metric status/reason propagation
11. synthetic manifest validity gates
12. paperpack tables, figure sources, appendices, and reproducibility statement
13. benchmark-effect aggregation and six-dataset evidence
14. submission-draft readiness gating
15. Codex-ready follow-up goal backlog

The prompt must explicitly check:
- CFM target is `x1 - z`
- DDPM target is epsilon
- Score SDE target is score/DSM-compatible
- model condition is `fault_label` + `domain_id`
- load/rpm are not direct model conditions
- FFT metrics are eval-only in V0
- benchmark-valid requires manifest, config hash, protocol hash, normalization artifact/hash, condition counts, leakage checks, metric status/reasons, and paperpack traceability

## Acceptance criteria
- The new review prompt is self-contained and directly usable by Claude Code.
- It references branch-native paths and not outdated planned paths.
- It scores paper readiness from 0 to 100.
- It emits blocking issues, scorecard, metric gap matrix, paper evidence matrix, validation commands, and Codex-ready follow-up goals.
- It preserves the active Speckit feature directory as the canonical process artifact location.

## Validation commands
```bash
python -m scripts.validate_docs
````

## Failure handling

If `scripts.validate_docs` fails, report the exact broken document path and line if available. Do not silently skip doc validation.

## Review checklist

* Does the prompt inspect actual `Feature_factory-update` files?
* Does it include paper-level quantitative metrics?
* Does it distinguish smoke metrics from paper-grade metrics?
* Does it preserve factory-first integration?
* Does it avoid asking for a large refactor?
* Does it convert gaps into small Codex-ready `/goal`s?

````

---

# 3. 更新后的核心判断

`Feature_factory-update` 分支已经不是“从零开始”的状态。它已经有：

```text
Pipeline_06_generative
CFM train/sample/eval loop
generative model factory
CFM loss
DDPM/RF/ScoreSDE loss files
temporal/spectral/distribution/diversity/leakage/TSTR metrics
synthetic manifest writing
normalization artifacts
condition sampling policies
paperpack
benchmark-effect aggregation
submission draft gate
paper configs
````

所以新的评审 prompt 应该从 **“是否规划正确”** 升级为 **“是否足以支撑论文主张”**。当前最可能的论文级缺口不再是“有没有基础代码”，而是：

```text
1. utility 仍偏 smoke/proxy，需要 real-only vs real+synthetic augmentation protocol
2. spectral metrics 需要 sampling_rate-aware、envelope spectrum、fault-frequency evidence
3. leakage 需要 train-vs-test distance ratio 和 threshold sensitivity
4. paperpack 需要更强 figure-source raw data，而不仅是 metric summary
5. six-dataset / multi-seed evidence必须由 observed_configured_dataset_count 和真实 metric_source_paths/manifest_paths 支撑
```

这版 prompt 的目标就是让 Claude / Codex 每次审查都能把这些差距转成小 `/goal`，最终让仓库支撑一篇完整、高水平、可审计的 PHM 生成模型 benchmark 论文。

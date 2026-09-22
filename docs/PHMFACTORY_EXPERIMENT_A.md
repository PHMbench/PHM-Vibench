# Goal: first real-data execution batch for the PHMFactory paper

## Core claim and essential gap

Paper: `AI4Engineering-L/phmfactory`, active source `paper/tex/main.tex`. Unique implementation: `PHMbench/PHM-Vibench`.

The framework is intended to execute declared PHM experiments through reusable data, model, task and trainer responsibilities. The paper has a method specification, not demonstrated automation superiority. Obtain current-source real-data execution, selected-state prediction checks and complete metrics before a broader comparison. Without these, valid task execution is unestablished.

This batch uses the existing MFPT mean-linear reference and one fixed TimesNet classification replacement. It is not a hyperparameter search or a claim that either configuration is a strong predictive baseline. Do not build a new task, replay unrelated model PRs, write Results for unexecuted studies, or rerun until scores improve. The existing `paper/project/` is an unrelated historical source map; use this existing `docs/` location rather than recreating a paper tree.

## Inputs and protocol

Source inspected: `16dfe24343554c8ce086e09f1e453edb79a8efa0`. Inspect current dev, active PRs and local changes before execution. Record one source revision for each matched run. Reuse valid previous outputs only when their complete numerical protocol and relevant source remain compatible; explain that decision. Existing TimesNet integration/fidelity evidence can be reused within its stated scope, but is not real-MFPT evidence.

Verified inputs:

- `scripts/prepare_mfpt_baseline.py`
- `configs/baselines/01_mfpt/mfpt_global_average_linear.yaml`
- `configs/base/model/global_average_linear.yaml`
- `configs/experiments/model_integration/timesnet_dummy.yaml`
- `src/model_factory/Baseline/GlobalAverageLinear.py`
- `src/model_factory/CNN/TimesNet.py`
- `src/task_factory/Default_task.py`
- `src/task_factory/Components/metrics.py`
- `src/runtime/classification.py`
- `src/utils/run_summary.py`

The preparation script selects the exact 20-file subset from the MathWorks MFPT provider. Online preparation pins `d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d`. Offline preparation records `user-provided-checkout`; do not rewrite that as a verified pin without provenance. Respect the dataset-specific licence and do not commit raw data.

Keep provider-source partitions: 14 training-source files and 6 test-source files. `Domain_id=0/1` encodes those partitions, not a physical domain shift. Training/validation are grouped by `File`, split seed 17 and requested fractions 0.75/0.25 within the training source. Check actual memberships, class support and physical metadata before fitting. File disjointness does not establish independent assets.

Retain the YAML settings: `DG/classification`, CE, windows 2048, 16 evenly spaced windows per file, window seed 17, float32, batch 32, no normalization/noise, Adam, learning rate 0.001, weight decay 0.0001, CPU/one device, maximum 5 epochs, validation-loss selection and patience 3. Report accuracy and macro-F1 on the complete test population.

Use training seeds 17, 18, 19, fixed by the existing configuration. These are repeated training conditional on one partition, not independent machines. Reuse selected checkpoints for prediction validation; do not retrain to export. Report every seed and per-record predictions. Do not treat correlated windows as independent units or add repetitions after inspecting favorable/unfavorable scores.

Local absolute paths, provider availability and CPU/RAM are unknown to this design session. Set them explicitly. No GPU, paid API or LLM is required. Record the available time/memory budget before running; preserve the fixed epoch/seed budget. An exhausted resource cap produces a retained incomplete run, not a smaller substituted experiment.

## Execution

### 0. Source and environment

```bash
: "${PHMFACTORY_ROOT:?Set the actual implementation checkout}"
cd "$PHMFACTORY_ROOT"
git status --short
git fetch origin --prune
git rev-parse origin/dev
# Inspect active PRs with the available GitHub client.
# With a clean working tree, create a topic branch from current origin/dev:
git switch -c experiment/phmfactory-realdata-pair origin/dev
python -m pip install -e .
phmfactory doctor
```

Do not discard local changes. Record source/dependency versions and hardware. Installation is not experiment success.

### 1. Prepare or reuse data

Set `MFPT_ROOT` to the absolute prepared data root. When correct data already exist, inspect and reuse them; preparation deliberately refuses to overwrite. Inspect metadata, raw paths, sample rates, channels, labels and provider provenance. Choose exactly one preparation route when new data are needed:

```bash
# Online preparation of a new root:
python -m scripts.prepare_mfpt_baseline --output "$MFPT_ROOT"
```

```bash
# Offline preparation of a new root from the actual provider checkout:
python -m scripts.prepare_mfpt_baseline \
  --provider-checkout "$MFPT_PROVIDER_ROOT" --output "$MFPT_ROOT"
```

These are alternatives, not a fallback chain. Missing data must not be replaced, synthesized, relabeled or selectively dropped. If the population cannot be established, preserve the reason and stop data-dependent execution.

### 2. Existing mean-linear reference

Overrides change local paths only. Give the invocation a new output parent and its own split record; do not share writable split paths across concurrent runs.

```bash
set -euo pipefail
: "${MFPT_ROOT:?Set the absolute prepared MFPT data root}"
: "${RUN_PARENT:?Set a new absolute output parent}"
mkdir "$RUN_PARENT"
CFG=configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
COMMON=(--config "$CFG"
  --override "data.data_dir=$MFPT_ROOT"
  --override "environment.output_dir=$RUN_PARENT/results"
  --override "data.split.manifest_path=$RUN_PARENT/split_manifest.json")
phmfactory preflight "${COMMON[@]}" 2>&1 | tee "$RUN_PARENT/preflight.log"
phmfactory "${COMMON[@]}" 2>&1 | tee "$RUN_PARENT/execution.log"
```

Use actual returned `result_dir`, `best_checkpoint`, `test_metrics`, `run_summary` and primary metrics. Do not guess result paths or select a different run by scanning output directories. Check all three iterations/seeds. A failed preflight stops execution; training-only completion is not evaluated completion.

### 3. Selected-state prediction and independent metric checks

Reuse retained predictions/window identities if complete. No universal standalone prediction-export command is asserted here. If export is missing, add only the necessary experiment-specific export/check helper under existing `scripts/paper/`, reusing the current data factory, task and selected checkpoint. Verify and record its actual invocation before running it. Do not create a second loader, trainer or result authority.

The independent mean-linear reference is:

```text
features[b,c] = mean over t of float32 x[b,t,c]
reference_logits = features @ classifier.weight.T + classifier.bias
```

Read weight/bias from the exact selected state after strict restore. Compare the native and reference logits on identical ordered batches. Predeclare CPU float32 tolerances `atol=1e-6, rtol=1e-5`; retain maximum differences and class changes. Investigate a tolerance failure rather than increasing tolerance. Independently recompute accuracy and macro-F1 using the full class list and estimator's zero-division convention; compare each dataset-specific reported metric at absolute tolerance `1e-6`. Retain population counts and record/window identities. This is a validity check, not predictive superiority.

### 4. One fixed compatible TimesNet replacement

The current model supports fixed-length fully observed float32 classification. Use `seq_len=2048`, `input_dim=1`, `num_classes=3`, `d_model=16`, `d_ff=32`, `e_layers=1`, `dropout=0`, `top_k=2`, `num_kernels=2`. Check that the selected metadata really has the declared three-class ontology; do not force labels to match. These bounded settings use the existing example's structure and the MFPT window length, not a competitively tuned configuration.

```bash
set -euo pipefail
: "${TIMESNET_PARENT:?Set a separate new absolute output parent}"
mkdir "$TIMESNET_PARENT"
TN=(--config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
  --override "data.data_dir=$MFPT_ROOT"
  --override "environment.output_dir=$TIMESNET_PARENT/results"
  --override "data.split.manifest_path=$TIMESNET_PARENT/split_manifest.json"
  --override model.type=CNN --override model.name=TimesNet
  --override model.seq_len=2048 --override model.input_dim=1
  --override model.num_classes=3 --override model.d_model=16
  --override model.d_ff=32 --override model.e_layers=1
  --override model.dropout=0.0 --override model.top_k=2
  --override model.num_kernels=2)
phmfactory preflight "${TN[@]}" 2>&1 | tee "$TIMESNET_PARENT/preflight.log"
phmfactory "${TN[@]}" 2>&1 | tee "$TIMESNET_PARENT/execution.log"
```

Require the same actual split, windows and labels as the linear case. Keep the same seeds, optimizer/epoch budget and selection policy. TimesNet period selection is batch dependent, so replay must preserve batch membership/order; arbitrary rebatching need not preserve predictions. Its source discloses the upstream DC-tie correction. Do not claim forecast, missing-data or pretrained capabilities.

Reuse step 3's checkpoint export and independent metric check, replacing only the model-specific linear-formula oracle with existing TimesNet fidelity evidence and native replay. Report both models even if TimesNet is less accurate or fails within the resource budget. Accuracy differences are model results, not effects of decoupling.

### Next batch, not a fictional command in this one

A matched direct-assembly comparison and defensible closest-platform comparison remain essential to the usefulness claim C3. The control must reuse identical numerical components but not merely call the canonical Pipeline under another name. Its exact harness and invocation are not established in this snapshot. Return steps 1–4 artifacts for the next design/implementation handoff. Do not invent a control, launch unbounded tuning, or infer architecture superiority from two-model scores.

## Artifacts and validation

Reuse canonical configuration, explicit overrides, versions, seeds, split membership, selected states, complete metrics, result locations, logs and failure records. Prediction evidence identifies record/window, target, prediction/logits and selected-state source. Extra check CSVs belong alongside the original run; do not overwrite results or introduce a parallel tracking system.

Validity requires the declared population and labels, permitted preprocessing, correct task/model/state, complete prediction coverage, finite declared metrics, independent agreement, and honest failure accounting. Imports, Dummy smoke, finite accuracy alone and catalogue entries are insufficient.

Return results to manuscript Sections III–V and the existing `paper/experiments/evidence_matrix.md`. This batch can establish a real reference, one bounded compatible replacement and selected-state evaluation. It cannot establish competitive ranking, architecture efficiency, physical domain generalization or diagnosis-to-prognosis automation.

## Failure and sync

Separate valid negative results, invalid protocols, infrastructure failures and not-executed work. Keep failed/interrupted runs. Environment fixes are allowed only with unchanged scientific protocol. Changes to data, split, model, target, objective, selection or estimator require explicit design handoff and cannot be pooled with old results.

Commit only necessary experiment code/configuration and small shareable evidence to the implementation topic branch. Large artifacts follow existing result storage with accessible references; do not force them into Git. No hash/receipt/ledger layer is required. Open a PR to dev, run affected checks and merge validated work under normal rules; never modify main, force-push or merge unrelated changes.

After implementation synchronization, update the existing paper evidence matrix and research state with actual artifact locations. Execution instructions remain unique here. Stop after the batch and return evidence or concrete blockers, without rerunning until outcomes look favorable.

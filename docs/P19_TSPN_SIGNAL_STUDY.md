# Goal — P19 native signal-program correction: essential evidence

## Core claim, gap and reuse

Paper: `AI4Engineering-L/P-19-Deterministic-Representation-Gain-tpami`, formal source `paper/tex/main.tex`. Unique owner of new execution: PHMFactory (`PHMbench/PHM-Vibench`). Agent Benchmark is not a runtime dependency; its completed E1–E6/G6 evidence and the earlier correction code are historical/migration references only.

**Core claim:** a named signal program can improve a competent frozen TSPN beyond constant logit adjustments and alternative native-feature readouts. Without the native qualification and matched real comparison this practical contribution is unestablished. The target remains IEEE TPAMI; software checks and standard algebra do not supply the missing empirical value.

**Current gap:** at inspected dev `bc21a9ea65d56c0083d837f16e882a6d6f36c6d5`, TSPN, public configuration and native runtime exist, but `configs/experiments/p19/` was not retrievable; `scripts/paper/p19/` contains only `plot_signal_method.py`. No ready PHMFactory OP1 training/correction command is asserted. Inspect local research branches before writing code: an unmerged valid implementation may already exist.

**Reuse:** original prepared PHM population and physical roles; historical pilot/G6 results and failures; any genuinely compatible completed TSPN checkpoint/export. Do not repeat G6, the old six model fits, old bootstraps or completed data preparation. The attached older routing review does not authorize a new geometry objective or a return to Agent Benchmark.

## Inputs and protocol

### Inputs and ownership

Read actual local `AGENTS.md`, `CORE.md`, the current P19 method/claim map, this Goal and only relevant owner documentation. Retain unrelated changes.

Verified native owners are `phmfactory/config.py`, `phmfactory/cli.py`, `src/model_factory/X_model/TSPN.py`, `src/data_factory/`, `src/runtime/` and `docs/custom_dataset.md`. `scripts/paper/p19/` is the existing narrow helper location; use it only where an existing native export/evaluator cannot cover the required comparison. Do not clone a loader, trainer, split resolver or result manager.

The former reference routines are `signal_programs.py`, `operator_gain.py`, `operator_study.py`, and the native export binding at Agent Benchmark revision `76a195133c23149147d204fd2883eebcc2e9289a`. Read them for numerical migration, not as an installed/imported execution dependency. Preserve the minimum relevant computation and its license; do not migrate unrelated routing/Agent code. Existing source checks of that reference do not establish a PHMFactory-native run.

Local inputs previously supplied by the user:

- PHM root: `/home/user/data/PHMbenchdata/PHM-Vibench/`, including original `metadata.xlsx` and README; read-only.
- Prepared-role reference: `/home/user/LQ/B_Signal/astra/phm-agent-benchmark/local_outputs/p19_native_v2_20260919/prepared.json`. Reading an existing record is not permission to execute that repository. Resolve its actual raw paths, labels, unit IDs and source provenance through PHMFactory.
- Extra time-series root: `$HOME/data/timeseries`; HAR at `$HOME/data/timeseries/uci-har/raw/UCI HAR Dataset`. Do not download it again when complete. Never append the terminal `$` to a directory name.
- Previously used native Python: `/home/user/anaconda3/envs/LQ_signal/bin/python`. Current availability, PHMFactory installation and available memory/time must be checked locally, not assumed.

### Fixed Paderborn specification

Retain the declared study, not a new parameter search:

| Item | Fixed condition |
|---|---|
| Observation | Original 32-bearing assignment; 64 kHz vibration channel; 4,096-sample windows; 32 evenly spaced windows per record; original per-window standardization |
| Roles | Fit/tune/grow/select/test: 5/3/10/7/7 bearings. Actual IDs come from saved preparation, not invented from these counts |
| Backbone | Native TSPN, one I/WF/MWF/HT layer, 16 output channels, RMS/AbsMean/Std/Kurtosis/CrestFactor/Skewness statistics; three classes; no skip or additional internal instance normalization |
| Training | Original OP1 budget 100 epochs and seeds 11/23/37. Recover remaining optimizer, batching, precision and checkpoint-selection values from the approved effective native config before fitting; unresolved values are blockers, not silently chosen defaults |
| Qualification | Equal-bearing **tune** accuracy at least 0.80 after tune checkpoint/temperature selection; not grow accuracy and not a test lower confidence bound |
| External program | Fixed center/width pairs (1000,500), (4000,1000), (10000,2000), (20000,4000) Hz; modulation intervals [20,100), [100,300), [300,1000) Hz; lags 64/256/1024; 42 named features, not the superseded trainable-band design |
| Readouts | Intercept, native-matched, native-full, signal-program; same frozen calibrated logits and grow data; equal-unit offset multinomial NLL plus L2=0.01 on coefficients and intercept; zero initialization; at most 1,000 L-BFGS-B iterations |
| Native matching | Label-blind Gaussian QR projection, seed 11, to 42 columns; retain the full native vector as a separate stronger, unequal-capacity control |
| Probability decision | Same grow Brier segment and empirical select rule for all corrections. The fixed class-stratified pilot does not use an iid deployment certificate |
| Output | TSPN plus raw/projected/guarded versions of four corrections: 13 outputs, not 13 independent experiments |

Treat a PHMFactory source change as a real dependency change. Test frozen-state behavior and checkpoint compatibility; do not assume old runtime predictions remain identical. No existing PHM data or 1,517 record-length discrepancies may be erased or silently repaired in metadata.

### Metrics, repetitions and budget

Primary: equal-bearing Brier of `signal_program/guarded` minus TSPN. Information contrasts: projected signal minus projected intercept/native-matched/native-full. Keep accuracy and unit-weighted macro-F1 separately. Lower Brier difference and higher accuracy difference favor the signal correction; no sign is presumed.

The existing three training seeds measure optimization variability and are retained because this specific study already registered them and the old pilot had seed-dependent outcomes. They do not create extra bearings. First average seed effects within a bearing; then use the existing 2,000 paired whole-bearing stratified draws, seed 20260922, for the fixed 1/3/3 test composition. Report the singleton healthy-stratum limitation and each seed. Compute paired accuracy uncertainty from the same predictions for any accuracy-gain claim. Do not add training repeats after seeing an interval.

Binding uses only focused CPU checks and a native small-batch forward/backward/checkpoint smoke. Long training starts only after the actual native command and complete numerical config are established. At most the three fixed backbone runs are authorized for OP1, minus compatible completed work. All readout controls reuse each export. GPU policy remains one physical GPU 0, no GPU 2 and no multi-GPU; if unavailable, continue independent CPU/data tasks rather than selecting another device. Record actual local resource limits before a long run.

## Execution

### 1. Read-only binding checks

These are actual inspection commands, **not OP1 execution**:

```bash
: "${PHMFACTORY_ROOT:?Set the actual PHMFactory worktree}"
cd "$PHMFACTORY_ROOT"
git status --short
git branch --show-current
git rev-parse HEAD
git ls-files 'configs/experiments/*p19*' 'configs/experiments/p19/*' 'scripts/paper/p19/*'
python -m phmfactory --help
```

Check active PRs and the installed module origins. Do not reset, force-pull, replace the environment or edit the paper's Results. If compatible native code already exists, reuse and validate it; do not repeat a migration to satisfy this document.

### 2. Bind only the missing correction path

Reuse TSPN construction, native data/Task, checkpoint selection and prediction export. Port only absent named-feature, offset-readout, projection/acceptance and unit-analysis routines. Do not add an alternative Pipeline solely to launch this paper. Keep the fixed 42-feature specification and data roles.

Use a small labeled software fixture for arithmetic and serialization, clearly separated from scientific evidence. Verify fixed-input features, frozen backbone parameters and running state, exact zero-correction recovery, contribution sums, class-log-odds contrasts, physical-unit weights and saved-state replay. The log-odds contrast is computed from saved logits to avoid taking logarithms of underflowed probabilities. No new architecture or target-tuned normalization is introduced.

Complete a small native data/model/checkpoint smoke in the real environment. Missing metadata, unsupported configuration or a failed native smoke stops the affected long experiment; synthetic fixtures cannot replace this step. Validate only affected owners plus normal required CI.

### 3. Record and run the real native invocation

After the native binding exists, set `P19_CONFIG` to its actual reviewed training config. The following is the verified **public command interface**, conditional on that config being implemented and validated:

```bash
: "${P19_CONFIG:?Set the implemented and reviewed native P19 config}"
test -f "$P19_CONFIG"
python -m phmfactory preflight --config "$P19_CONFIG"
python -m scripts.config_inspect --config "$P19_CONFIG" --dump resolved --format yaml
```

Record the actual per-seed train/export/correction/analyze invocations in the existing execution notes. No undocumented `operator_fit`, `source`, `evaluate` or `--split` stage is assumed to exist in PHMFactory. `phmfactory --config` is the native training interface; it is not automatically an offset-readout analysis command. Use the existing helper or the minimal verified binding from step 2 and retain its real command.

Train/select/calibrate source-only, export the same windows and capture native features at the declared position. For each source-qualified seed, fit all four corrections and freeze their coefficients, segment strengths and select decisions before scoring test labels. Do not retrain a completed backbone to recover an export/analysis-only failure. No post-selection refit or post-test setting change is authorized.

A floor failure writes qualification and failure/completion status without a new correction test score, following the retained protocol. It is a valid source outcome but leaves the planned correction contrast incomplete. Do not report a favorable successful-seed subset as the three-seed primary.

### 4. External time-series work that can proceed independently

UCI HAR remains the first external task for the user's retained time-series scope. Its data/reader binding can proceed while Paderborn is blocked. Keep the official subject train/test boundary; use its own sampling rate, channels, provided window semantics and label ontology. Do not re-window across subjects or copy Paderborn's 64 kHz/4,096-sample/band configuration.

The exact HAR reader, role IDs inside training subjects, signal profile, TSPN dimensions, fitting budget and repetitions are **not yet bound in this snapshot**. Record those source-only choices and the verified invocation before a HAR performance run. No HAR training/test run is authorized by a placeholder path in this Goal. Return the binding for the next experiment-A handoff. The need for independent cross-task evidence is decided by the claim, not by whether Paderborn is positive. Do not download five datasets or launch a geometry-routing replication.

## Artifacts and validation

Use native returned result/checkpoint/config locations. Preserve input identities, raw/projected/guarded probabilities, unit-level loss and correct/count, class order, calibration, coefficients, feature names, projection/acceptance states, seeds, real command, necessary logs and failures. Keep native checkpoint and input-source references; raw data remain local.

The small P19 additions must support independent metric recomputation and feature-term reconstruction. Report both added-logit sums and class-log-odds contrasts from the same saved arrays; reference retention is checked separately for the deployed prediction. A nonzero raw correction that is rejected has zero deployed effect. Exact arithmetic is not proof of physical causality or a predictive benefit.

Use the entire prescribed cohort and all available registered-seed states. Distinguish infrastructure failure, invalid protocol, source-floor failure, valid negative effect and guard rejection. No missing metric is filled with zero. Time and memory fields remain missing unless actually measured; fitting time is not deployment latency.

Map evidence to the paper's C1–C4 table, Sections 3–4 and the existing OP1 settings/comparison locations. Plot only from saved analysis CSVs; no model inference or new bootstrap may hide inside a plot script. The method diagrams are reused, not redrawn as simulated results.

## Failure and sync

If a required input is absent, preserve the precise blocker and continue other independent tasks in this batch. With no executable item left, stop and return the binding gap. Environment fixes may preserve the protocol; changes to source roles, observations, model, objective, calibration, selection or estimator need an explicit scientific handoff and cannot be pooled with the original protocol.

Validated minimal native code/configuration and shareable evidence go through focused PHMFactory branches and PRs to dev. No force push, master/main changes, unrelated merges, raw-data upload or second runtime. Large artifacts follow the existing authorized storage with real accessible references. The paper links the actual accepted source and artifacts; it does not copy the execution body or write new Results in this stage.

Return the actual native config and commands, completed/reused/blocked items, all source qualifications and paired outcomes, retained failures, artifact paths and synchronization state. If no new execution occurred, say so. Stop after the essential batch; the evidence-B stage decides what the results support.

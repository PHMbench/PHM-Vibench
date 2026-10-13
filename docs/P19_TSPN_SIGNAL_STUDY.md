# P19 Local Execution SOP

This is the sole execution SOP for the P19 multiview study. The paper repository owns the scientific claims; PHMFactory owns configuration, data access, training, correction, evaluation and analysis. Agent Benchmark is a historical numerical/artifact reference, not a runtime dependency.

The implementation and synthetic CPU checks establish software behavior. **Real PHM data qualification, local CUDA/WSL validation, strong-baseline tuning and scientific performance experiments have not been run for this implementation.** Do not report software checks as multiview efficacy. The required second PHM dataset and the HAR profile remain unbound.

## A. Goal and fixed comparisons

Determine whether the existing signal-view correction improves a capable frozen TSPN beyond an intercept, native-feature readouts and each single view. Preserve the existing method; add joint condition/entity DG as a separate prospective experiment. The original seven-bearing OP1 target has historical exposure, so OP1 remains exploratory. The supplied Paderborn DG configurations also retain an exploratory evaluation label rather than inventing a clean target-access history.

| Question | Controlled comparison | Outcome |
|---|---|---|
| C1: deployed incremental value | `signal_program/guarded` minus the same TSPN | Primary: equal-entity full multiclass Brier difference; lower favors the correction |
| C1: value of supplied signal coordinates | Projected signal program minus projected intercept, native-matched and native-full readouts | Separates view information from another fitted head and readout capacity |
| C1: joint-view utility | Projected signal program minus each projected single-view readout | Tests the four declared families without searching view subsets |
| C2: computational explanation | Replay saved feature/view terms against raw added logits and class log-odds | Arithmetic fidelity, not physical causality or a linear decomposition of mixed probabilities |
| Generalization | Repeat the same comparisons over frozen joint condition/entity folds and at least two qualified PHM datasets | Paderborn alone is incomplete multi-PHM evidence; HAR cannot replace the second PHM dataset |

Accuracy and Macro-F1 are secondary outcomes with their own paired intervals. RAW outputs are diagnostic. A rejected correction has zero deployed effect. The original 13 outputs remain intact; four single-view readouts add 12 outputs from the same TSPN export, for 25 in total. They require no new backbone fits.

### OP1 and DG are distinct budgets

| Setting | Retained OP1 | New Paderborn DG |
|---|---|---|
| Configuration | [op1.yaml](../configs/experiments/p19/op1.yaml) | [paderborn_dg_0.yaml](../configs/experiments/p19/paderborn_dg_0.yaml) through `_3.yaml` |
| Population | Actual original 32-bearing assignment; fit/tune/grow/select/test = 5/3/10/7/7 | Same fixed bearing roles; source-role bearings in three source conditions, test-role bearings in the held-out condition; exclude cross combinations |
| Backbones | TSPN only | TSPN, ResNet1D and TCN; same source search budget |
| Training | 100 epochs; AdamW LR `1e-3`, WD `1e-4`; CE; batch 128; seeds 11/23/37 | Four ordered `(LR,WD)` trials: `(1e-3,0)`, `(1e-3,1e-4)`, `(3e-4,0)`, `(3e-4,1e-4)`; each 100 epochs at seed 11 |
| Checkpoint | Minimum `val_loss`, earliest epoch on an exact tie; no early stopping or automatic post-fit test | Same checkpoint rule; trial ranking by uncalibrated entity-weighted tune Macro-F1, Accuracy, lower Brier, then trial order |
| Repetition | Three fixed fits; no added HPO | Reuse selected seed-11 fit; train the selected configuration at 23 and 37: **six fits/model/fold**, 18 per fold, 72 across four folds |
| Calibration/refit | Tune-only temperature; no fit+tune refit | Temperature fitted after trial selection; no fit+tune refit |
| Source floor | Equal-bearing tune Accuracy >=0.80; a failed seed receives no correction test score and blocks complete evaluation | Tune Accuracy/Macro-F1 below 0.80 is diagnostic, not a filter; keep valid difficult folds and all seeds |
| Bootstrap | 2,000 paired class-stratified whole-bearing draws, seed 20260922; fixed 1/3/3 test composition has a singleton healthy stratum | 2,000 paired whole-bearing draws, seed 20261013; report folds separately |

The fit counts above describe formal model selection/repetition. The bounded smoke and source-overfit diagnostics below are additional software-qualification work and do not contribute comparison scores. The native model set is an implemented comparison, not yet a qualified strong-baseline result. An appropriate official strong time-series reference, such as Hydra, has not been bound or validated under this protocol. Do not substitute published scores on other splits or claim SOTA. TimesNet is not one of the current three configured baselines; any future qualification must retain the declared source-only budget and inference boundary.

The DG reference architectures are frozen before optimizer search: ResNet1D basic blocks `[2,2,2,2]` and initial width 64 follow the [native defaults](../src/model_factory/CNN/ResNet1D.py), rather than its reduced testing example. TCN channels `[32,64,64,64]` and kernel size 7 follow the existing [P04 comparator configuration](../configs/experiments/p04/baseline_tcn_indomain.yaml); dropout 0.2 is the [native default](../src/model_factory/CNN/TCN.py). P04 supplies architecture provenance only, not P19 data/split/training rules or DG qualification. These choices are not measured competitive results and do not change OP1, the method or the equal search/epoch/seed budget.

## B. Environment and inputs

Use the reviewed PHMFactory worktree, currently `/home/user/LQ/B_Signal/PHMFactory-p19-implementation-20261013`, and record its actual revision and dirty state. The corresponding paper worktree is `/home/user/LQ/B_Signal/astra/P19-implementation-20261013`. These local paths are not proof of a merged remote revision.

- Existing Python candidate: `/home/user/anaconda3/envs/LQ_signal/bin/python`; package metadata requires Python >=3.10. Verify the actual environment locally.
- Dependencies: the repository [requirements.txt](../requirements.txt), including PyTorch 2.6.0, Lightning, NumPy, SciPy, pandas, HDF5 and Excel readers. Do not upgrade/install silently to make a failed run pass. Record installed versions and the actual CPU/CUDA build.
- Raw PHM root: `/home/user/data/PHMbenchdata/PHM-Vibench/`, original `metadata.xlsx` and signals read-only. No metadata repair, normalization rewrite or sample deletion in place.
- Existing historical preparation: `/home/user/LQ/B_Signal/astra/phm-agent-benchmark/local_outputs/p19_native_v2_20260919/prepared.json`. Read its references to recover the actual approved physical assignment; do not execute the historical repository or infer bearing IDs from counts.
- `study.role_map` must point to the actual physical-split JSON. Its `assignments` maps bearing IDs to roles; historical `design` assignments are resolved by the saved `grow_units` and `select_units` lists. All five final roles and their fixed counts must be covered. Record/window IDs are not physical entity substitutes.
- Existing HAR data are under `$HOME/data/timeseries/uci-har/raw/UCI HAR Dataset`. The channel/role/frequency/lag/native-dimension profile is unbound, so there is no HAR performance command in this SOP.
- GPU: query only physical GPU 0 with `nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader`, require its single nonempty GPU UUID, and export that exact UUID as `CUDA_VISIBLE_DEVICES` before Python imports. The logical model device remains `cuda:0`. Numeric `CUDA_VISIBLE_DEVICES=0` is rejected because CUDA device ordering can differ from the physical index. No GPU 2, multi-GPU or automatic fallback is allowed. Explicit CPU smoke/overfit calls set `CUDA_VISIBLE_DEVICES=''`. Available GPU memory, CUDA driver/runtime and WSL behavior must be checked locally.

The Paderborn profile is fixed at 64 kHz, channel index 2, 4,096 samples, 32 evenly spaced windows/record, native per-window standardization and no additional TSPN instance normalization. Its four Gaussian center/width pairs are `(1000,500)`, `(4000,1000)`, `(10000,2000)`, `(20000,4000)` Hz; modulation intervals are `[20,100)`, `[100,300)`, `[300,1000)` Hz; lags are 64/256/1024 samples. Check actual channel/label semantics before running. The 20 kHz center requires sampling rate >40 kHz; no silent clipping, zero padding or resampling transfers this profile to another dataset. Each modulation interval must contain an observable DFT bin and each lag must fit the window.

The native TSPN has 96 feature coordinates here; label-blind Gaussian QR seed 11 supplies the 42-dimensional native-matched control. Native-full remains a stronger unequal-capacity control. All readouts use the same calibrated offset and grow data, equal-entity multinomial NLL, L2=0.01 on coefficients and intercept, zero initialization and at most 1,000 L-BFGS-B iterations. Grow fits the probability segment and select fixes the empirical guard. A multichannel profile must explicitly satisfy the matching dimension requirement rather than pad missing coordinates.

## C. Preflight sequence

Proceed in order **environment → data → split → baseline → method → metric**. Failure stops affected formal experiments; preserve the failure and fix the actual cause before retrying into a fresh output directory.

1. Verify the selected interpreter imports this worktree and the requested GPU maps to physical 0. `research preflight` resolves configuration/module ownership only: it does not qualify real data or prove execution.
2. `sanity` reads metadata, checks the profile, labels, role counts and class support, and writes derived metadata/manifests. Source training receives a separate source-only metadata population; target evaluation receives the separate evaluation population. Original metadata are unchanged. The preparation may inspect target assignment/class-support metadata as a protocol check; target signals/predictions cannot enter source fitting or selection.
3. Inspect actual entity IDs, `Domain_id` meanings and excluded cross combinations. Each DG fold must contain multiple source conditions and a disjoint target condition **and** disjoint source/target bearings. A condition code without a verified physical entity is insufficient. Mark unsupported class/entity coverage infeasible; do not redraw roles after seeing results.
4. Run native baseline and TSPN CPU smoke phases. Each uses one epoch and two windows/record through the native loader/task/trainer. Then run `overfit`: 100 epochs and two windows/record on source fit data, requiring entity-weighted fit Accuracy >=0.99 at the restored minimum-val-loss checkpoint. It uses ResNet1D for DG and TSPN for OP1, with tune/checkpoint access still source-only. A failure stops formal experiments for inspection; do not reduce the threshold or inspect target to fix it. This diagnostic tests a source training path; it does not certify every baseline or prove competitiveness.
5. Confirm source checkpoint replay, finite tensors/gradients, frozen parameters **and running buffers**, and native features captured from the same forward pass as logits. Fit/tune/grow/select all precede test: fit trains the backbone; tune selects/calibrates; grow fits readouts/normalizers/segments; select fixes empirical guards. Validation, calibration and export batch size is 1 for every model. This also prevents TimesNet's native batch-level period statistic from using other target items if it is later qualified.
6. Verify the saved class order, probability rows and metric units. Each of G entities has total weight `1/G`; each of its m_g windows has weight `1/(G*m_g)`. Accuracy and full multiclass Brier are entity averages. Macro-F1 is computed from that weighted confusion matrix over the fixed classes, with zero-denominator class F1 set to zero and missing support reported; it is not a mean of per-bearing F1 values. Seeds are repeated fits, not extra independent entities.

For Brier/Accuracy, average paired seed effects within each entity before whole-entity resampling. For Macro-F1, recompute each seed's weighted confusion/F1 for each common resample, then average paired seed differences. Never average probabilities into an undeclared ensemble. Report seed dispersion and the healthy-singleton limitation; intervals condition on the registered cohort and fitted models.

## D. Executable commands

These commands use the implemented public entrypoint and require actual local inputs. They do not fabricate a role map or a second-dataset profile. Fill the three required local variables once using absolute paths. Use an explicit local YAML for machine paths only; all commands retain the same local layer and overrides in Bash arrays. Do not change scientific values between phases.

### 1. Environment, configuration and data sanity

```bash
set -euo pipefail
PHMFACTORY_ROOT="${PHMFACTORY_ROOT:-/home/user/LQ/B_Signal/PHMFactory-p19-implementation-20261013}"
P19_PYTHON="${P19_PYTHON:-/home/user/anaconda3/envs/LQ_signal/bin/python}"
: "${P19_LOCAL_CONFIG:?Set the actual machine-local YAML path}"
: "${P19_ROLE_MAP:?Set the approved physical-split JSON path}"
: "${P19_OUTPUT_ROOT:?Set a fresh local result root for this execution}"
test -f "$P19_LOCAL_CONFIG"
test -f "$P19_ROLE_MAP"
test ! -e "$P19_OUTPUT_ROOT"
cd "$PHMFACTORY_ROOT"
P19_GPU0_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)"
[[ "$P19_GPU0_UUID" =~ ^GPU-[[:xdigit:]-]+$ ]]
export CUDA_VISIBLE_DEVICES="$P19_GPU0_UUID"
git status --short
git rev-parse HEAD
"$P19_PYTHON" --version
"$P19_PYTHON" -m phmfactory --help
"$P19_PYTHON" -m pip show phmfactory torch pytorch-lightning numpy scipy
nvidia-smi --id=0
mkdir -p "$P19_OUTPUT_ROOT/console"
p19_common=(--local-config "$P19_LOCAL_CONFIG" --override "study.role_map=$P19_ROLE_MAP")
p19_name=paderborn_dg_0
p19_config=configs/experiments/p19/paderborn_dg_0.yaml
p19_args=(--config "$p19_config" "${p19_common[@]}")
p19_run() {
  local phase="$1"
  shift
  local log="$P19_OUTPUT_ROOT/console/${p19_name}_${phase}.log"
  test ! -e "$log"
  "$P19_PYTHON" -m phmfactory research "$phase" "${p19_args[@]}" "$@" \
    2>&1 | tee "$log"
}
p19_run preflight
p19_run sanity --output "$P19_OUTPUT_ROOT/$p19_name/sanity"
p19_data="$P19_OUTPUT_ROOT/$p19_name/sanity/prepared.json"
```

If the GPU is unavailable, the environment check stops before formal execution. Do not start tuning or select another GPU. Independent metadata checks and explicit CPU diagnostic work may proceed separately after recording the unavailable GPU. Read `source_metadata.csv`, `source_split.json`, `evaluation_split.json` and `excluded_cross_combinations.csv` locally and complete the split checks above before progressing.

### 2. Baseline smoke

```bash
CUDA_VISIBLE_DEVICES='' p19_run baseline_smoke --data "$p19_data" --device cpu \
  --output "$P19_OUTPUT_ROOT/$p19_name/baseline_smoke"
```

The DG baseline smoke uses the configured ResNet1D. Formal tuning still includes all three configured backbones.

### 3. Proposed-method smoke

```bash
CUDA_VISIBLE_DEVICES='' p19_run method_smoke --data "$p19_data" --device cpu \
  --output "$P19_OUTPUT_ROOT/$p19_name/method_smoke"
CUDA_VISIBLE_DEVICES='' p19_run overfit --data "$p19_data" --device cpu \
  --output "$P19_OUTPUT_ROOT/$p19_name/overfit"
```

The method smoke exercises the native TSPN training/checkpoint path. `overfit` is the separate 100-epoch source diagnostic described above. Synthetic tests cover correction arithmetic and replay; the real source readout fit is the later `fit` phase. Do not claim a real correction comparison from a smoke/overfit result.

### 4. Strong-baseline tuning

```bash
p19_run tune --data "$p19_data" \
  --output "$P19_OUTPUT_ROOT/$p19_name/tune"
```

`tune` performs the complete declared grid and selected-seed repeats; `selection.json` records the selected checkpoints and source selection rule. A failed trial invalidates the declared search budget: retain `tuning_failures.json` and stop, rather than choose among a smaller successful grid. Source performance/failure review precedes method assessment. No additional architecture/trial search is hidden in `fit`.

### 5. Small source comparison before target access

```bash
p19_run fit --data "$p19_data" \
  --selection "$P19_OUTPUT_ROOT/$p19_name/tune/selection.json" \
  --output "$P19_OUTPUT_ROOT/$p19_name/fit"
p19_run source_compare --data "$p19_data" \
  --selection "$P19_OUTPUT_ROOT/$p19_name/fit/frozen_index.json" \
  --output "$P19_OUTPUT_ROOT/$p19_name/source_compare"
```

`fit` calibrates and freezes every registered model/seed state. `source_compare` replays the 25 TSPN/readout predictions on saved source select data without another fit or target access. These labels already determined the guard: this is a source diagnostic, not an independent validation estimate or a new selection rule. Review infrastructure/protocol failures before target evaluation; do not retune the method in response to this diagnostic.

### 6. Full DG evaluation and independent OP1

After source qualification and freeze, evaluate the first fold, then continue the predeclared folds with the same machine layer and physical assignment:

```bash
p19_run evaluate --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/fit/frozen_index.json" --output "$P19_OUTPUT_ROOT/$p19_name/evaluate"
p19_run analyze --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/evaluate/evaluation.json" --output "$P19_OUTPUT_ROOT/$p19_name/analyze"
for p19_fold in 1 2 3; do
  p19_name="paderborn_dg_$p19_fold"
  p19_config="configs/experiments/p19/$p19_name.yaml"
  p19_args=(--config "$p19_config" "${p19_common[@]}")
  p19_run preflight
  p19_run sanity --output "$P19_OUTPUT_ROOT/$p19_name/sanity"
  p19_data="$P19_OUTPUT_ROOT/$p19_name/sanity/prepared.json"
  CUDA_VISIBLE_DEVICES='' p19_run baseline_smoke --data "$p19_data" --device cpu --output "$P19_OUTPUT_ROOT/$p19_name/baseline_smoke"
  CUDA_VISIBLE_DEVICES='' p19_run method_smoke --data "$p19_data" --device cpu --output "$P19_OUTPUT_ROOT/$p19_name/method_smoke"
  CUDA_VISIBLE_DEVICES='' p19_run overfit --data "$p19_data" --device cpu --output "$P19_OUTPUT_ROOT/$p19_name/overfit"
  p19_run tune --data "$p19_data" --output "$P19_OUTPUT_ROOT/$p19_name/tune"
  p19_run fit --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/tune/selection.json" --output "$P19_OUTPUT_ROOT/$p19_name/fit"
  p19_run source_compare --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/fit/frozen_index.json" --output "$P19_OUTPUT_ROOT/$p19_name/source_compare"
  p19_run evaluate --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/fit/frozen_index.json" --output "$P19_OUTPUT_ROOT/$p19_name/evaluate"
  p19_run analyze --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/evaluate/evaluation.json" --output "$P19_OUTPUT_ROOT/$p19_name/analyze"
done
```

A valid negative effect is not a reason to alter the protocol, drop a fold or add trials. Target evaluation is downstream of complete source freeze; it cannot revise the source choices.

The separate OP1 command uses its original fixed three-fit budget. First recover any compatible completed work and verify checkpoint/dependency behavior; do not blindly duplicate old training or treat the historical routing pilot as this method's result.

```bash
p19_name=op1
p19_args=(--config configs/experiments/p19/op1.yaml "${p19_common[@]}")
p19_run preflight
p19_run sanity --output "$P19_OUTPUT_ROOT/$p19_name/sanity"
p19_data="$P19_OUTPUT_ROOT/$p19_name/sanity/prepared.json"
CUDA_VISIBLE_DEVICES='' p19_run method_smoke --data "$p19_data" --device cpu --output "$P19_OUTPUT_ROOT/$p19_name/method_smoke"
CUDA_VISIBLE_DEVICES='' p19_run overfit --data "$p19_data" --device cpu --output "$P19_OUTPUT_ROOT/$p19_name/overfit"
p19_run tune --data "$p19_data" --output "$P19_OUTPUT_ROOT/$p19_name/tune"
p19_run fit --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/tune/selection.json" --output "$P19_OUTPUT_ROOT/$p19_name/fit"
p19_run source_compare --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/fit/frozen_index.json" --output "$P19_OUTPUT_ROOT/$p19_name/source_compare"
p19_run evaluate --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/fit/frozen_index.json" --output "$P19_OUTPUT_ROOT/$p19_name/evaluate"
p19_run analyze --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/evaluate/evaluation.json" --output "$P19_OUTPUT_ROOT/$p19_name/analyze"
```

An OP1 floor failure yields `incomplete_qualification`; evaluation rejects that index. Do not edit it to score passing seeds alone. The stage named `tune` still uses only OP1's one fixed configuration at three seeds, not DG HPO.

These four Paderborn folds do **not** complete multi-PHM DG. Qualification of a second PHM dataset requires actual physical entities, multiple meaningful conditions, fixed class support and a legal profile; its identity and bindings must be settled before a config/performance run. No placeholder dataset/HAR loop is supplied.

### 7. Ablation and mechanism analysis

The existing `fit`/`evaluate`/`analyze` stages already fit all four single-view controls and save their raw/projected/guarded outputs, plus the exact feature/view explanation replay. Use those artifacts for E1–E3; do not train another backbone for each view. Optional plots consume the saved analysis CSV, with no inference, new resampling or outcome-selected comparisons:

```bash
for p19_name in paderborn_dg_0 paderborn_dg_1 paderborn_dg_2 paderborn_dg_3 op1; do
  p19_args=(--config "configs/experiments/p19/$p19_name.yaml" "${p19_common[@]}")
  p19_data="$P19_OUTPUT_ROOT/$p19_name/sanity/prepared.json"
  p19_run plot --data "$p19_data" --selection "$P19_OUTPUT_ROOT/$p19_name/analyze/analysis.json" --output "$P19_OUTPUT_ROOT/$p19_name/plot"
done
```

Only invoke this optional loop when all listed analyses exist; preserve an incomplete OP1 gate as incomplete, without a substitute plot. If only export/analysis failed, repair and rerun that stage from compatible saved inputs into a fresh path rather than retrain a completed backbone.

## E. Expected outputs

Every execution stage creates its own fresh directory and `stdout.log`/`stderr.log`, writes `config.json` and `command.json` for valid stage configuration, and records `completion.json` on success or `failure.json` with the phase/error/traceback on failure after stage creation. Unsupported CLI/configuration failures may occur before output creation; the console log preserves these. `preflight` creates no stage output. Native checkpoints and arrays are local artifacts, not uploads.

| Stage | Required artifacts and meaning |
|---|---|
| `sanity` | `prepared.json`, `source_metadata.csv`, `evaluation_metadata.csv`, `source_split.json`, `evaluation_split.json`, `excluded_cross_combinations.csv`; derived populations only |
| `baseline_smoke`, `method_smoke`, `overfit` | `native_smoke/training.json`, `effective_config.json`, native YAML/checkpoint paths, `stdout.log`, `stderr.log`, `tune_reference.npz`, `fit_reference.npz`, `fit_metrics.json`; status `smoke_only` or `overfit_diagnostic_only`, no target score |
| `tune` | `tuning_trials.csv`, `selection.json`; each native run's configuration, log, minimum-val-loss checkpoint and tune reference; `tuning_failures.json` and per-trial failure records if the grid fails |
| `fit` | `frozen_index.json`; per-model/seed `qualification.json`, `frozen.json` when qualified; TSPN `source_export/{fit,tune,grow,select}.npz`, `operator_inputs.npz`, `effective_config.json`, `sources.json`; temperature, coefficient/projection/guard state in the freeze |
| `source_compare` | `source_select_metrics.csv`; source-only replay of the 25 TSPN/readout outputs, not an independent estimate after guard selection |
| `evaluate` | `evaluation.json`; per-model/seed `target_export/`, `predictions.npz`, `metrics.csv`; TSPN `explanations.npz` and `explanation_check.json` |
| `analyze` | `analysis.json`, `summary_metrics.csv`, `seed_metrics.csv`, `paired_effects.csv`, `contribution_results.csv`; all registered models/seeds, physical-entity intervals and claim mapping |
| `plot` | `comparison_metrics.png`, `comparison_metrics.pdf`; saved summary CSV only, no model/data/uncertainty recomputation |
| Any failure | Original console/native logs and `failure.json` where created; exact command/config, failed phase and retained predecessor artifacts; no invented zero metric |

Full multiclass Brier sums over classes, without dividing by their count. Accuracy and Macro-F1 are fractions in `[0,1]`; any presentation as percent must label the conversion. A negative paired Brier difference favors the candidate; positive Accuracy/F1 differences favor it. Preserve class order, item/window coordinates, unit IDs, seed, all probabilities and the frozen state so scores can be recomputed independently.

## F. Acceptance and stopping criteria

Accept the **software chain** only when actual local environment/import checks, complete physical/domain splits, source baseline/TSPN smoke, the declared source-overfit check, checkpoint replay, source freeze and metric recomputation pass. CUDA binding must match the physical GPU 0 UUID before any CUDA initialization and expose exactly one device; a failed identity check stops execution. Synthetic CPU tests alone are insufficient for real-data qualification. Native export requires checkpoint logits/features to replay within `1e-6`; saved raw feature/view contributions must reconstruct logits within `1e-10`. Reject nonfinite arrays/gradients, invalid probabilities, dimension/rate/label mismatches, missing roles/classes or any target population entering the source data factory.

Accept a **scientific comparison** only with complete registered trials/models/seeds, equal information/training/search budgets, source-only choice rules and the same target items. OP1 retains its stop gate. DG retains low-source-performance diagnostics without censoring valid negative outcomes. Floor passage alone does not establish a strong baseline. Missing source-overfit qualification, a failed grid, an incomplete freeze or a changed dependency/checkpoint replay blocks the affected claim.

Freeze after source stages; test does not select roles, profiles, models, checkpoints, temperatures, hyperparameters, seeds or new datasets. Preserve guard rejection, zero mixing, difficult folds and inconsistent signs. No favorable subset replaces the planned complete comparison. No efficiency claim follows unless the complete preprocessing/training/inference cost is actually measured.

TSPN's constant-feature gradient fix changes native dependency behavior. Although it retains the forward population-standard-deviation formula and running-state update, do not assume identical training trajectories or old-checkpoint predictions without replay. See the [implementation changelog](../doc/changelog/p19-native-multiview-20261013.md).

## G. Result return format and boundary

Return a concise local report with actual paths, not raw/private arrays:

1. Source revision/worktree, interpreter/package/CUDA/device versions, data version/path and exact common local-config/override arguments.
2. Successful, failed, reused and not-run experiments by protocol/fold/model/seed; infrastructure/protocol/source-floor/valid-negative outcomes distinguished.
3. Complete baseline search table, selected configurations, source qualification/learning checks and any incomplete budget; no claim that an unqualified external reference is baseline-valid.
4. Proposed-method primary, projected information/SV controls, secondary Accuracy/Macro-F1, per-seed dispersion, paired intervals and arithmetic reconstruction checks.
5. Bugs/anomalies, excluded cross combinations, singleton/class-support limitations, and complete config/log/checkpoint/prediction/metric/summary/failure paths.
6. `Contribution–Experiment–Result` mapping using `contribution_results.csv`; mark unsupported or incomplete conclusions plainly.
7. The next smallest necessary experiment, with its hypothesis and stopping rule; identify the unresolved second-PHM/HAR binding explicitly.

Only code, documentation and synthetic software fixtures are shareable by default. Keep raw signals, spreadsheets, private predictions/checkpoints and other private arrays local; do not upload them or enable external logging. No merge/push/data download, dependency upgrade, target-driven rerun or new dataset experiment follows automatically from this SOP. Write new paper Results only after the actual evidence supports them.

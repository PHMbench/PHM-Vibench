# P08 TII — native execution and Local Codex SOP

Updated 2026-10-02. This replaces the former P4 ZIP/import-only handoff. All P08 algorithm, data, experiment, metric and plotting code now belongs to PHMFactory; the paper stays in AI4Engineering-L/P08-HSE-Prompt-CDDG. No copied HSE, raw reader, config compiler or trainer is maintained in the paper repository.

## Goal and ownership

RQ1: record-level benefit of deployment-measured physical conditions over HSE-only. RQ2: same-information fusion comparison. RQ3: conditional sensitivity and applicable boundaries. Tests establish software semantics, not industrial accuracy or a qualified strong baseline.

| Owner | Existing/native implementation |
|---|---|
| Data | `src/data_factory/p08_data.py`; delegates raw reading to ExplicitDataFactory and H5 to H5DataDict |
| Model | `src/model_factory/ISFM/M_P08_PhysicalConditioning.py`; imports maintained E_01_HSE |
| Task | `src/task_factory/task/DG/p08.py`; original-record probability aggregation and unit-balanced selection |
| Training | Existing `src/runtime/classification.py` + Trainer Factory, `test_after_fit=false` |
| Experiment | `scripts/p08_experiments.py`; native analyze_config and pipeline, bounded source tuning, separate checkpoint evaluation |
| Configuration | `configs/experiments/p08/source_only.yaml` plus explicitly supplied local YAML |

`DG/tii_joint` is another paper's protocol and is not a P08 substitute. Old HSE-prompt reference presets may use IDs/other tasks; they are not P08 strong baselines. Do not restore the old P4 training or intervention suites.

## Environment and local inputs

Use the actual runtime checkout, conda `LQ_signal`, and `CUDA_VISIBLE_DEVICES=0`. Formal training/evaluation uses one GPU only. No torchrun, DDP, DataParallel, GPU2, automatic device switching or CPU fallback. Explicit CPU software tests are separate from formal GPU experiments. Use upstream requirements and the CUDA-compatible PyTorch installation; do not downgrade an existing working environment silently.

Set `PHMFACTORY_ROOT`, `P08_LOCAL`, `DATA_ROOT`, `TARGET_INVENTORY` and `RUN_ROOT` to real absolute paths. The `/mnt/e/D01_vibench` in the template is an example, not a remotely verified mount. `P08_LOCAL` supplies only the relevant explicit overrides. No tool here can confirm local data or WSL/CUDA availability.

Each outer fold needs two separate original-record tables, using the native CSV/TSV/Excel reader:

```text
Id,Name,File,Dataset_id,Label,LabelName,Group,record_id,role,Channel,Sample_rate,<physical fields>
```

Source table contains only declared source IDs and `source_train`/`source_val`; every source has disjoint physical units for both roles. Target table contains only the declared unseen system and `target_test`. `Group` and `record_id` must identify real physical units and original records globally, not renamed windows. A source and target physical unit may not overlap. Label integers and `LabelName` must agree with one justified physical ontology. Do not guess labels from filenames or equate dataset-local codes.

For `storage=raw`, native reader resolves `data_dir/raw/Name/File`; for `h5`, native H5DataDict reads `data_dir/Name.h5[Id]`. H5 must contain original sample-by-channel records; set `h5_layout=sample_channel` or `sample_channel_singleton` explicitly. Pre-windowed arrays with unknown provenance are not accepted. The current adapter keeps selected original signals in RAM; check their total size before a full inventory run, never silently drop records.

Each condition field declares name, continuous/categorical kind, unit, physical meaning and independent deployment measurement source. Missing values/default flags remain explicit. Source_train observed records alone fit median/IQR/range/vocabulary. An all-missing or zero-IQR field is a scientific input issue, not permission to invent a scale. Dataset/file/unit IDs and sampling rate cannot be condition fields. Sampling rate belongs to HSE coordinates. Set `condition_dim` from preflight output; set `num_classes` from the fixed ontology.

The template deliberately has `label_names: []`: local ontology binding is required before a real experiment. Candidate systems remain CWRU/Ottawa-19/THU/JNU/HUST24, but their actual independent units, label overlap and condition availability must be confirmed. Do not manufacture a full five-system result from incompatible populations.

## Preflight and commands

From the checked-out, reviewed implementation branch or its merged dev descendant:

```bash
conda activate LQ_signal
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PHMFACTORY_ROOT"
git status --short
git rev-parse HEAD
python -m pytest test/test_p08_model.py test/test_p08_protocol.py -q
python -m scripts.p08_experiments --help
CFG=configs/experiments/p08/source_only.yaml
python -m scripts.p08_experiments preflight --config "$CFG" --local-config "$P08_LOCAL" --out "$RUN_ROOT/preflight"
```

Check native record/label/channel/fs semantics, source/validation units, source condition schema, window counts and condition width before fitting. Fix binding mistakes, not labels/splits chosen from target performance. Check source RAM/disk then GPU/CUDA. All output directories must be new; failures are retained.

First source-only baseline and method smokes, not performance comparisons:

```bash
python -m scripts.p08_experiments fit --config "$CFG" --local-config "$P08_LOCAL" --override model.fusion=none --override trainer.num_epochs=1 --override data.train_batches_per_epoch=2 --override trainer.early_stopping=false --out "$RUN_ROOT/baseline_smoke"
python -m scripts.p08_experiments fit --config "$CFG" --local-config "$P08_LOCAL" --override model.fusion=film --override trainer.num_epochs=1 --override data.train_batches_per_epoch=2 --override trainer.early_stopping=false --out "$RUN_ROOT/method_smoke"
```

Verify finite CE, gradients, checkpoint restore, condition use and complete record-level validation. The source fit never reads a target inventory. Small-batch overfit is already a software test; neither it nor one epoch qualifies a tuned baseline.

## Bounded strong-baseline selection

Current search template: none (B1), late_concat, then film (P0); learning rates 0.0003/0.001/0.003; weight decays 0/0.0001; maximum40epochs with identical source early-stop rules. This is6trials per arm,18 per outer fold/selection seed. Each arm selects its own minimum source unit-balanced Brier; exact ties choose the first declared trial. Stored and active parameter counts are reported separately. This search is an initial bounded opportunity, not an assertion of optimality or public SOTA.

Only start after explicit local compute approval for the resulting total folds/seeds/trials. The command refuses a budget smaller than its full planned grid. A budget does not excuse an unstable or obviously failed baseline.

```bash
python -m scripts.p08_experiments tune --config "$CFG" --local-config "$P08_LOCAL" --max-fits 18 --out "$RUN_ROOT/source_tuning"
```

Tuning writes per-trial configs/checkpoints and `selected.json`; no target score is computed. Retain losses and all trials, including failures. Inspect source learning stability before opening target data. Select architecture/normalization/fields only from permitted sources. A task-compatible published competitor remains a required comparison if the eventual contribution claims advantage beyond these same-information baselines; the included late fusion is not an official reproduction of every DI-ERM variant.

For a minimal repeat plan, predeclare two optimization seeds7and23 before target access. Tune with seed7; refit each arm's chosen configuration under seed23 without reselecting on target. Keep data split fixed and set both `environment.seed` and `data.sampling_seed` explicitly. This assesses optimization sensitivity, not independent devices. More repeats require a source-variance/precision justification, not an unattractive target result. Exact repeat count can change before target evaluation if resources/precision require it; record the choice once.

## Frozen evaluation and smallest complete comparison

Obtain `CHECKPOINT` from the source-selected arm in `source_tuning/selected.json` (or its predeclared repeat's fit.json). Never select a trial, seed or checkpoint using target performance. Only trusted own checkpoints are loaded.

```bash
python -m scripts.p08_experiments evaluate --checkpoint "$CHECKPOINT" --inventory "$TARGET_INVENTORY" --data-root "$DATA_ROOT" --device cuda --out "$RUN_ROOT/target_selected_arm"
```

Run all predeclared selected arms on the same target record inventory. This is the first complete comparison, not a pilot used to change the method and then counted again as confirmation. Repeat for separately frozen valid outer-fold bindings. Cross-condition domains may use the same machinery only after the domain/unit definition is explicitly frozen; don't rename a system ID to claim a different experiment.

For the factorial contrast use the same source-selected common settings, not each cell's separately optimized settings: B0=(index,none), B1=(physical,none), F01=(index,film), P0=(physical,film). Existing matching fits can be reused; record exact compatibility. Fit through the same command with `--override model.coordinates=index` and/or `--override model.fusion=film`. Keep source sampling, inputs, preprocessing and epochs/selection opportunity identical. Report main contrasts and interaction without requiring it to be positive.

Frozen mechanism checks require no refitting:

```bash
python -m scripts.p08_experiments evaluate --checkpoint "$P0_CHECKPOINT" --inventory "$TARGET_INVENTORY" --data-root "$DATA_ROOT" --device cuda --detach --out "$RUN_ROOT/p0_detached"
# WRONG_CONDITIONS is predeclared from physical/source knowledge, never target labels.
python -m scripts.p08_experiments evaluate --checkpoint "$P0_CHECKPOINT" --inventory "$TARGET_INVENTORY" --data-root "$DATA_ROOT" --device cuda --conditions "$WRONG_CONDITIONS" --out "$RUN_ROOT/p0_wrong_conditions"
```

Alternative table contains `record_id` plus exactly the declared physical fields, with identical record population. Define a physically meaningful donor/replacement before target evaluation; numeric perturbation and discrete observation-state changes are distinct. Inapplicable controls justify narrowing the corresponding interpretation, not arbitrary field permutation. A changed representation is not an accuracy gain.

## Analysis, outputs and acceptance

```bash
python -m scripts.p08_experiments compare --predictions "$BASELINE_PREDICTIONS" "$P0_PREDICTIONS" --bootstrap-repeats 1000 --analysis-seed 0 --out "$RUN_ROOT/paired_analysis"
python -m scripts.p08_experiments plot --predictions "$BASELINE_PREDICTIONS" "$P0_PREDICTIONS" --out "$RUN_ROOT/figures"
```

Only existing CSVs are read by analysis/plotting; neither re-invokes the model. Compare requires identical records, labels and physical units; percentile intervals resample paired physical units within each target system, conditional on the fixed checkpoints. They are not confidence intervals for new machines or retraining. A single unit gets no bootstrap interval. Seeds and shared-training LOSO folds are not independent physical units. Plot SVGs are descriptive; effect intervals are in comparison.json.

Native outputs: source config/logs/checkpoints; each checkpoint contains source condition schema, ontology, group exclusions and model/data configuration. Wrapper outputs: invocation/config/fit/search/completed_trials/selected JSON. Frozen evaluation saves window_predictions.csv, record predictions.csv, metrics.json, parameter counts and evaluation configuration. Analysis saves comparison.json/summary.csv and SVG. Any error writes failure.txt with the original traceback; don't overwrite failures or silently omit records/classes.

Acceptance concerns correct execution, fixed information boundary, complete population and valid metric/selection, not method victory. Qualified negative results remain results. Broad intervals are inconclusive. If late fusion explains the apparent gain, narrow FiLM-specific claims; don't weaken its tuning. Full external-competitor qualification and actual industrial effect remain unestablished by this repository's software tests.

## Return and sync

Return exact repo/code version and relevant local diff; environment/GPU; inventory/ontology/physical field sources; completed/failed/invalid runs; all source trial costs and selected settings; per-system record metrics and paired intervals; active parameter counts; anomaly/correction list; Contribution–Experiment–Result mapping and only the next necessary experiment. Large checkpoints/raw data stay in existing local storage with accessible references; commit only code/config/small necessary summaries via PR to dev. Do not write Results before valid outputs. Do not merge failing PRs, change master/main, force push or introduce a second runtime. Environment repairs preserving the protocol are allowed; scientific protocol changes return to design before reuse of old results.

# Selected-state prediction export and explicit simulation demos

`IdIncludedDataset` returns its existing per-file `window_id` alongside `file_id`.
Observations, labels, selection and sampling are unchanged.

The exporter reuses the configuration authority and Data/Model/Task factories,
restores the selected checkpoint and retains native test batches. It writes file/window
identity, raw intervals, logits, target and checkpoint; independently computes complete
accuracy and macro-F1; and compares the original per-seed metrics. The linear case also
checks the NumPy mean-linear formula at `atol=1e-6, rtol=1e-5`.

## Explicit simulation, not substituted real data

The author authorized simulation for software execution on 2026-11-04.
`prepare_simulated_phm.py` writes 27 two-channel CSV records: three signal classes,
three conditions, three simulated units per class. Harmonics, decaying pulses, noise,
and a declared class-dependent calibration offset are analytical constructions, not
calibrated physical bearings. Metadata explicitly records `Data_Origin=simulation`.
The existing strict Dummy_Data CSV reader is reused; no reader synthesizes a fallback.

```bash
python -m scripts.prepare_simulated_phm --output data/simulated_phm
phmfactory --config configs/experiments/phmfactory/simulated_dg_linear.yaml
phmfactory --config configs/experiments/phmfactory/simulated_dg_timesnet.yaml
```

Both configurations use source conditions 0/1, source-only file-grouped validation,
frozen evaluation on condition 2, seed 17, and two epochs. Each has 72 test windows
from nine simulated records. These configurations do not change Experiment A's
MFPT population, windows, three-seed budget, or current baseline-valid status.

Replay the actual returned selected checkpoint and original one-row per-seed metrics:

```bash
python -m scripts.paper.export_mfpt_predictions \
  --profile simulation \
  --config configs/experiments/phmfactory/simulated_dg_linear.yaml \
  --checkpoint "$BEST_CHECKPOINT" \
  --reported-metrics "$ITERATION_METRICS" \
  --split-manifest "$SOURCE_SPLIT" \
  --seed 17 --output "$NEW_EXPORT_DIR"
```

TimesNet uses its own configuration and selected checkpoint; batches are not reshuffled.
Omit `--profile simulation` for the unchanged default MFPT reference, with its original
configuration, inputs and result files. The exporter never trains or selects a model.

The existing core workflow runs these public commands and two separate interventions:
cyclically permute only evaluator labels, or scale only target observations. Source
signals/labels and training budget are unchanged. It compares selected state tensors,
complete predictions, split membership and independently recomputed metrics. Simulation
execution, existing CI and real-data qualification remain separate outcomes; exact
run results are recorded in the PR, not inferred from this implementation description.
No new test file, mock, test framework, runtime, data reader or result manager is added.

## Setting-contract refinement

The simulated DG configuration now states population, access, timing/state, selection,
budget and estimator next to the fields already consumed by the runtime. It remains a
file-level, seen-simulated-unit/new-condition experiment, not asset-isolated transfer.
No new `task.protocol` fields or registry status are introduced by comments.

The existing simulation CI step also compares optimizer state, learning-rate scheduler
state and source-validation checkpoint scores under the two existing target-only
interventions. A direct check of the retained seed-17 checkpoints passed all four added
optimizer/selection comparisons without fitting again. Checkpoint RNG state was not
recorded; the TimesNet configuration has no corresponding intervention arms. These
limits are explicit in the output instead of declaring every invariant passed.

PR status and scientific qualification are separate. The simulation branch may carry
configuration-specific execution evidence before merge, but it does not become a
capability of `dev` or a real-data `baseline_valid` result until the applicable conditions
are met. The exact MFPT protocol and historical results are unchanged.


## CFM group-disjoint test evaluation

A separate `simulated_cfm_holdout.yaml` now trains, selects and evaluates on disjoint
simulated assets, using the existing read-only split implementation. The public
`DataSplitConfig` accepts that implementation's `manifest_mode` and `partition_map`;
`generative` is admitted to its partition lifecycle. The analytical data preparation
command writes an optional fixed split with `--cfm-split`.

Pipeline 06 uses training metadata for model construction and metadata-derived
sampling conditions. It accepts complete test-reference evaluation only for grouped
metadata. The existing evaluation artifact records this split without promoting the
run to `baseline_valid`. No new loader, trainer, split algorithm, hash scheme or test
file was introduced. Original train/validation smoke semantics are retained.

Actual seed-17 simulation executions: three two-epoch fits (original, test-label
permutation, test-waveform replacement), three strict checkpoint/sample stages and
three complete 72-window test evaluations with nine generated samples. The initial
evaluation stopped at the old artifact writer's blanket prohibition on test; that
failed record was preserved before correcting the writer and rerunning only evaluation.
An interrupted subprocess launch was resumed only for its unstarted evaluation stage.

Both interventions preserved selected model/optimizer/scheduler state, selection
scores, processed-training normalization, post-train/post-sample RNG states and sampled
tensors exactly. Independent NumPy window construction and two metric calculations
matched the stored values within the predeclared tolerances. A legacy overlapping
split with `eval_split=test` was explicitly rejected. All evidence is simulation-only;
MFPT/PU protocols, existing results and registry support levels remain unchanged.

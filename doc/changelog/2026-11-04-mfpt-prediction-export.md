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

# MFPT Experiment A: selected-state prediction export

This increment adds an Experiment A export command, not a new training runtime.
`IdIncludedDataset` now returns the already existing per-file `window_id` alongside
`file_id`; observations, labels, selection and sampling are unchanged.

The exporter reuses the configuration authority and Data/Model/Task factories. It
restores the explicitly supplied selected checkpoint and preserves native test batches.
It writes each file/window identity, raw interval, logits, target and checkpoint,
recomputes accuracy and full-population macro-F1, and compares the original per-seed
metrics. For GlobalAverageLinear it also checks the explicit NumPy mean-linear formula
at `atol=1e-6, rtol=1e-5`. Original split and result files remain unchanged.

The fixed TimesNet substitution from Experiment A is available at
`configs/experiments/phmfactory/mfpt_timesnet.yaml`; it inherits the reference data,
split, optimizer, seed and checkpoint policy. It is not a tuned performance baseline.

After running Experiment A through the existing public command, use the actual returned
checkpoint and per-seed metric file, and the original split file:

```bash
python -m scripts.paper.export_mfpt_predictions \
  --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml \
  --override "data.data_dir=$MFPT_ROOT" \
  --checkpoint "$BEST_CHECKPOINT" \
  --reported-metrics "$ITERATION_METRICS" \
  --split-manifest "$SOURCE_SPLIT" \
  --seed 17 \
  --output "$NEW_EXPORT_DIR"
```

Repeat export for the already fitted seeds 18 and 19 with their own returned artifacts;
do not retrain for export. TimesNet uses its fixed configuration and checkpoints.
The TimesNet replay retains native batch membership because period selection is batch
dependent. Per-record window accuracy is descriptive, not independent-device evidence.

Current validation: Python compilation and CLI help completed locally. Full-factory
replay, MFPT training, metric agreement on real data and the TimesNet demo have not run:
the original checkout, MAT files and checkpoints are not mounted, and the container's
GitHub connection fails DNS resolution. No registry support level is promoted. Existing
CI is separate from this real-data limitation. No new test file or test fixture is added.

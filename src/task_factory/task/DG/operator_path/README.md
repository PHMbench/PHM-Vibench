# Bounded operator-path execution

This research task uses the existing `P07OperatorPath.OperatorNet`. The proposed
objective is cross-entropy plus the nearest-operator residual; the concentration
control uses the same architecture with its declared allocation penalty. Model
selection always uses unregularized source-validation cross-entropy.

The packaged configuration is `configs/experiments/p07/operator_path.yaml`. The
public command is `phmfactory research <phase> --config <yaml> --data <input>
--output <new-directory>`. Every phase refuses existing output directories.

| Phase | Explicit input | Result |
| --- | --- | --- |
| `export-source` | Existing record-spec JSON containing only train/validation records | Source-only NPZ and window-to-record table |
| `export` | Existing complete train/validation/test record-spec JSON | Complete split NPZ |
| `source-check` | Source-only NPZ | Declared source group/domain counts and normalization check |
| `data-check` | Complete split NPZ | Within-export split checks |
| `tune` | Source-only NPZ | Equal candidate/seed search per arm and `selection.json` |
| `train` | Same source-only NPZ plus `--selection <selection.json>` | Source-selected checkpoints; no target evaluation |
| `overfit` | Source-only NPZ | Existing CE-only tiny-source optimization diagnostic |
| `evaluate` | Complete split NPZ plus `--checkpoint <model.pt>` | Entire-partition utility, prespecified-cohort extraction and independent replay |
| `replay` | Complete split NPZ, `--checkpoint <model.pt>`, `--selection <extractions.csv>` | Re-execution of saved paths without new search |

Tune/train/overfit reject a target-containing NPZ before reading its signal array.
Selected checkpoints record train/validation unit identities, labels, domains and
normalization dtype. Evaluation requires the same source partitions; it never
refits normalization. Historical checkpoints without these fields fail explicitly.

`comparison: competitive` uses each arm's separately selected configuration.
`comparison: mechanism` gives sparse, concentration and unbounded controls the
proposed arm's selected optimizer configuration. The packaged 18-candidate search
is the existing pilot range, not evidence that any baseline is already qualified.

Path execution retains coefficient-one branches. The 216 candidates and 238-call
budget are unchanged. Replay validates both complete extraction cohorts, input
identity, saved predictions/discrepancies and the original acceptance inequality.
Its extra verification forwards are reported separately from extraction calls.
Accepted-path fidelity is conditional agreement with the network, not correctness
against fault labels. Whole-partition diagnostic utility, cohort coverage,
accepted-but-wrong diagnoses and analytical sufficiency are reported separately.
Analytical rates are inapplicable to learned/unbounded controls.

Fixture tests and successful execution establish software semantics only. Raw-data
provenance, baseline qualification and hypotheses require their own experiments.

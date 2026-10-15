# Physical-symbolic diagnosis

The research entrypoint executes
`src.task_factory.task.classification.symbolic_diagnosis.execute` with an already
resolved config. Options live in `task.symbolic_diagnosis`; no paper checkout is
imported. This is the existing non-gradient prototype and exact affine
certificate protocol, not a Lightning training task.

| Phase | Explicit inputs | Direct outputs |
| --- | --- | --- |
| `synthetic` | `symbolic_fixture.yaml`, new output directory | generated NPZ, exact head state, per-view certificates/predictions, unit bootstrap, activation and representation contrasts |
| `tune` | `source_only_baselines.yaml`, source-only train/val NPZ, new output directory | source-only SVM grid, selected configurations, source-fitted scaling and prototype heads |
| `evaluate` | same baseline config, full evaluation NPZ, completed tune directory as checkpoint, new output directory | every test prediction and fixed-class descriptive metrics; zero new fits |
| `panel` | config with explicit `orders`, `source_condition`, `target_condition`; train/val NPZ; new output directory | unpaired acquisition/unit statistics; no classifier or pointwise certificate |

The public command form is `phmfactory research <phase> --config <yaml>
--output <new-directory>`, adding `--data <NPZ>` and `--checkpoint
<completed-tune-directory>` as required above. Paths are never inferred from a
paper checkout. Existing output directories are rejected.

Public `tune` requires a separate NPZ containing only the declared source
condition and exactly train/val splits. It reads split and condition metadata
before loading any signal array; full train/val/test archives are rejected.
`evaluate` uses a separate complete protocol NPZ and requires the source train/val
metadata multiset to match the tuned checkpoint, including split, unit, label,
measured speed, optional acquisition/condition fields and repeat multiplicity.
Row reordering is allowed. This binds declared membership, not waveform content.
Legacy full-archive tuning checkpoints are not accepted by the public evaluation
route. The underlying mathematical comparison functions retain their original
full-protocol defaults for parity checks.

The supplied orders `[3.2, 4.8, 6.4]` and source speed `30 Hz` belong to the
existing synthetic protocol. They do not qualify real bearing geometry. Real
panel execution additionally requires independently reviewed identity,
acquisition, channel, sampling rate, fault-order mapping and operating condition.
Each NPZ row represents the declared complete acquisition; no cropping, padding,
window multiplication or manufactured pairing is performed here.

A banded representation and a width-times-four sensitivity contrast exist. A
separate repair optimizer does not. A certificate preserves the fixed model's
decision under supplied symbols; a stable or certified prediction can be wrong.
The output keeps activation, uncertified cases, decision changes and label errors
separate. Synthetic fixtures establish software behavior only.

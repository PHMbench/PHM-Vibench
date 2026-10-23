# Expert-role interventions

This package executes the prospective M05 method using the existing
`MoE.M_05_FixedRouteMoE` model. It does not replay or reinterpret historical
M04/G050 runs. A software fixture does not qualify a physical probe, independent
sampling design, or unseen-condition experiment.

The public research command resolves configuration with `analyze_config()` and
calls `execute(config, phase=..., output=..., data=..., checkpoint=..., device=...)`.
The config is `configs/experiments/p04/expert_intervention.yaml`; all outputs must
use a new directory. There is no paper import, repository lookup, or implicit
checkpoint selection. Only `train` and `tune` perform optimization.

| Phase | Required inputs | Computation and output |
| --- | --- | --- |
| `prepare` | `task.expert_intervention.manifest`, `raw_root`, `roles`, `window`, `strength`, `partitions`, `class_names` | Check complete manifest identities before reading selected partitions; explicit order-band components and equal-L2 target/control attenuation; `feature_pack.npz` and the preparation description |
| `sanity` | `--data` and explicit device | Shape, split and source-only normalization checks; no model fit |
| `train` | Source-only `--data` with train/val, task settings and explicit device | CE plus configured route balance; minimum group-averaged source validation CE selects `model.pt`; target-containing packs are rejected before feature reading |
| `tune` | Same, plus declared arms/grid | Six default candidates per generic/aligned/shuffled arm, equal epochs, seed and selection rule; generic must beat the training-prior predictor; `selected_configs.json` |
| `evaluate` | Match/test-only `--data`, `--checkpoint` and explicit device | Saved normalization and model, checkpoint/source identity and schema checks, held-out source role fitting, frozen-gate paired intervention responses; audit CSV, clean probabilities, signed contrast and group-averaged test accuracy |

`task.expert_intervention` owns width, arm, optimizer parameters and training
budget. Expert, feature and class axes come from the explicitly supplied pack;
if declared in `model`, the dimensions must agree. Evaluation obtains model
identity, width and arm from its explicit checkpoint, never from a new fit.
The per-run `alpha` must already include any across-model/seed multiplicity
correction. Within each run the existing bound covers the domain-role cells.
The historical name `certified_positive` denotes a signed lower bound above
zero; it is not a faithfulness success flag. Restoring a useful expert can
produce a negative signed contrast.

Prepared arrays use `raw[N,D]`, `views[N,K,D]`, `compatibility[N,K]`,
`probe_raw/control_raw[N,R=K,D]`, `probe_views/control_views[N,R=K,K,D]`, and
one-dimensional `labels/group/split/domain`. The complete qualification pack uses
`train/val/match/test`; source-only train/tune and match/test evaluation packs
are prepared separately from explicitly selected manifest partitions. All
closed-set labels occur in train. Each `group`
belongs to one domain, label and split. The `specimen[N]` names the
physical specimen across domains and is globally split-disjoint. Inferential
bounds additionally require an explicit independent-group assertion and forbid
multiple aggregation groups for one specimen in a domain. Descriptive analyses
also require one aggregation group per specimen/domain cell. These assertions do
not establish acquisition independence from data alone. Aggregation remains the
existing equal-window mean within group and equal-group mean between groups.
Public train/tune/evaluate require `specimen`, `role_names[K]` and
`class_names[C]` in label-index order. The checkpoint saves these names and
train/val identities; evaluation rejects schema changes and source/evaluation
specimen overlap. The standalone `load_pack` default still supports the original
four-partition shape for historical parity and non-training qualification.

The declared `h5` preparation reader accepts exactly `[samples,channels,1]` with
explicit dataset key and zero-based channel; it never flattens or crops signals.
MAT variable and existing `RM_027_PU` readers remain available. Sampling rate,
rpm, target/control order bands, label and identity are supplied by the manifest
and roles JSON. Duplicate declared acquisitions are rejected before signal
reading, including attempts to alias a recording with another group ID.
Matched spectral edits alone do not certify physical validity.

`replacement` evaluates the original fixed clean-reference replacement. Optional
`deletion` evaluates the existing renormalized-deletion identity with the same
clean route and labels; it rejects a route mass of one. Both produce the same
signed target-minus-mean-mismatched contrast and retain the same blinded role
assignment. No routing-only intervention is defined.

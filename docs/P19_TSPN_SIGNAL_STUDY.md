# P19 execution authority — PHMFactory + external time-series datasets

## Scope

Paper: `AI4Engineering-L/P-19-Deterministic-Representation-Gain-tpami`.

**Unique execution owner for all new P19 experiments:** `PHMbench/PHM-Vibench` (PHMFactory).

`liq22/phm-agent-benchmark` is **not** an execution dependency for the active TSPN/signal-program paper. Its completed P19 frozen-bank pilot and G6 source attribution remain historical evidence only. Do not add new P19 models, data adapters, training stages, correction code, metrics, plots, or experiment Goals there.

The active paper studies interpretable time-series signal processing on top of TSPN. New experiments must therefore use PHMFactory's existing data/model/task/trainer/runtime path and dataset-specific readers.

## Repository ownership

- `PHMbench/PHM-Vibench`
  - TSPN and interpretable signal-processing implementations.
  - Dataset readers/adapters and metadata contracts.
  - Public configuration and preflight.
  - Training, checkpoint selection, inference/export, metrics, statistics, plotting.
  - P19 experiment configuration and executable scripts.
- `AI4Engineering-L/P-19-Deterministic-Representation-Gain-tpami`
  - TeX manuscript, theory/formulation, method description, claim/evidence table and paper-facing references.
- `liq22/phm-agent-benchmark`
  - read-only historical E1–E6 P19 pilot evidence; no active P19 implementation.

Do not cross-import Agent Benchmark into PHMFactory.

## Data roots

PHM data already available locally remain read-only:

```text
/home/user/data/PHMbenchdata/PHM-Vibench/
```

All additional general time-series datasets live under:

```bash
export TS_DATA_ROOT="$HOME/data/timeseries"
mkdir -p "$TS_DATA_ROOT"
```

The terminal prompt `$` is not part of the path.

Prepared UCI HAR currently belongs under:

```text
$HOME/data/timeseries/uci-har/raw/UCI HAR Dataset
```

Other time-series datasets are added only when required by a retained paper claim. Their raw payloads stay outside Git.

## Scientific study boundary

The new study is distinct from the historical ResNet1D/TCN frozen-bank pilot.

Primary scientific question:

> Do explicit modulation, periodicity and impulsiveness signal programs add useful corrections beyond a competent PHMFactory TSPN and matched native-feature readouts?

The baseline qualification target is a predeclared diagnostic-accuracy floor. A requested 80% threshold is a qualification criterion, **not a guaranteed result**. Report source qualification and held-out accuracy separately. Never change a split, drop a seed, extend training after test inspection, or replace a dataset merely to cross 80%.

The core correction contrasts are:

1. frozen TSPN;
2. intercept-only correction;
3. equal-dimensional native TSPN feature correction;
4. full-native-feature correction;
5. named signal-program correction.

Every condition shares the same TSPN checkpoint, native window population, calibration, grow/select roles and readout objective within a dataset.

## Implementation location

Use existing PHMFactory owners before adding code:

- public config: `phmfactory/config.py`;
- data: `src/data_factory/`;
- TSPN/signal operators: `src/model_factory/X_model/`;
- task and metrics: `src/task_factory/`;
- runtime: `src/runtime/`;
- experiment configs: `configs/experiments/p19/`;
- narrow paper helpers when a public runtime hook is insufficient: `scripts/paper/p19/`.

A helper may export selected-checkpoint logits/features or perform the fixed correction analysis, but it must reuse PHMFactory data/model/task construction. It must not create a second loader, trainer, split resolver, label ontology or result authority.

The active P19 implementation presently located in Agent Benchmark is migration reference only. Port the minimum scientifically necessary routines into PHMFactory, verify numerical equivalence on a small legal/synthetic fixture, then run all new experiments only from PHMFactory.

## Dataset A — Paderborn / PHM

Use the existing local Paderborn source through PHMFactory.

Preserve the already declared physical-unit split semantics unless a new protocol is explicitly approved. The statistical unit is the physical bearing; windows and training seeds do not increase the independent-unit count.

The active TSPN study keeps interpretable signal processing as the central mechanism. Paderborn-specific analysis bands may be used only for the Paderborn protocol and must be recorded in the visible experiment config.

Before a long run:

```bash
cd "$PHMFACTORY_ROOT"
git status --short
git rev-parse HEAD
phmfactory doctor
phmfactory preflight --config configs/experiments/p19/<paderborn-config>.yaml
```

Then execute through the public PHMFactory entry, not Agent Benchmark.

## Dataset B — UCI HAR external time-series replication

UCI HAR is the first external general time-series dataset because it is already prepared locally. Its official subject train/test boundary must be preserved.

Do **not** copy Paderborn's:

- 64 kHz sampling rate;
- 4,096-sample window;
- vibration-channel meaning;
- frequency bands;
- class head;
- physical-bearing grouping.

Instead, implement or reuse a PHMFactory dataset reader/adapter whose metadata preserves subject identity, channel order, sampling rate and official split. Allocate fit/tune/design only inside the official training subjects; keep official test subjects untouched until all model/operator/readout settings are frozen.

If a PHMFactory-native HAR reader does not yet exist, add it using `docs/custom_dataset.md`:

```text
src/data_factory/reader/<HAR_name>.py
metadata with subject/group and class fields
focused reader/adapter tests
configs/experiments/p19/<har-config>.yaml
```

The signal-program specification must be task-specific. Use the actual sampling rate and observable frequency grid; do not transfer Paderborn hertz bands. Any normalized-frequency design must be declared before HAR test scoring.

The independent unit is the subject, not the window.

## Additional time-series datasets

PAMAP2, Sleep-EDF, PTB-XL, Speech Commands or another dataset may be added under `$HOME/data/timeseries` only if one is needed to support a retained cross-domain claim after Paderborn + HAR evidence is inspected.

Do not create a five-dataset checklist in advance. Each added dataset requires:

- source/license verification;
- PHMFactory reader/metadata binding;
- native sampling/channel/class semantics;
- independent-unit definition;
- train/tune/design/test separation;
- dataset-specific signal-program coordinates.

## Minimal execution sequence

### G0 — PHMFactory migration and smoke

1. Inspect current PHMFactory `dev`, active PRs and working tree.
2. Locate the current TSPN model and all operators that can be reused directly.
3. Port only the missing P19 signal-program/correction routines from the historical Agent Benchmark implementation.
4. Add focused CPU tests for signal-program values, exact logit reconstruction, zero-correction recovery, grouped weighting and saved-state replay.
5. Run a real PHMFactory model/data forward-backward/checkpoint smoke in the existing native environment.
6. Do not run test performance during migration validation.

### G1 — Paderborn TSPN qualification and correction

1. Freeze configuration, physical roles, TSPN training budget and seeds before test scoring.
2. Train/select/calibrate TSPN through PHMFactory.
3. Fit the four correction controls on the declared source grow units.
4. Freeze probability projection and source acceptance before test scoring.
5. Evaluate the full fixed test population.
6. Save per-unit Brier, accuracy, macro-F1, raw/projected/guarded predictions, coefficients and named logit contributions.
7. Preserve qualification failures, zero strength, rejected corrections and adverse effects.

### G2 — HAR replication

Only after the PHMFactory HAR binding and protocol are complete:

1. verify official subject split and PHMFactory metadata;
2. freeze task-specific TSPN and signal-program settings;
3. run the same scientific contrasts;
4. aggregate uncertainty over subjects;
5. retain all signs and failures.

HAR is an external replication of the mechanism, not a search for a favorable dataset.

### G3 — paper-facing analysis

Plots and tables read saved PHMFactory artifacts only:

```text
PHMFactory run
-> raw predictions / unit metrics / saved states
-> analysis CSV
-> paper figure/table
```

No plotting stage trains or performs model inference.

## Required artifacts

For each dataset/seed keep the existing PHMFactory run outputs plus the minimum P19 additions:

- fully resolved configuration and actual command;
- selected checkpoint;
- source/test unit assignments;
- class/channel/sampling metadata;
- baseline qualification;
- calibration state;
- raw/projected/guarded predictions;
- unit-level Brier/accuracy/macro-F1;
- readout coefficients and signal feature names;
- exact added-logit reconstruction check;
- failure record when applicable;
- measured runtime fields only when actually measured.

Raw datasets stay outside Git.

## Acceptance

Software completion and scientific support are separate.

A valid experiment may show:

- baseline < 80%;
- signal correction = baseline;
- signal correction worse than native control;
- source guard rejection;
- positive signal correction.

All are valid outcomes if the protocol is followed.

A positive operator-specific claim requires real independent-unit evidence beyond both the frozen TSPN and the matched native-feature control. A broad time-series claim additionally requires the independent external dataset.

## Stop and sync

Validated implementation/configuration changes go to PHMFactory `dev` through the normal PR process. The paper repository records only the current protocol, artifact locations and evidence interpretation.

Do not add new P19 implementation to Agent Benchmark. Do not rerun the historical E1–E6 pilot.

Stop after the smallest batch needed to resolve the current claim and hand the actual artifacts back to the paper evidence stage.

# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory logo" width="300"/>

  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>Configuration-first, fail-fast PHM experiments for industrial signals.</strong></p>
  <p><em>Declare one experiment; execute that experiment.</em></p>

  <p>
    <img src="https://img.shields.io/badge/status-release%20blocked-critical" alt="Release blocked pending current-source baseline validation"/>
    <img src="https://img.shields.io/badge/version-0.3.0rc1-blue" alt="Version 0.3.0rc1"/>
    <img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB" alt="Python 3.10 or newer"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="Core quality gates"/></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>

  <p>
    <a href="#quick-start">Quick start</a> •
    <a href="#scientific-contract">Scientific contract</a> •
    <a href="#how-it-works">How it works</a> •
    <a href="#support-boundary">Support boundary</a> •
    <a href="#documentation">Documentation</a>
  </p>
</div>

---

PHMFactory is a modular research runtime for fault diagnosis and related PHM experiments.
A visible configuration connects data, model, task objective, trainer, checkpoint
selection, evaluation, and direct result paths.

The repository is [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench); the
project and Python package are named **PHMFactory** and `phmfactory`.

> **Current source state.** The source version is `0.3.0rc1`, but release readiness is
> blocked. The offline Dummy path is maintained. The MFPT transparent experiment is a
> reviewed `smoke_only` candidate pending current-source scientific requalification.
> There is currently no current-source `baseline_valid` registry row, RC1 tag, GitHub
> Release, or package-index publication.

The compact project authority is [`CORE.md`](CORE.md).

## Quick start

The first run is fully offline and uses repository-shipped Dummy data.

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .

phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

A successful demo prints direct result locations:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

These paths are the user result authority. A manifest, attestation, evidence index,
receipt, ledger, or hash is not required for success.

The Dummy demo proves that the maintained installation and runtime can complete one
bounded experiment. It does **not** prove a real-data benchmark, strong accuracy,
state-of-the-art performance, or arbitrary compatibility among repository components.

See the complete [Quickstart](docs/quickstart.md) and
[Installation guide](docs/installation.md).

## Scientific contract

The governing invariant is:

```text
requested experiment = executed experiment
```

Represent an experiment as:

$$
\mathcal E=(\mathcal D,\Pi,f_\theta,\mathcal L,\widehat R),
$$

where $\mathcal D$ is the data population, $\Pi$ the protocol, $f_\theta$ the actual
model, $\mathcal L$ the optimized objective, and $\widehat R$ the reported estimator.
Every term must match the visible request.

PHMFactory therefore rejects, rather than silently repairs:

- missing or malformed configuration;
- unavailable explicit devices;
- impossible splits or target domains;
- invalid labels, reader outputs, patch sizes, metrics, or regularizers;
- missing or incompatible selected checkpoints;
- empty, incomplete, non-scalar, NaN, or Inf evaluation results;
- an alternate backend or data source after the requested one fails.

Training may be stochastic. Validation and test must still define a reproducible
estimator.

## How it works

```text
YAML or preset
    ↓
one resolved configuration
    ↓
canonical Pipeline
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → selected checkpoint → test → complete finite metrics
    ↓
direct result paths
```

| Boundary | Owns | Must not do |
| --- | --- | --- |
| **Data Factory** | reader, metadata, selected IDs, datasets, samplers, loaders | repair model, task, device, or metric configuration |
| **Model Factory** | model identity, construction, explicit weights | select splits or move the model to a device |
| **Task Factory** | task identity, objective, metric lifecycle | control hardware or checkpoint selection |
| **Trainer Factory** | device, callbacks, checkpoint selection, fit/test lifecycle | invent missing data or task semantics |
| **Pipeline** | orchestration, success gating, direct result locations | silently repair any Factory input |

Replacing one compatible component should require changing that component and its
configuration, not the other factories or the public command router.

## Configuration

Maintained experiments use a top-level Pipeline and five logical blocks:

```yaml
pipeline: "Pipeline_01_Fault_Diagnosis"

environment:  # output root, seed, repeat count
  ...
data:         # metadata, raw data, windows, sampling
  ...
model:        # model identity and parameters
  ...
task:         # objective, metrics, optimizer/scheduler
  ...
trainer:      # device, epochs, checkpoint and logging lifecycle
  ...
```

Use the same visible inputs for preflight and execution:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

Machine-specific YAML is used only when passed explicitly with `--local-config`.
Configuration composition is documented in [`configs/README.md`](configs/README.md).

## Support boundary

PHMFactory distinguishes:

```text
discoverable       source or registry entry exists
runnable           a reviewed execution path exists
execution-verified the exact command has bounded current execution evidence
baseline-valid     the exact complete experiment passed its current scientific protocol
```

`baseline-valid` is configuration-specific. It cannot be inferred from importability,
source presence, another configuration, or historical results.

Current authorities:

- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Configuration registry](configs/config_registry.csv)
- [Known limitations](KNOWN_LIMITATIONS.md)
- [Release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)

The MFPT + `GlobalAverageLinear` configuration remains a transparent real-data candidate.
Its historical three-seed result is not treated as current-source promotion evidence until
its unchanged protocol satisfies the current metric and requalification gates.

## Public entrypoints

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

Use `phmfactory` for normal work. `python main.py` is a compatibility launcher.

Bounded utility commands:

```bash
phmfactory doctor
phmfactory preflight --config <preset-or-yaml>
phmfactory demo
phmfactory data --help
```

## Documentation

| Goal | Start here |
| --- | --- |
| Install and complete the first run | [Quickstart](docs/quickstart.md) |
| Understand the governing engineering/scientific contract | [Core contract](CORE.md) |
| Configure an experiment | [Configuration guide](configs/README.md) |
| Connect local data | [Data layout](data/README.md) and [custom dataset guide](docs/custom_dataset.md) |
| Add or select a model | [Model Factory](src/model_factory/README.md) |
| Add or select a task | [Task Factory](src/task_factory/README.md) |
| Use the optional browser workspace | [Streamlit workspace](apps/streamlit/README.md) |
| Extend the repository | [Contributing](CONTRIBUTING.md) |
| Inspect release blockers | [Release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) |

The complete navigation is in [`docs/index.md`](docs/index.md).

## Development rules

Development follows Occam's razor:

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

Do not add consumerless hashes, silent fallback, broad future-oriented abstractions,
Factory/Manager/Registry nesting, or another config/runtime/result authority. One PR
should protect one primary invariant and provide one user-observable outcome.

Routine PRs start from current `dev` and target `dev`. Read [`CORE.md`](CORE.md),
[`AGENTS.md`](AGENTS.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md) before broad changes.

## Citation and license

PHMFactory is distributed under the [Apache License 2.0](LICENSE). Citation metadata are
provided in [`CITATION.cff`](CITATION.cff). Dataset and third-party component licenses
remain separate and must be reviewed before redistribution or commercial use.

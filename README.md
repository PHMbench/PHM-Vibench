# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory logo" width="300"/>

  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>Configuration-first, fail-fast PHM experiments for industrial signals.</strong></p>
  <p><em>One configuration from data selection to evaluation, without hidden scientific fallbacks.</em></p>

  <p>
    <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha"/>
    <img src="https://img.shields.io/badge/version-0.3.0.dev0-blue" alt="Version 0.3.0.dev0"/>
    <img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB" alt="Python 3.10 or newer"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="Core quality gates"/></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>

  <p>
    <a href="#quick-start">Quick start</a> •
    <a href="#why-phmfactory">Why PHMFactory</a> •
    <a href="#how-it-works">How it works</a> •
    <a href="#choose-your-path">Documentation</a> •
    <a href="#support-boundary">Support boundary</a> •
    <a href="#contributing-and-citation">Contribute</a>
  </p>
</div>

---

> **Repository identity.** The project and Python package are named
> **PHMFactory**. During the v0.3 pre-release, the GitHub repository remains
> [`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench). Use this
> repository URL until a future rename is explicitly announced.

PHMFactory is a modular research framework for fault diagnosis and related PHM
experiments on industrial signals. A single resolved configuration connects the data,
model, task objective, trainer, checkpoint, evaluation, and user-visible results.

The governing invariant is:

```text
requested experiment = executed experiment
```

A run must fail clearly when its requested data, task, device, objective, checkpoint, or
evaluation cannot be executed as declared. PHMFactory does not silently replace the
experiment with an easier one.

## Why PHMFactory

Industrial PHM repositories often contain strong algorithms but weak experimental
boundaries. The resulting code may run while executing a different data split, task,
device, objective, or estimator than the user intended.

| Common failure in PHM experimentation | PHMFactory response |
| --- | --- |
| Configuration defaults silently change the experiment | Explicit configuration resolution and fail-fast validation |
| Data, model, task, and trainer logic are entangled | Four factories with narrow, reviewable responsibilities |
| Evaluation depends on uncontrolled random sampling | Deterministic validation and test behavior on maintained paths |
| A source file is mistaken for a supported capability | Generated support tables distinguish discovery, execution, and scientific evidence |
| Replacing one component requires editing the runtime | Component selection remains configuration-first |

Two practical design rules follow:

```text
Replace one module
→ change that module and its configuration
```

```text
Training may be stochastic
→ evaluation must still be a defined estimator
```

<details>
<summary><strong>What PHMFactory is—and is not</strong></summary>

PHMFactory is intended to provide:

- one public configuration-first execution path;
- explicit data, model, task, and trainer boundaries;
- actionable failures rather than silent fallback;
- maintained smoke configurations and generated support documentation;
- one scientifically closed real-data reference with an explicit claim boundary;
- a common runtime for CLI and optional Streamlit usage.

PHMFactory does not claim that every implementation in the repository is mutually
compatible, benchmark-valid, state of the art, or ready for redistribution. Those claims
require an explicitly maintained configuration and a scientifically closed protocol.

</details>

## Quick start

The first run is fully offline. It uses repository-shipped synthetic data and does not
download an external dataset.

### 1. Install

```bash
git clone https://github.com/PHMbench/PHM-Vibench.git
cd PHM-Vibench

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10 or newer is required. CPU-only and GPU-specific installation notes are in
[Installation](docs/installation.md).

### 2. Diagnose, preflight, and run

```bash
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

The three commands have separate purposes:

```text
doctor
→ checks the installed environment and repository resources

preflight
→ resolves and validates the exact configuration without training

demo
→ executes one bounded CPU experiment through data → model → task → trainer
```

A successful run writes below:

```text
results/demo/dummy_dg_smoke/
```

The terminal prints the compatibility run-record path when that optional diagnostic is
available. A `run_manifest=unavailable` warning does not invalidate completed fitting,
best-checkpoint restoration, evaluation, or finite metrics. For expected output and
failure diagnosis, see [Quickstart](docs/quickstart.md).

### What the offline demo proves

It proves that the package can be installed and that the maintained public runtime can
complete one bounded experiment. It does **not** prove real-data benchmark validity,
algorithm superiority, or universal compatibility across repository components.

## How it works

```text
YAML or preset
    ↓
resolved configuration
    ↓
canonical Pipeline
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → best checkpoint → evaluation → finite metrics
    ↓
user-visible result path
```

### Factory responsibilities

| Boundary | Owns | Must not do |
| --- | --- | --- |
| **Data Factory** | readers, metadata, selected IDs, datasets, samplers, loaders | repair model or task configuration |
| **Model Factory** | model identity, construction, explicitly requested weights | select data splits or move the model to a device |
| **Task Factory** | task identity, objective, metric lifecycle | control hardware or rewrite the requested task |
| **Trainer Factory** | device, callbacks, checkpoints, loggers, training and evaluation lifecycle | invent missing task or data semantics |
| **Pipeline** | orchestration, success gating, result location | silently repair any factory input |

This division is deliberately narrow. New datasets, models, tasks, and trainers should
normally extend their own factory rather than add special cases to `main.py`.

## Core capabilities

| Capability | User-visible behavior |
| --- | --- |
| **Configuration-first execution** | The same resolved configuration drives CLI, module, and compatibility entrypoints. |
| **Fail-fast scientific semantics** | Invalid labels, unavailable devices, impossible splits, missing checkpoints, and invalid metrics stop the run. |
| **Deterministic evaluation boundaries** | Maintained validation and test paths do not depend on uncontrolled patch or augmentation randomness. |
| **Offline first-run path** | `doctor`, `preflight`, and `demo` work without downloading an external dataset. |
| **Modular replacement** | Data, model, task, and trainer choices remain explicit and independently reviewable. |
| **Configuration-specific evidence** | Generated support authority separates bounded smokes from an exact `baseline_valid` protocol. |
| **One runtime, optional interfaces** | The CLI is authoritative; Streamlit adapts the same command rather than creating a second training system. |

## Choose your path

| Your goal | Start here |
| --- | --- |
| Understand the first run and its outputs | [Quickstart](docs/quickstart.md) |
| Install on CPU, GPU, Linux, macOS, or Windows | [Installation](docs/installation.md) |
| Run an existing maintained experiment | [Configuration guide](configs/README.md) |
| Prepare and run the first real-data reference | [MFPT baseline](configs/baselines/01_mfpt/README.md) |
| Connect local PHM data | [Data layout](data/README.md) and [custom dataset guide](docs/custom_dataset.md) |
| Select or add a model | [Model Factory](src/model_factory/README.md) |
| Select or add a task | [Task Factory](src/task_factory/README.md) |
| Use the browser workspace | [Streamlit workspace](apps/streamlit/README.md) |
| Extend or maintain the framework | [Developer guide](docs/developer_guide.md) |
| Inspect the exact maintained surface | [Supported combinations](SUPPORTED_COMBINATIONS.md) |

The complete documentation map is in [docs/index.md](docs/index.md).

## Configuration contract

Maintained experiments use a top-level `pipeline` and five logical blocks:

```yaml
pipeline: "Pipeline_01_Fault_Diagnosis"

environment:  # output path, seed, repeat count, process-level settings
  ...
data:         # metadata, raw data root, windows, workers, sampling policy
  ...
model:        # model family and model-specific parameters
  ...
task:         # diagnosis, generalization, few-shot, or pretraining objective
  ...
trainer:      # device, epochs, precision, logging, checkpoint behavior
  ...
```

Start from the nearest maintained file under `configs/demo/` or `configs/baselines/`.
Put research variants under `configs/experiments/`, and pass machine-specific paths
explicitly:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

After preflight succeeds, execute the same configuration:

```bash
phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/phm-data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.num_epochs=1
```

Composition and precedence rules are defined in [configs/README.md](configs/README.md).

<details>
<summary><strong>Public entrypoints</strong></summary>

The following process entrypoints share the same configuration and exit-status semantics:

```bash
phmfactory --config <yaml> [--override key=value ...]
python -m phmfactory --config <yaml> [--override key=value ...]
python main.py --config <yaml> [--override key=value ...]
```

Use the installed `phmfactory` command for normal work. `python main.py` remains a
repository compatibility launcher.

Useful bounded commands:

```bash
phmfactory doctor
phmfactory preflight --config <preset-or-yaml>
phmfactory demo
phmfactory data --help
```

</details>

## Support boundary

PHMFactory distinguishes four terms:

```text
discoverable       = an implementation or registry entry exists
runnable           = a reviewed execution path exists
execution-verified = the exact maintained command has current execution evidence
baseline-valid     = the exact complete experiment passed its declared scientific protocol
```

The software relationship is:

```text
execution-verified ⊆ runnable ⊆ discoverable
```

`baseline-valid` is a separate property of one complete configuration. It is not inferred
from component importability, Pipeline maturity, or another configuration's successful
run.

Current authority is generated from repository configuration and runtime descriptors:

- [Supported components](SUPPORTED_COMPONENTS.md)
- [Supported combinations](SUPPORTED_COMBINATIONS.md)
- [Configuration registry](configs/config_registry.csv)
- [Configuration Atlas](docs/CONFIG_ATLAS.md)

`execution_status=sanity_ok` means that the exact registered command has current evidence.
`protocol_status=smoke_only` forbids benchmark or algorithm-validity claims.
`protocol_status=baseline_valid` applies only to the exact data population, split, model,
task, checkpoint-selection, seed, and estimator combination that passed review.

The current `baseline_valid` reference is
[MFPT + GlobalAverageLinear](configs/baselines/01_mfpt/README.md). Its deliberately weak
model produced low accuracy, which is retained as an honest result. The promotion proves
protocol closure, not strong performance, state-of-the-art results, unrestricted dataset
redistribution, or arbitrary Cartesian-product compatibility.

## Optional Streamlit workspace

The browser workspace adapts the same public CLI:

```bash
python -m pip install -r apps/streamlit/requirements.txt
streamlit run apps/streamlit/app.py
```

Start with **Use safe CPU smoke defaults**. The interface can prepare a configuration,
validate it, launch the public command, and display logs and outputs. Its single-worker
scope and troubleshooting are documented in
[apps/streamlit/README.md](apps/streamlit/README.md).

<details>
<summary><strong>Developer architecture and repository map</strong></summary>

```text
phmfactory command / python -m phmfactory / main.py
  └── public command router
      └── resolved configuration + canonical Pipeline
          └── protected src runtime
              ├── data factory
              ├── model factory
              ├── task factory
              └── trainer factory
```

Primary paths:

- `phmfactory/` — public package, commands, configuration resolver, Pipeline descriptors, and run control;
- `configs/` — reusable blocks, maintained demos, baselines, research experiments, and registry;
- `src/data_factory/` — metadata, readers, datasets, samplers, and data assembly;
- `src/model_factory/` — model families and model construction;
- `src/task_factory/` — tasks, objectives, metrics, and task construction;
- `src/trainer_factory/` — trainer construction and extensions;
- `apps/streamlit/` — optional browser workspace;
- `test/` — maintained pytest suite;
- `docs/` — user, extension, development, release, and historical documentation.

Run the maintained checks before requesting review:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.gen_support_matrix
git diff --exit-code SUPPORTED_COMPONENTS.md SUPPORTED_COMBINATIONS.md
python -m pytest test/ -q
```

See [docs/testing.md](docs/testing.md) for focused gates and test terminology.

</details>

## Current pre-release limits

PHMFactory remains an alpha `0.3.0.dev0` source release:

- the Dummy demo is the only fully offline, repository-shipped first-run path;
- one exact real-data reference is now `baseline_valid`: MFPT + `GlobalAverageLinear`;
- that reference establishes protocol and estimator closure, not strong diagnostic accuracy;
- most other real-data configurations still require local metadata and raw signals and remain `smoke_only` unless explicitly listed otherwise;
- broader SEU, PU, and final CWRU acceptance are not yet complete;
- the GitHub repository has not been renamed;
- no final `v0.3.0` tag or package publication is claimed;
- experimental Pipelines and unlisted model/task combinations are not release-supported.

Read [Known limitations](KNOWN_LIMITATIONS.md) and the
[v0.3 release-readiness page](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) before making a
release or benchmark claim.

## Branch policy

`main` is the user-facing stable default branch. `dev` is the integration branch. Routine
feature, fix, documentation, test, CI, cleanup, and migration pull requests start from
the latest `dev` and target `dev`.

Only an explicitly authorized release-promotion pull request or emergency hotfix may
target `main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Contributing and citation

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request. A useful
problem report includes the exact commit, configuration, overrides, environment, data
source, and complete terminal output.

- Bugs and feature requests: [GitHub Issues](https://github.com/PHMbench/PHM-Vibench/issues)
- Development workflow: [Developer guide](docs/developer_guide.md)
- Release readiness: [v0.3 release status](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- Software citation metadata: [CITATION.cff](CITATION.cff)
- License: [Apache License 2.0](LICENSE)

Dataset and model artifacts may have separate source licenses. Cite the exact commit or
release used for each experiment.

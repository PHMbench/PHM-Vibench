# PHMFactory

<div align="center">
  <img src="pic/PHM-Vibench.png" alt="PHMFactory logo" width="260"/>

  <p>
    <a href="README.md"><strong>English</strong></a> |
    <a href="README_CN.md">中文</a>
  </p>

  <p><strong>Configuration-first PHM experiments for industrial signals.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/status-release%20blocked-critical" alt="Release blocked"/>
    <img src="https://img.shields.io/badge/version-0.3.0rc1-blue" alt="Version 0.3.0rc1"/>
    <img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB" alt="Python 3.10 or newer"/>
    <a href="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml"><img src="https://github.com/PHMbench/PHM-Vibench/actions/workflows/core-quality-gates.yml/badge.svg" alt="Core quality gates"/></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license"/>
  </p>
</div>

PHMFactory runs fault-diagnosis and related PHM experiments from one visible
configuration. The framework must execute the data, model, task, training, checkpoint,
and evaluation choices that the user requested.

The repository is still named
[`PHMbench/PHM-Vibench`](https://github.com/PHMbench/PHM-Vibench). The project, Python
package, and command are named **PHMFactory**, `phmfactory`, and `phmfactory`.

> **Current status:** the offline Dummy path is maintained. The MFPT transparent
> experiment remains `smoke_only` pending current-source requalification. There is no
> package-index release or current `baseline_valid` reference, so release readiness is
> blocked.

## Quick start

The first run is offline and uses repository-shipped Dummy data.

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

A successful run prints the paths you need:

```text
result_dir=...
best_checkpoint=...
test_metrics=...
run_summary=...
primary_metrics={...}
```

Use those paths to inspect the run. The Dummy demo verifies the installed software path;
it is not a real-data benchmark or a performance claim.

See [Quickstart](docs/quickstart.md) for the complete walkthrough and
[Installation](docs/installation.md) for platform notes.

## Run an experiment

Experiments require an explicit configuration:

```bash
phmfactory preflight \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1

phmfactory \
  --config configs/demo/01_cross_domain/cwru_dg.yaml \
  --override data.data_dir=/absolute/path/to/data \
  --override data.metadata_file=metadata.xlsx \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1
```

Machine-specific values are read only from an explicitly supplied `--local-config` file.
Configuration composition and precedence are documented in
[`configs/README.md`](configs/README.md).

## Project structure

```text
PHM-Vibench/
├── phmfactory/           # Public commands and configuration entrypoint
├── configs/              # Demo and research experiment configurations
├── src/
│   ├── data_factory/     # Readers, datasets, sampling and loaders
│   ├── model_factory/    # Models and representation modules
│   ├── task_factory/     # Objectives, metrics and optimization
│   ├── trainer_factory/  # Devices, callbacks and model selection
│   └── runtime/          # Experiment execution
├── data/                 # Bundled Dummy data and data-layout guide
├── test/                 # Runtime and component tests
├── apps/streamlit/       # Optional browser workspace
├── docs/                 # User and developer guides
├── doc/changelog/        # Upgrade notes
└── paper/project/        # Research source and migration notes
```

Start with `configs/` to run an experiment and the relevant Factory to add a
component. Run outputs are written to the paths printed by the command, not to a
fixed directory implied by this tree.

## Runtime structure

```text
resolved configuration
    ↓
canonical Pipeline
    ↓
Data Factory → Model Factory → Task Factory → Trainer Factory
    ↓
fit → selected checkpoint → test → finite metrics
    ↓
direct result paths
```

| Boundary | Responsibility |
| --- | --- |
| Data Factory | metadata, readers, selected samples, datasets, samplers, loaders |
| Model Factory | model identity, construction, explicitly requested weights |
| Task Factory | objective, metrics, optimizer and scheduler |
| Trainer Factory | device, callbacks, checkpoint selection, fit/test lifecycle |
| Pipeline | orchestration and success gating |

A compatible component should be replaceable by changing that component and its
configuration, not the other factories or the command router.

## Failure behavior

PHMFactory fails at the boundary that owns the problem. It does not switch to an easier
experiment after a requested data source, device, task, checkpoint, or metric fails.
Useful errors should state the requested value, the observed value, the expected
contract, and the smallest repair.

## Support terms

| Term | Meaning |
| --- | --- |
| `discoverable` | source or registry entry exists |
| `runnable` | a reviewed execution path exists |
| `execution-verified` | the exact command has current bounded execution evidence |
| `baseline-valid` | the exact full experiment passed its current scientific protocol |

Support belongs to an exact configuration. Importable code alone is not support evidence.
See [Supported combinations](SUPPORTED_COMBINATIONS.md),
[Known limitations](KNOWN_LIMITATIONS.md), and
[Release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md).

## Documentation

| Task | Start here |
| --- | --- |
| Install and complete the first run | [Quickstart](docs/quickstart.md) |
| Configure an experiment | [Configuration guide](configs/README.md) |
| Connect local data | [Data layout](data/README.md) |
| Add or select a model | [Model Factory](src/model_factory/README.md) |
| Add or select a task | [Task Factory](src/task_factory/README.md) |
| Configure training | [Trainer Factory](src/trainer_factory/README.md) |
| Use the optional browser workspace | [Streamlit](apps/streamlit/README.md) |
| Contribute code | [Contributing](CONTRIBUTING.md) |
| Understand project invariants | [Core contract](CORE.md) |

The full navigation is in [`docs/index.md`](docs/index.md).

## Development rules

Follow Occam's razor:

```text
DELETE → INLINE → MERGE → SIMPLIFY → DOCUMENT → ADD
```

One PR should protect one primary invariant and produce one user-visible result. Prefer
clear code and direct errors over fallback, wrapper layers, duplicate registries, or
future-oriented abstractions. Comments should explain a scientific or compatibility
reason; they should not restate the code.

Routine work starts from current `dev` and targets `dev`. Read [`CORE.md`](CORE.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) before broad changes.

## Publications and research

### Project paper

Qi Li, Bojian Chen, Xuan Li, Qitong Chen, Liang Chen, Changqing Shen, Lu Lu,
Zhaoye Qin, and Fulei Chu.
**[PHM-Vibench: A Unified and Factory-Style Vibration Benchmarking Framework for the Foundation Model Era](https://papers.phmsociety.org/index.php/phmap/article/view/4303)**.
*PHM Society Asia-Pacific Conference*, 5(1), 2025 proceedings;
published online January 13, 2026. DOI: [10.36001/phmap.2025.v5i1.4303](https://doi.org/10.36001/phmap.2025.v5i1.4303).

The paper describes PHM-Vibench. For capabilities of the current PHMFactory source,
use the [supported combinations](SUPPORTED_COMBINATIONS.md) and [known limitations](KNOWN_LIMITATIONS.md).

### Related method

Qi Li, Bojian Chen, Qitong Chen, Xuan Li, Zhaoye Qin, and Fulei Chu.
**[HSE: A plug-and-play module for unified fault diagnosis foundation models](https://doi.org/10.1016/j.inffus.2025.103277)**.
*Information Fusion*, 123, 103277, 2025.

HSE is listed as a related representation method, not as a claim that all published
experiments used the current software. In-progress work and historical paper sources
are separate from published results; see [research source notes](paper/project/README.md).

### Research using PHMFactory

To add a study, [open an issue](https://github.com/PHMbench/PHM-Vibench/issues) with its
publication link, code or experiment configuration, and the software version used.
Only studies with a documented use of this project belong in this category.

## Roadmap

| Stage | Focus |
| --- | --- |
| Available | Configuration-first CLI, offline Dummy first run, and direct result paths |
| Next | Complete declared metrics and result semantics; requalify the real-data reference experiment |
| Research | Evaluate interpretable-model explanation and heterogeneous-signal extensions before promoting them to maintained examples |

[Upgrade notes](doc/changelog/) record completed changes.
[Release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) records current blockers.
Research candidates are not release promises.

## Contributors and community

Core project contributors include [Qi Li](https://github.com/liq22) and
[Xuan Li](https://github.com/Xuan423). See [all contributors](https://github.com/PHMbench/PHM-Vibench/graphs/contributors)
for the full contribution history.

Use [Issues](https://github.com/PHMbench/PHM-Vibench/issues) for reproducible bugs and
bounded feature proposals, and [Discussions](https://github.com/PHMbench/PHM-Vibench/discussions)
for usage questions and research ideas. Contributions follow the
[contribution guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

[Star history](https://www.star-history.com/#PHMbench/PHM-Vibench&Date)

## Citation and license

PHMFactory is distributed under the [Apache License 2.0](LICENSE). Citation metadata are
in [`CITATION.cff`](CITATION.cff). Dataset and third-party licenses remain separate.

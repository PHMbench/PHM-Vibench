# PHM-Vibench v0.2.0 Baseline Report

Generated: 2026-07-10

## Scope and evidence class

This is the first **static cloud baseline** for release work. It records facts observable from the GitHub repository and connected GitHub metadata. It does **not** claim that tests or pipelines passed.

Runtime commands must be executed by the independent local route and attached as logs before any component is promoted to release-supported status.

## Repository identity

| Field | Observed value | Evidence status |
|---|---|---|
| Repository | `PHMbench/PHM-Vibench` | confirmed |
| Repository visibility | public | confirmed |
| Default branch | `main` | confirmed |
| Baseline commit | `ab064118b735eac641cdab41665591fe5c97e3fe` | confirmed |
| Active release branch | not established | no matching release branch found through available branch search |
| Latest tag | unknown | tag enumeration requires the local goal pack or GitHub tag API evidence |
| Protected branches | unknown | branch-protection metadata is not exposed by the available connector |
| Open pull requests | `#39`, `#40`, `#41`, `#42`, `#43` | confirmed at baseline time |

## Protected public contracts

The following contracts are release invariants:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Configuration sections:

```text
environment
data
model
task
trainer
```

Primary extension boundaries:

```text
src/data_factory/
src/model_factory/
src/task_factory/
src/trainer_factory/
```

## Environment and dependency declaration

| Field | Repository declaration | Baseline assessment |
|---|---|---|
| Python | README says `3.8+`; installation example uses `3.10`; dormant workflow uses `3.10` | compatibility range not release-verified |
| PyTorch | README says `2.0+`; requirements pins `2.6.0` | declaration mismatch requires resolution |
| CUDA | README says `11.1+`; requirements comment references CUDA 12.6 wheels | compatibility range not release-verified |
| Operating systems | no supported-OS matrix found | unknown |
| Hardware actually available | not observable from repository metadata | local inspection required |
| Dependency lock | no `pyproject.toml`, `setup.py`, or lock file found in initial inspection; `requirements.txt` mixes exact and broad constraints | not release-grade |
| CI | workflow exists under `.github/workflows_TODO/`, not active `.github/workflows/` | no active release gate confirmed |

## Static inventory baseline

| Inventory | Discovered count | Evidence class | Release support conclusion |
|---|---:|---|---|
| Pipeline source entrypoints | 6 | code search | all `S0_DISCOVERED`; runtime status unknown |
| Task registry entries | 15 | `src/task_factory/task_registry.csv` | discovered, not release-verified |
| Model registry entries | 35 | `src/model_factory/model_registry.csv` | discovered, registry test status not populated |
| Concrete sampler classes | 4 | sampler documentation and factory description | discovered, not contract-tested |
| Trainer implementations documented | 1 | trainer factory documentation | discovered, not platform-tested |
| Maintained demo configs | 7 | config registry / demo layout | schema and runtime baseline must be rerun |
| Config registry rows | 21 | `configs/config_registry.csv` | static SSOT count |
| Dataset readers | exact count pending local discovery | code search found multiple `RM_*` readers | not release-supported by count alone |

## Discovered pipelines

```text
src/Pipeline_01_default.py
src/Pipeline_02_pretrain_fewshot.py
src/Pipeline_03_multitask_pretrain_finetune.py
src/Pipeline_04_unified_metric.py
src/Pipeline_05_default_w_explain.py
src/Pipeline_ID.py
```

The repository also contains GUI, examples, plotting scripts, historical docs, and branch-only generative work. These are not alternative release entrypoints unless explicitly promoted through the config-first contract and release gates.

## Commands requested by the release task

The following commands have **not been executed in this cloud baseline** because the GitHub connector provides repository metadata/file access but not a checked-out runtime environment:

```bash
git status
git branch -a
git remote -v
git log --oneline --decorate -n 30
git tag --sort=-creatordate | head
find . -maxdepth 4 -type f | sort
python --version
python main.py --help
pytest -q
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

They are mandatory first-cycle commands in the external local goal pack. Their outputs must update `baseline_machine.json`; absence of a result must remain `NOT_EXECUTED`, never `PASS`.

## Initial evidence-backed findings

### P1: No active CI gate is confirmed

A config validation workflow is stored under `.github/workflows_TODO/config_tools_ci.yml`; therefore it does not function as a normal GitHub Actions workflow path.

### P1: Public support claims exceed current evidence

README currently presents `20+` datasets, `30+` algorithms, strong reproducibility language, and an `82% computational efficiency improvement` claim. The current baseline has no release-grade claim-evidence matrix proving these statements.

### P1: Dependency and platform support are not normalized

Python, PyTorch, and CUDA declarations are inconsistent across README, requirements, and the dormant workflow. Supported OS and tested hardware are not declared.

### P1: Registry discovery is not support evidence

Model registry `test_status` values are `/`; task registry test-status cells are empty. Registry presence must not be translated into `SUPPORTED_COMPONENTS.md` without construction, contract, and pipeline evidence.

### P1: Pipeline selection can silently fall back

`main.py` catches YAML parsing errors while selecting the pipeline and silently falls back to `Pipeline_01_default`. Invalid pipeline-selection input should eventually fail with an actionable error or produce explicit compatibility behavior with a regression test.

### P2: Default no-argument execution is not the offline smoke path

`main.py` defaults to `configs/demo/01_cross_domain/cwru_dg.yaml`, while repository onboarding identifies `configs/demo/00_smoke/dummy_dg.yaml` as the no-download smoke command. This is a compatibility/product decision requiring explicit resolution, not an unreviewed edit.

### P2: README inventory language is internally inconsistent

The badge says `20+` datasets, while the project-highlight text says `15+`. Neither number is yet tied to a verified reader/data compatibility ledger.

## Baseline release verdict

```text
NOT_READY
```

Reason: pipeline execution, component contracts, parameter tracing, legal combination coverage, active CI, supported-platform evidence, and independent review are incomplete.

## Next gates

1. Execute the local baseline commands and attach raw logs.
2. Populate exact repository, branch, pipeline, component, config, and test counts.
3. Promote no pipeline above `S0_DISCOVERED` without runtime evidence.
4. Merge no broad feature branch into `main`; continue small evidence-scoped PRs.
5. Require independent R1 and Claude counterexample review before release approval.

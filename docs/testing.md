# Testing and Validation

Run the narrowest test that exercises a change, then apply the relevant integration gate.
A passing import is not an end-to-end run, and a synthetic smoke is not scientific
performance evidence.

## Evidence terms

Use precise language in issues and pull requests:

- **Local pass** — the exact command succeeded in a named local environment.
- **CI pass** — a GitHub Actions job attached to the commit succeeded.
- **Not executed** — the command was not run; no result may be inferred.
- **Expected failure** — an intentionally invalid input failed at the documented boundary.
- **Smoke evidence** — a bounded execution path completed.
- **Contract evidence** — explicit input, output, shape, error, state, or artifact
  assertions passed.
- **Benchmark-valid evidence** — the exact data/protocol/metric/seed policy required for
  comparison was satisfied; ordinary smoke evidence is not enough.

## Maintained pytest suite

```bash
python -m pytest test/ -q
```

Historical release folders, `dev/test_history/`, `test/todo_test/`, and isolated research
workspaces are diagnostic unless a maintained document explicitly promotes a command.

## Documentation and configuration gate

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m pytest test/test_config_analysis_parity.py -q
python -m scripts.gen_config_atlas
python -m scripts.gen_support_matrix
git diff --exit-code \
  docs/CONFIG_ATLAS.md \
  SUPPORTED_COMPONENTS.md \
  SUPPORTED_COMBINATIONS.md
git diff --check
```

Generated files must be changed through their source registry or descriptor. Do not edit
generated output to hide drift.

## One configuration truth gate

The core invariant is:

```text
run effective config
= preflight effective config
= inspector effective config
= validator effective config
= Streamlit effective config
= Pipeline 06 effective config
```

For identical visible inputs, all paths must report the same
`effective_config_sha256`. The focused gate is:

```bash
python -m pytest test/test_config_analysis_parity.py -q
```

It covers:

- maintained preset versus explicit YAML path;
- equivalent YAML value versus CLI override;
- base/config/explicit-local/CLI precedence;
- absence of implicit `configs/local/local.yaml` discovery;
- compatibility `resolve_config` view;
- inspector, validator, and preflight parity;
- mapping-order-independent semantic hashing.

When changing config behavior, also run:

```bash
python -m pytest \
  test/test_phmfactory_config_resolver.py \
  test/test_phmfactory_commands.py \
  test/test_phmfactory_entrypoints.py \
  test/test_pipeline_name_migration.py \
  test/test_pipeline_maturity.py \
  test/test_runtime_control_plane.py \
  test/test_runtime_attestation.py -q
```

## Inspect one configuration

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The output includes the effective config hash, resolved values, leaf-field sources,
factory targets, and sanity checks. A failed sanity check returns non-zero.

To test an explicit machine-local layer:

```bash
python -m scripts.config_inspect \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

Omitting `--local-config` means no local YAML participates.

## User-first offline gate

The maintained first-run path is:

```bash
python -m pip install --no-deps -e .
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

A pass requires:

- all required doctor checks pass;
- preflight reports `effective_config_sha256` and does not create the configured output;
- the Dummy data/model/task/trainer path completes;
- the command exits zero;
- the terminal run manifest records `succeeded` and the effective config hash;
- per-iteration and aggregate metrics are indexed.

This path uses synthetic repository data and does not validate industrial accuracy.

## Pipeline 06 config contract

Pipeline 06 keeps its scientific implementation in `src`, but the public CLI imports a
narrow compiled-config adapter. Run:

```bash
python -m pytest \
  test/generative/test_pipeline06_import.py \
  test/generative/test_pipeline06_preflight.py \
  test/generative/test_pipeline06_dispatch.py \
  test/generative/test_pipeline06_compiled_config.py -q
```

The adapter test proves that the public path does not re-read YAML, reapply overrides, or
accept a mismatched compiled Pipeline.

## Streamlit parity gate

```bash
python -m pytest \
  test/test_streamlit_config_service.py \
  test/test_streamlit_runtime_policy.py \
  test/test_streamlit_onboarding.py \
  test/test_streamlit_run_service.py \
  test/test_streamlit_result_service.py \
  test/test_streamlit_ui_imports.py -q
```

These tests run on Linux and Windows. The UI remains an adapter around the public
inspector and CLI; it does not define hidden local config or a second training runtime.

## Other focused gates

```bash
# UXFD assembly contract
python -m pytest test/test_tspn_uxfd_assembly.py -q

# Shared classification lifecycle and evidence
python -m pytest \
  test/test_classification_runtime.py \
  test/test_pipeline02_runtime.py \
  test/test_runtime_evidence.py -q

# Complete generative research tests when applicable
python -m pytest test/generative/ -q
```

Verify every referenced file exists on the branch before copying a command into a PR.

## Active GitHub Actions

Current pull requests can trigger:

- **Core quality gates** — docs/config parity, Pipeline 06 adapter contract, UXFD focused
  contract, and actual offline user-first smoke;
- **PHMFactory public package** — source tests, wheel contents, clean-wheel preflight,
  process exit semantics, and compiled-config dispatch;
- **Streamlit quality gates** — focused UI services on Linux and Windows;
- repository layout, dependency ownership, data-bundle, submodule, and release-readiness
  gates as their paths require.

A green focused workflow proves only its declared scope.

## Test selection by change type

| Change | Minimum focused gate | Additional integration gate |
| --- | --- | --- |
| Documentation wording or links | `validate_docs` | config validation and clean generated diff when commands/claims change |
| Config resolver, CLI, inspector, validator | config parity tests | public-package wheel and offline smoke |
| Maintained YAML or registry | `validate_configs`, `config_inspect` | generated docs and applicable smoke |
| Data reader or sampler | focused reader/adapter test | Dummy smoke plus affected real-data test when legal assets exist |
| Model implementation | import/shape/assembly test | affected task integration and Dummy smoke |
| Task, loss, or metric | focused behavior and negative tests | affected maintained demo smoke |
| Trainer or checkpoint | lifecycle and failure tests | Dummy train/test and checkpoint round trip |
| Pipeline 06 shell/config | compiled-config adapter tests | affected train/sample/eval contract |
| Streamlit | focused Linux/Windows service tests | public config parity and CLI smoke |
| Release support claim | exact linked runtime evidence | independent review of support matrices and limitations |

## Environment reporting

Include:

```text
operating system
Python version
PyTorch and Lightning versions
CPU/GPU and CUDA when relevant
repository commit
config or preset
effective_config_sha256
explicit local config, if any
CLI overrides
data source
full command and exit code
```

For dependency diagnostics:

```bash
python --version
python -m pip freeze
```

Attach full environment dumps to an issue or CI artifact rather than committing them by
default.

## Reproducibility

A reproducibility claim requires the same effective configuration, code revision, data
version, split, seed, protocol, environment, and relevant artifacts. Hardware-dependent
numerical tolerance must be declared; GPU bitwise identity is not assumed.

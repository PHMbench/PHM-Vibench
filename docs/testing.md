# Testing and Validation

This page is the maintained source for PHM-Vibench test commands and evidence
terminology. Run the narrowest test that exercises a change, then apply the
relevant merge gate.

## Evidence terms

Use precise language in issues and pull requests:

- **Local pass** — the command ran successfully in a named local environment.
- **CI pass** — a GitHub Actions job attached to the commit completed successfully.
- **Not executed** — the command was not run; do not infer a result.
- **Expected failure** — an intentionally invalid case failed with the documented
  error boundary.
- **Smoke evidence** — a bounded path executed; this is not benchmark-performance
  evidence.
- **Contract evidence** — inputs, outputs, shapes, devices, errors, or artifacts
  satisfied explicit assertions.

Never describe local output as CI output, an import as an end-to-end run, or
synthetic data as evidence of scientific performance.

## Maintained test directory

The maintained pytest gate is:

```bash
python -m pytest test/ -q
```

Files under `dev/test_history/`, `test/todo_test/`, historical release folders, or
research workspaces are diagnostic unless a maintained page explicitly promotes a
specific command.

## Lightweight documentation and configuration gate

Use this gate for documentation, registry, and YAML changes:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
```

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`. Do not
hand-edit the atlas to make the diff disappear.

A documentation-only pull request may stop at this gate when it does not change a
command, configuration, runtime claim, or executable example. The pull-request
description must explain why runtime tests are not applicable.

## Configuration inspection

Inspect the resolved values, sources, runtime targets, and sanity checks:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The command must return non-zero when a sanity check fails. Missing dependencies,
unknown imports, or invalid targets are failures, not warnings to ignore.

## Offline end-to-end smoke

The repository-shipped dummy configuration is the default runtime gate:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

A pass requires a clean exit through config loading, data construction, model and
task assembly, trainer execution, and output creation. See the
[quickstart](quickstart.md) for expected behavior.

This smoke uses synthetic repository data and does not validate accuracy on an
industrial dataset.

## Focused tests

Examples of focused gates already present in the repository:

```bash
# Configuration tooling
python -m pytest test/test_config_tools.py -q

# Streamlit configuration and process boundaries
python -m pytest \
  test/test_streamlit_config_service.py \
  test/test_streamlit_runtime_policy.py \
  test/test_streamlit_run_service.py \
  test/test_streamlit_result_service.py -q

# TSPN_UXFD assembly contract
python -m pytest test/test_tspn_uxfd_assembly.py -q

# Generative/pretraining contract tests
python -m pytest test/generative/ -q
```

Verify that a referenced test file exists on the branch before copying a command
into documentation or a pull request.

## Active GitHub Actions gate

`.github/workflows/core-quality-gates.yml` currently runs on pull requests and
pushes to `main`.

The active workflow checks:

1. documentation and maintained configuration consistency;
2. generated atlas cleanliness and whitespace;
3. focused CPU compilation and tests for the UXFD U1 assembly contract.

The workflow is intentionally narrower than the entire optional research stack.
A green focused workflow does not mean every model family, dataset, or external-
data demo has been executed.

## Test selection by change type

| Change | Minimum focused gate | Additional merge gate |
| --- | --- | --- |
| Documentation wording or links | `validate_docs` | `validate_configs`, atlas clean diff, `git diff --check` |
| Maintained YAML or registry | `validate_configs`, `config_inspect` | atlas clean diff and applicable smoke |
| Data reader or sampler | focused contract test | dummy smoke plus affected real-data test when assets exist |
| Model implementation | import/assembly/shape test | affected task integration and dummy smoke |
| Task, loss, or metric | focused behavior and negative tests | affected maintained demo smoke |
| Trainer or checkpoint behavior | focused lifecycle test | dummy train/test loop and checkpoint round trip |
| Streamlit | focused service tests | core config checks; CLI smoke remains authoritative |
| Release support claim | linked runtime evidence | independent review of supported components/combinations |

## Environment reporting

Include at least:

```text
operating system
Python version
PyTorch and PyTorch Lightning versions
CPU/GPU and CUDA version when relevant
repository commit
configuration path and overrides
data source
full command
exit code
```

For dependency diagnostics:

```bash
python --version
python -m pip freeze
```

Do not commit a complete environment dump unless it is a deliberate, reviewed
release artifact; attach it to an issue or CI artifact instead.

## Reproducibility

For a reproducibility claim, record the commit, configuration, overrides, data
version, split, seed, environment, and output artifact. A repeated run can be
considered reproducible only against an explicit tolerance appropriate to the
metric and hardware path; GPU bitwise identity is not assumed.

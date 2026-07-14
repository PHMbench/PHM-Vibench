# Test and Validate PHM-Vibench

This page is the canonical testing and validation guide. Use the narrowest test
that proves the changed contract, then run the broader maintained gate before
merging.

## Evidence levels

Do not describe every successful command as the same kind of evidence.

| Level | What it proves | What it does not prove |
|---|---|---|
| Import | A module can be imported in one environment | Factory construction or runtime behavior |
| Unit | A focused function/class contract holds | Pipeline integration |
| Component contract | Shape, dtype, device, error, or registry contract holds | Full training/evaluation path |
| Config inspection | YAML composition, sources, targets, and sanity checks resolve | A batch can run |
| Smoke | The selected path runs a minimal batch/epoch/sample and exits cleanly | Scientific performance or broad compatibility |
| Mini end-to-end | Load, build, train/evaluate, and artifact path complete | Full-scale convergence or benchmark validity |
| Reproducibility | Repeated documented runs satisfy a stated tolerance | Universal determinism across hardware |

A skipped, not-executed, missing-data, or missing-dependency case is not a pass.

## Fast documentation and configuration checks

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
```

`docs/CONFIG_ATLAS.md` is generated from `configs/config_registry.csv`. An
intentional registry change should update both files in the same pull request.

## Inspect one configuration

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
```

The inspector returns non-zero when a sanity check fails. A zero exit code is a
configuration/import contract, not an end-to-end result.

## Run focused tests

Examples:

```bash
python -m pytest test/test_config_tools.py -q
python -m pytest test/test_tspn_uxfd_assembly.py -q
python -m pytest test/test_streamlit_config_service.py -q
```

For a failure, preserve full context:

```bash
python -m pytest path/to/test_file.py::test_name -vv --tb=long
```

A useful regression test should assert behavior, output, error semantics, or
observable state. Merely asserting that code does not raise is usually
insufficient.

## Run the maintained pytest suite

```bash
python -m pytest test/ -q
```

The `test/` directory is the maintained gate. Historical or experimental tests
outside that directory are diagnostic unless explicitly promoted.

## Run the offline config-first smoke

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Expected behavior is documented in [Quickstart](quickstart.md). The run uses
repository-shipped dummy data and provides functional evidence only.

## Validate a changed maintained config

For each affected config:

```bash
python -m scripts.config_inspect \
  --config <yaml> \
  --override trainer.num_epochs=1

python main.py \
  --config <yaml> \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

External-data runs require the correct local data root and metadata. Record the
source, split, preprocessing, seed, environment, and overrides.

## Test a component contribution

A model, task, sampler, trainer, reader, or configuration contribution should
normally include:

- import or registry lookup coverage;
- constructor validation;
- input/output or batch contract assertions;
- invalid-input behavior;
- CPU coverage where feasible;
- GPU coverage only when the component requires it and suitable hardware exists;
- checkpoint/artifact coverage when the change affects persistence;
- the smallest applicable config-first smoke.

## Randomness and reproducibility

Record the configured seed and test at least the seeds needed for the claim being
made. Fixed-seed reproducibility should use explicit tolerances and account for
hardware/library differences; do not require cross-GPU bitwise identity unless
that is an explicit supported contract.

Always check for:

```python
assert torch.isfinite(loss)
assert torch.isfinite(output).all()
```

where numerical stability is part of the contract.

## GitHub Actions and local evidence

The active workflow is `.github/workflows/core-quality-gates.yml`. Read that file
for the exact jobs and dependency sets enforced on the current branch.

Use these labels accurately in pull requests:

- **GitHub Actions evidence**: a named workflow run attached to the PR head;
- **local evidence**: commands executed in a described local environment;
- **not executed**: a required gate that could not be run;
- **expected failure**: a negative test whose non-zero exit is the assertion.

Never describe local output as CI evidence. Never hide a failed check with
`continue-on-error`, a broad skip, or an unrelated dependency installation.

## Documentation-only changes

A documentation-only pull request may omit runtime tests when it explains why
they are not applicable. It should still run:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
```

## Legacy test material

`dev/test_history/` contains historical validation material from earlier
refactors. It may require obsolete dependencies and is not part of the maintained
gate. Preserve it as evidence, but do not cite it as current pass status without
re-running and reviewing it on the current commit.

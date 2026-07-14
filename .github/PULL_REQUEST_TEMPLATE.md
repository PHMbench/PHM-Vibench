## Problem and motivation

Describe the specific problem, user/research scenario, and why this repository
should address it. Link the related issue or design evidence when available.

## Scope

### Included

- 

### Non-goals

- 

## Public behavior and compatibility

Describe changes to the CLI, config keys, registry/factory routing, data/batch
contracts, tensor shapes, checkpoints, artifacts, or documentation.

- Backward compatibility:
- Migration or deprecation:
- Invalid combinations rejected:
- Supported/experimental boundary:

## Changes

List the important files or components and why each changed.

## Validation evidence

Record the exact environment and command outcomes. Use `NOT_EXECUTED` rather than
claiming a pass when a gate could not run.

```text
Repository commit:
Operating system:
Python:
PyTorch/CUDA:
Data or fixture:
```

```bash
# Commands run, one per block or line
```

```text
# Results, counts, exit codes, or evidence paths
```

Distinguish clearly between:

- GitHub Actions evidence attached to this PR head;
- local evidence;
- expected negative-test failures;
- skipped or not-executed gates.

## Tests

- [ ] Added or updated focused regression/contract tests.
- [ ] Tested invalid inputs or combinations where applicable.
- [ ] Did not weaken, broadly skip, or delete tests to hide a failure.
- [ ] Ran the narrowest affected tests.
- [ ] Ran `python -m pytest test/ -q`, or explained why it is not applicable.

## Config, registry, and documentation

- [ ] Preserved `python main.py --config <yaml> [--override key=value ...]`.
- [ ] Preserved the `environment/data/model/task/trainer` configuration model.
- [ ] Put unverified experiment configs under `configs/experiments/`.
- [ ] Updated the applicable registry and generated `docs/CONFIG_ATLAS.md` together.
- [ ] Updated the authoritative documentation rather than duplicating instructions.
- [ ] Added new maintained pages to the appropriate navigation.
- [ ] Did not add unsupported performance, scale, compatibility, or status claims.

## Data, model, or external-source provenance

When applicable, provide:

- source and stable identifier;
- license and redistribution constraints;
- preprocessing or conversion steps;
- model/paper attribution;
- expected output and known limitations.

## Risks and limitations

List remaining risks, optional dependencies, unsupported environments, missing
full-data evidence, or scientific claims that this PR does not establish.

## Rollback

Explain how to revert the change and whether configs, checkpoints, data, or
artifacts need special handling.

## Final review checklist

- [ ] Diff has one coherent primary goal.
- [ ] No credentials, personal absolute paths, caches, logs, or local goal packs.
- [ ] No unrelated formatting, mass move, or broad deletion.
- [ ] Failures are explicit rather than silent fallbacks.
- [ ] Documentation/config commands use paths and keys that exist on this branch.
- [ ] `python -m scripts.validate_docs` passed or is explicitly `NOT_EXECUTED`.
- [ ] `python -m scripts.validate_configs` passed or is not applicable.
- [ ] `python -m scripts.gen_config_atlas` leaves no unintended diff.
- [ ] `git diff --check` passed.

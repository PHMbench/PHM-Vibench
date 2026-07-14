# Contribute a Trainer or Trainer Extension

This page is the trainer-factory addendum to the repository-wide
[contributor guide](../../CONTRIBUTING.md). The maintained trainer interface is
described in [`README.md`](README.md).

## Factory contract

The trainer is selected primarily by:

```yaml
trainer:
  name: "Default_trainer"
```

`src/trainer_factory/trainer_factory.py` first checks the trainer registry and
then uses the documented legacy import fallback based on `trainer.trainer_name`.
New configs should use `trainer.name` unless a compatibility case requires the
fallback.

A trainer implementation is a callable that returns a configured
`pytorch_lightning.Trainer` using:

```python
def trainer(args_e, args_t, args_d, path):
    ...
```

Do not document or implement a new subclass interface unless the actual factory
is changed in the same reviewed PR.

## Decide whether a new trainer is needed

Prefer extending `Default_trainer.py` or an existing extension when the change is
only:

- callback configuration;
- logger configuration;
- accelerator/device/precision settings;
- checkpoint or early-stopping behavior;
- bounded artifact or manifest handling.

Create a separate trainer only when configuration cannot express a coherent
lifecycle difference. Avoid parallel training frameworks and pipeline-specific
trainer branches.

## Document the behavior

State:

- new or changed `trainer.*` fields, types, defaults, and allowed values;
- CPU/GPU/accelerator/precision behavior;
- distributed strategy and worker assumptions;
- callbacks, checkpoint names, monitor keys, and resume semantics;
- loggers and external services;
- output directories and side effects;
- failure behavior when hardware or dependencies are unavailable;
- task/data assumptions;
- supported and unsupported environments.

Do not silently convert an unavailable GPU request to a different device unless
that compatibility behavior is explicit and tested.

## Registration and configuration

Register the implementation with the existing trainer registry or document the
legacy import path. Add a base config only when the fields are reusable. Start
new experiment combinations under `configs/experiments/`, not directly under
`configs/demo/`.

Inspect the result before running:

```bash
python -m scripts.config_inspect \
  --config configs/experiments/<trainer>_smoke.yaml \
  --dump targets \
  --override trainer.num_epochs=1
```

## Add tests

Focused tests under `test/` should cover:

- registry/fallback resolution;
- callable invocation with minimal namespaces;
- CPU construction;
- requested accelerator/precision values;
- callback and logger presence;
- monitor/checkpoint configuration;
- resume or checkpoint metadata when changed;
- missing dependency/hardware errors;
- output path and artifact behavior;
- no unexpected network service requirement for the offline smoke path.

For process-management or artifact extensions, test cancellation, path safety,
atomic writes, malformed files, and bounded scanning where relevant.

## Run integration validation

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/experiments/<trainer>_smoke.yaml \
  --override trainer.num_epochs=1
python main.py \
  --config configs/experiments/<trainer>_smoke.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m pytest test/ -q
```

When a GPU or distributed test cannot run, report it as `NOT_EXECUTED` and keep
that mode outside the supported boundary until evidence exists.

## Promote to the maintained surface

Promotion requires a portable config, focused trainer tests, a passing stated
smoke command, explicit hardware/dependency limits, registry/config traceability,
and documentation updates. A trainer feature that only passed with external
services or a private cluster must not be implied to work in the offline path.

## Checklist

- [ ] Implementation matches the callable factory contract.
- [ ] New configs use `trainer.name` unless compatibility requires otherwise.
- [ ] Config fields, callbacks, loggers, hardware, and side effects documented.
- [ ] CPU/error behavior covered; GPU/distributed claims backed by evidence.
- [ ] Checkpoint/resume behavior tested when changed.
- [ ] No implicit scheduler or parallel framework introduced.
- [ ] Experimental config starts under `configs/experiments/`.
- [ ] Config inspection, focused tests, and smoke results recorded accurately.

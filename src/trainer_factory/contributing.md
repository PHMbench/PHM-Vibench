# Contributing Trainers and Trainer Extensions

Use this page for trainer-factory-specific work. General contribution, testing,
documentation, and pull-request rules are in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

The trainer factory constructs a `pytorch_lightning.Trainer`; it does not expect a
custom subclass with a new `fit` API.

## Runtime contract

The factory selects `trainer.name` from its runtime registry. Its compatibility
fallback imports:

```text
src.trainer_factory.<trainer.trainer_name>
```

and calls a module-level function:

```python
def trainer(args_e, args_t, args_d, path):
    """Return a configured pytorch_lightning.Trainer."""
```

The maintained implementation is registered as `Default_trainer` and configures
callbacks, loggers, accelerator, devices, epoch count, strategy, and output path.
A new trainer implementation should preserve the pipeline's expectation that the
factory returns a ready-to-use Lightning Trainer or fail explicitly.

## Choose the extension point

Before adding another trainer, determine whether the change is better expressed as:

- a new or updated callback under `src/trainer_factory/extensions/`;
- a logger option;
- a configuration field consumed by `Default_trainer`;
- a task-level optimizer or scheduler change;
- a genuinely different trainer-construction policy.

Prefer the smallest matching extension point. Do not create a second trainer
merely to change one callback or logger default.

## Implement a trainer or extension

For a new trainer policy:

1. Add `src/trainer_factory/<TrainerName>.py`.
2. Expose `trainer(args_e, args_t, args_d, path)`.
3. Register it with `@register_trainer("<TrainerName>")`.
4. Return a configured `pytorch_lightning.Trainer`.
5. Keep public values under `trainer.*` or `environment.*` and document defaults.
6. Preserve CPU operation when CPU support is claimed.
7. Make optional logging services opt-in and usable without network access when
   disabled.
8. Ensure multi-process callbacks write artifacts only from the appropriate rank.
9. Make checkpoint, resume, early-stopping, pruning, and manifest behavior
   explicit.

For a callback or manifest extension, keep it isolated under
`src/trainer_factory/extensions/` and document its lifecycle, state, output files,
and distributed behavior.

## Configuration

The maintained base trainer config is:

```text
configs/base/trainer/default_single_gpu.yaml
```

A new public trainer config belongs in `configs/experiments/` first. Promotion to
a maintained base or demo requires config inspection, focused tests, an applicable
smoke path, registry/atlas synchronization when relevant, and migration notes for
changed defaults.

Do not silently reinterpret legacy fields such as `gpus`, `devices`, `num_epochs`,
or `max_epochs`. Compatibility fallback must be documented and tested.

## Tests

Add pytest coverage under `test/` for:

- construction through the real trainer factory;
- returned `pytorch_lightning.Trainer` type and configured accelerator/devices;
- CPU behavior;
- callback and logger selection;
- checkpoint directory, monitor, filename, and `save_top_k` behavior;
- early-stopping and pruning boundaries;
- manifest/artifact creation and rank behavior;
- optional service disabled mode without credentials or network;
- invalid monitor, device, precision, or strategy values;
- resume/checkpoint behavior when supported;
- clean process exit and no unintended files outside the run directory.

Use temporary directories in tests. Do not write test checkpoints, logs, or
service credentials into the repository tree.

Suggested gates:

```bash
python -m pytest <focused-trainer-test> -q
python -m scripts.config_inspect --config <yaml> --dump all --format json
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.device=cpu \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
python -m pytest test/ -q
```

GPU and distributed claims require an environment that actually executes those
paths. A CPU test cannot be reported as CUDA or multi-GPU evidence.

## Pull-request evidence

Include the configuration fields consumed, returned Trainer settings, callback and
logger behavior, output/checkpoint paths, distributed assumptions, focused tests,
CPU/GPU evidence, compatibility/migration impact, and rollback method. If an
external tracking service is added, document its dependency, credentials,
offline behavior, privacy impact, and failure mode.

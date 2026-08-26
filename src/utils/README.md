# Internal utilities

`src/utils/` contains implementation helpers used by maintained and compatibility
Pipelines. It is not the public entrypoint for PHMFactory.

Public users should start with:

```bash
phmfactory doctor
phmfactory preflight --config smoke
phmfactory demo
```

Configuration composition and strict validation belong to
`phmfactory.config.analyze_config()`. Do not add another config loader under `src/utils/`.

## Current responsibilities

| Area | Current owner |
| --- | --- |
| Finite repeated-run summaries | `run_summary.py` |
| Explicit CLI override helpers used by compatibility code | `config_utils.py` |
| Logger lifecycle and selected-checkpoint loading | `utils.py` |
| Small internal registration helper | `registry.py` |
| Explicit multi-stage orchestration | `training/two_stage_orchestrator.py` |
| Named compatibility adapters | `config/pipeline_adapters.py` |

A file being present here does not make it a public API. The public contract is defined by
`README.md`, `CORE.md`, the current CLI, and maintained tests.

## Change rules

1. **Keep one authority.** Public configuration must enter through
   `phmfactory.config.analyze_config()` and reach the Runtime without a second YAML parse.
2. **Preserve original failures.** Do not turn reader, model, task, Trainer, checkpoint,
   or metric errors into warnings, `None`, or a different backend.
3. **Do not repair scientific input.** Do not drop samples, rewrite labels, copy channels,
   shorten requests, change devices, or select another checkpoint to keep a run alive.
4. **Avoid hidden defaults.** Values that alter data, optimization, selection, or
   evaluation must be visible in the maintained configuration.
5. **Delete unused identity machinery.** Do not add config hashes, receipts, ledgers,
   attestations, or manifests that no current decision consumes.
6. **Use the smallest abstraction.** Prefer a local function over a manager, adapter,
   wrapper, or registry when one function has one current caller.
7. **Keep cleanup catches narrow.** `finally` blocks may close files and loggers; broad
   exception handlers must not convert failure into success.

## Adding or changing a helper

Before adding code, identify:

```text
current caller
→ duplicated responsibility
→ smallest shared function
→ focused failure case
```

A new helper is justified only when it removes real duplication or gives two maintained
callers the same non-trivial behavior. Do not add utilities for hypothetical future
Pipelines.

Comments should explain a scientific invariant, ownership boundary, compatibility reason,
or non-obvious failure condition. Comments that restate the next line of code should be
removed.

## Validation

Run tests that cover the changed responsibility. Common examples are:

```bash
python -m pytest test/test_run_summary.py -q
python -m pytest test/test_runtime_execution.py -q
python -m pytest test/test_pipeline02_runtime.py -q
python -m scripts.validate_docs
```

Use the full offline Dummy lifecycle when a change can affect Data, Model, Task, Trainer,
checkpoint restoration, evaluation, or result paths. See [`docs/testing.md`](../../docs/testing.md)
for the repository validation matrix.

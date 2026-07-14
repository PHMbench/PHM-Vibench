# PHM-Vibench Runtime Source (`src/`)

The `src/` tree contains pipelines, factories, task/model/data implementations,
trainer construction, and shared runtime utilities.

Users should start from the root [README](../README.md) and run:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Do not invoke internal modules as a substitute for the public config-first path
unless debugging a specific implementation contract.

## Runtime flow

```text
main.py
  └── configured pipeline
      ├── data_factory
      ├── model_factory
      ├── task_factory
      ├── trainer_factory
      └── training / evaluation / artifacts
```

## Main directories

| Path | Responsibility |
|---|---|
| `data_factory/` | metadata, readers, dataset wrappers, samplers, data loaders |
| `model_factory/` | dynamic model import, model families, reusable model components |
| `task_factory/` | task modules, losses, metrics, optimizer/scheduler behavior |
| `trainer_factory/` | PyTorch Lightning trainer construction and extensions |
| `configs/` | internal Python config utilities and compatibility helpers |
| `utils/` | shared runtime/configuration/environment helpers |
| `explain_factory/` | explanation-related utilities used by selected paths |
| `plot_factory/` | plotting utilities; not a release-support claim by itself |

Each major factory has a maintained `README.md` and contribution addendum.
Architecture-wide development guidance is in
[`docs/developer_guide.md`](../docs/developer_guide.md).

## Pipelines

Pipeline files live directly under `src/`. Their presence does not mean every
pipeline is release-supported. The current public boundary is defined by:

- [`SUPPORTED_COMPONENTS.md`](../SUPPORTED_COMPONENTS.md)
- [`SUPPORTED_COMBINATIONS.md`](../SUPPORTED_COMBINATIONS.md)
- [`KNOWN_LIMITATIONS.md`](../KNOWN_LIMITATIONS.md)

Use a maintained config and `scripts.config_inspect` to identify the actual
pipeline and factory targets for a run.

## Extension rules

- Keep component-specific behavior inside the appropriate factory.
- Preserve the five config sections: `environment/data/model/task/trainer`.
- Do not create a second loader, registry, or training framework.
- Keep optional dependencies behind the selected component boundary.
- Reject invalid combinations with actionable errors before deep tensor failures.
- Add focused tests and a config-first smoke path.
- Put unverified configurations under `configs/experiments/`.
- Treat research, paper, old-plan, and agent workflow material as evidence, not
  current support documentation.

## Validation

See [`docs/testing.md`](../docs/testing.md) for evidence levels and maintained
commands. At minimum, changes to runtime source should run the narrow affected
tests plus the applicable config and smoke gates.

# CDDG Task

`CDDG` contains the current cross-system/domain classification wrapper used by
the maintained cross-system demo.

## Current Surface

| Task type | Task name | Module | Maintained config |
|---|---|---|---|
| `CDDG` | `classification` | `classification.py` | `configs/base/task/cddg.yaml` |

`classification.py` subclasses `Default_task`. Cross-system behavior is driven
by the CDDG dataset task, metadata fields, and task config values such as
`target_system_id` and `target_domain_num`.

Maintained demo path:

- `configs/demo/02_cross_system/multi_system_cddg.yaml`

## Configuration Notes

Use the inspect tool before changing CDDG fields:

```bash
python -m scripts.config_inspect \
  --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1
```

Typical task fields live in `configs/base/task/cddg.yaml`:

- `task.type: "CDDG"`
- `task.name: "classification"`
- `task.target_system_id`
- `task.target_domain_num`
- optimizer, loss, and metric fields inherited by `Default_task`

## Boundaries

- This README does not claim support for extra contrastive losses, domain
  adaptation losses, prompt-specific CDDG behavior, or system-aware samplers.
- Multi-stage HSE experiments should be documented through maintained configs
  and release evidence, not by speculative examples in this directory.
- Add new CDDG task variants only after updating `src/task_factory/task_registry.csv`,
  config validation, and a maintained smoke path.

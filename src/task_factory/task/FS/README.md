# FS Task

`FS` contains few-shot task implementations. The current maintained public demo
uses the config-first `FS.classification` wrapper.

## Current Surface

| Task type | Task name | Module | Status |
|---|---|---|---|
| `FS` | `classification` | `classification.py` | Maintained demo path |
| `FS` | `prototypical_network` | `prototypical_network.py` | Registered implementation |
| `FS` | `matching_network` | `matching_network.py` | Registered implementation |
| `FS` | `knn_feature` | `knn_feature.py` | Registered implementation |
| `FS` | `finetuning` | `finetuning.py` | Registered implementation |

Maintained demo path:

- `configs/demo/03_fewshot/cwru_protonet.yaml`

The registry source of truth is `src/task_factory/task_registry.csv`.

## Configuration Notes

Inspect the maintained demo before copying it:

```bash
python -m scripts.config_inspect \
  --config configs/demo/03_fewshot/cwru_protonet.yaml \
  --override trainer.num_epochs=1
```

Typical task fields live in `configs/base/task/fewshot.yaml`:

- `task.type: "FS"`
- `task.name`
- `task.n_way`
- `task.k_shot`
- `task.q_query`
- `task.episodes_per_epoch`

## Boundaries

- Do not treat every registered FS module as release-supported. Release support is
  established by registry status, maintained configs, and smoke evidence.
- This directory does not currently claim multi-scale episodes, cross-domain
  episodes, hierarchical few-shot learning, or learned distance metrics.
- Add new FS behavior by updating code, `src/task_factory/task_registry.csv`,
  config validation, docs, and a focused smoke path together.

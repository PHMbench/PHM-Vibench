# `configs/base/task/`

## 1) What This Block Controls

`task` controls the learning objective and data splitting logic:
- `task.type` + `task.name` selects a concrete task implementation in `task_factory`
- optimizer/loss/metrics live here (unless a task hardcodes behavior)

## 2) Minimal Example (YAML Snippet)

```yaml
task:
  type: "DG"
  name: "classification"
  lr: 0.001
  metrics: ["acc"]
```

## 3) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `task.type` | enum | One of `DG/CDDG/FS/GFS/pretrain/Default_task`. |
| `task.name` | str | Task name under the given type. |
| `task.target_system_id` | list[int] | System selection using metadata `Dataset_id`. |
| `task.source_domain_id` | list[int] | DG domain split ids (depends on metadata). |
| `task.target_domain_id` | list[int] | DG target domain split ids. |

## 4) Typical Overrides

```bash
python main.py --config <yaml> --override task.lr=0.0005
python main.py --config <yaml> --override task.target_system_id=[1]
```

## 5) Coupling Notes

- Task implementations are indexed by `src/task_factory/task_registry.csv`.
- `task.target_system_id` values must exist in your metadata `Dataset_id` column.

## 6) How to Extend

1) Add a new task module under `src/task_factory/task/<TYPE>/`.
2) Add an entry to `src/task_factory/task_registry.csv`.

## 7) Common Pitfalls

1) Confusing metadata `Dataset_id` with folder order (use inspect + metadata table).
2) Using domain ids that do not exist in metadata.
3) Setting `task.type` to a value without a matching registry entry.
4) Mixing FS/GFS semantics (ensure the right task.type is used).
5) Forgetting to update task_registry.csv when adding a new task.


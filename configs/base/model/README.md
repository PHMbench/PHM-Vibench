# `configs/base/model/`

## 1) What This Block Controls

`model` controls:
- model family (`model.type`) and implementation (`model.name`)
- (for ISFM) subcomponents: `embedding`, `backbone`, `task_head`

## 2) Minimal Example (YAML Snippet)

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
```

## 3) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `model.type` | str | Top-level family used by model_factory. |
| `model.name` | str | Concrete implementation under `model.type`. |
| `model.embedding` | str | ISFM component id (see `src/model_factory/ISFM/isfm_components.csv`). |
| `model.backbone` | str | ISFM backbone id. |
| `model.task_head` | str | ISFM head id. |

## 4) Typical Overrides

```bash
python main.py --config <yaml> --override model.backbone=B_08_PatchTST
python main.py --config <yaml> --override model.embedding=E_03_Patch
```

## 5) Coupling Notes

- Resolved by `src/model_factory/__init__.py:build_model`.
- Valid `model.type/model.name` combinations are indexed in `src/model_factory/model_registry.csv`.

## 6) How to Extend

1) Add a model implementation under `src/model_factory/<TYPE>/`.
2) Update `src/model_factory/model_registry.csv` with `model.type/model.name/module_path`.

## 7) Common Pitfalls

1) `model.type/model.name` mismatch to filesystem/module.
2) ISFM missing `embedding/backbone/task_head` ids.
3) `output_dim` mismatch across embedding/backbone/head (inspect with `scripts/config_inspect.py`).
4) Using a GPU-only model on CPU without guarding (set trainer.device=cpu and choose compatible model).
5) Copying paper-only component ids that are not in this repo.


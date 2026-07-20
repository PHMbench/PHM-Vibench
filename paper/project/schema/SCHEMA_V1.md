# Paper2 Schema v1（统一输出规范 / 供6篇Paper共用）

> **唯一权威**：本规范由 Paper2（Explainable FD Toolkit）维护；所有 Paper1–6 的实验输出必须满足本 schema，才能进入“顶刊证据链”。  
> **版本号**：`paper2_schema_v1`（写入 `run_meta.yaml` 与 `metrics.json`）。

---

## 1) 输出目录结构（推荐）

每次实验运行（一次 dataset × model × seed）建议输出为：

```
<RUN_DIR>/
  run_meta.yaml
  metrics.json
  artifacts/
    figures/
    tables/
    logs/
```

其中 `<RUN_DIR>` 推荐命名为：

```
<paper_dir>/outputs/<dataset_id>/<model_id>/seed_<seed>/<timestamp>/
```

---

## 2) `run_meta.yaml`（运行元信息，必需）

### 2.1 必需字段（Minimum）

```yaml
schema_version: paper2_schema_v1

paper:
  paper_id: paper1   # paper1..paper6
  paper_dir: Paper/1D-2D_fusion_explainable

run:
  run_id: "20251215_153000"         # 推荐 timestamp
  seed: 42
  dataset_id: RM_001_CWRU           # 推荐使用 Vibench Name（见 data/vibench_dataset_catalog.md）
  dataset_numeric_id: 1             # 推荐：与 metadata 的 Dataset_id 对齐
  model_id: Fusion1D2D              # 统一模型ID
  explainer_id: intrinsic           # 可选：intrinsic/posthoc/llm/...

repro:
  command: "CUDA_VISIBLE_DEVICES=0 python main.py --config_dir ..."
  config_path: "configs/unified_baseline/config_Fusion1D2D.yaml"

git:
  commit: "UNKNOWN"
  dirty: true

env:
  python: "3.9.x"
  torch: "x.y.z"
  device: "cuda:0"

outputs:
  run_dir: "<RUN_DIR>"
  metrics_path: "<RUN_DIR>/metrics.json"
```

### 2.2 推荐字段（Recommended）

- `timestamps.start_utc` / `timestamps.end_utc` / `timestamps.duration_sec`
- `data.root_dir`（若可公开；若不可公开，用脱敏路径）
- `data.split`（train/val/test划分与比例、或fold信息）
- `notes`（关键异常/警告）

---

## 3) `metrics.json`（指标真源，必需）

### 3.1 必需字段（Minimum）

```json
{
  "schema_version": "paper2_schema_v1",
  "paper_id": "paper1",
  "dataset_id": "CWRU",
  "model_id": "Fusion1D2D",
  "seed": 42,
  "task": "classification",
  "split_metrics": {
    "test": {
      "accuracy": 0.0
    }
  },
  "explainability": {},
  "artifacts": {}
}
```

### 3.2 推荐字段（Recommended）

#### (1) 主性能
- `split_metrics.train/val/test`：
  - `accuracy`、`f1_macro`、`f1_weighted`、`auc_macro`（按任务可选）
  - `confusion_matrix_path`（可选）

#### (2) explainability（按统一协议对齐）
- `explainability.faithfulness`：
  - `del_k_auc`（或 `aopc`）
  - `curve_path`
- `explainability.stability`：
  - `spearman_mean`（或任务适配指标）
  - `curve_path`
- `explainability.efficiency`：
  - `time_ms_per_sample`
  - `vram_mb_peak`（可选）
- `explainability.sparsity`（Fuzzy必需）：
  - `rules_activated_mean`
  - `coverage`

#### (3) artifacts
- `artifacts.figures`：关键图路径列表
- `artifacts.tables`：关键表路径列表
- `artifacts.logs`：日志路径列表

---

## 4) 最小合规等级（用于验收）

- **L0（P0验收）**：`run_meta.yaml` + `metrics.json` 满足 Minimum 字段；并能通过 `validate_schema.py`。
- **L1（P1验收）**：包含主性能（至少 test accuracy + F1）+ explainability 三项（faithfulness/stability/efficiency）中的至少两项。
- **L2（P2/投稿包）**：多 seed 汇总（mean±std/CI）+ 多数据集 + 完整 explainability + 失败案例证据链。

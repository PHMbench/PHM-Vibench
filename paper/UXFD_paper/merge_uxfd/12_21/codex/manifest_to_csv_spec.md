# `manifest.json` → CSV 汇总规范（v1）

目的：让本科生也能“扫一遍 runs 目录 → 得到一张清晰的总表”，并且让论文表格生成脚本有稳定输入。

---

## 1) 输入

- 每个 run 的索引文件：`<run_dir>/artifacts/manifest.json`

最低要求字段（缺失允许，但要在 CSV 里留空）：
- `paper_id`, `preset_version`, `run_id`
- `config_snapshot`
- `metrics`（路径或列表）
- `figures_dir`
- `data_metadata_snapshot`
- `eligibility`（可选）
- `explain_dir` / `explain_summary`（可选）
- `distilled_dir`（可选）

---

## 2) 输出（推荐两个 CSV）

### 2.1 `reports/uxfd_runs.csv`（一行一个 run）

必备列（最小集合）：
- `paper_id`
- `preset_version`
- `run_id`
- `manifest_path`
- `config_snapshot`
- `metrics_path`
- `figures_dir`
- `explain_dir`
- `distilled_dir`
- `meta_source`
- `degraded`
- `missing_keys`
- `explain_ok`
- `explainer_id`
- `explain_reasons`
- `explain_summary_path`

指标列（从 metrics 中自动展开）：
- 统一列名前缀：`metric/<key>`
  - 例：`metric/acc`, `metric/f1`, `metric/auc`, `metric/loss`

### 2.2 `reports/uxfd_explain.csv`（可选，解释明细）

当你希望把 explain 的多方法/多样本统计也列出来时输出：
- 一行一个 run×explainer（或 run×domain×explainer）
- 建议列：
  - `paper_id,run_id,explainer_id,domain_id`
  - `faithfulness/deletion_auc,stability/perturb_std,efficiency/avg_ms`
  - `notes`（JSON 字符串）

---

## 3) 扁平化与字段来源规则（写死在 collect 脚本里）

### 3.1 路径列一律用相对路径

避免机器差异：
- 相对 run_dir：推荐
- 或相对 repo root：也可以，但必须统一

### 3.2 dict 展开规则

- dict 用列名 `prefix/field` 展开（不使用 `.`，避免与 override 点路径混淆）
- 例：
  - `meta_source`（来自 `data_metadata_snapshot.meta_source`）
  - `metric/acc`（来自 metrics 文件里的 `acc`）

### 3.3 list/复杂结构规则

- `missing_keys`：用 `;` join 成字符串（或 JSON 字符串，二选一固定）
- `explain_reasons`：建议 JSON 字符串（因为是 list of dict）

### 3.4 缺失字段处理

- 缺失任何字段：填空字符串（`""`），并可选在脚本打印 warning
- 绝不因为某个 run 缺 explain/agent 就让 collect 崩溃

---

## 4) 推荐的 collect CLI（示例）

```bash
python -m scripts.collect_uxfd_runs --input save/ --out_dir reports/
```

输出：
- `reports/uxfd_runs.csv`
- （可选）`reports/uxfd_explain.csv`


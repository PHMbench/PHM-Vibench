# UXFD Merge 未完成工作（12_27 汇总 SSOT）

范围：仅整理 UXFD merge（WP0–WP5）相关未完成项；不把主仓库其他历史 TODO 混入。

原始材料（请优先以原文为准）：
- @`paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`（最完整的任务清单）
- @`paper/UXFD_paper/merge_uxfd/12_23/ops_library_completion_plan.md`（WP0/WP1 执行细化）
- @`paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`（DoD 对照与当前完成度）
- @`paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`（上游缺口与目标落位）

---

## P0（阻塞项：先做，否则后续无法验证）

### WP0：Submodules 最小入口（至少 1 个 pilot）

目标（DoD 要求）：任意 1 篇 paper 的 submodule 内存在并可跑通：
- `paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml`
- `paper/UXFD_paper/<paper_id>/VIBENCH.md`

执行清单：@`paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`

验收建议（跑完后检查产物）：
- 运行：`python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml --override trainer.num_epochs=1`
- Post-run 检查：`python scripts/uxfd_postrun.py --config paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`

---

## P1（高优先：让 pilot 跑通所需“血肉”）

### WP1：UXFD 通用组件真正移植（不是骨架）

需要从上游移植并整理进主仓库的核心模块（按优先级）：
- `Signal_processing_2D.py`（2D 时频/BTFC）
- `Fusion1D2D*.py`（1D↔2D 融合）
- `FuzzyLogic*.py`（模糊逻辑/推理）
- `operator_attention*.py` / `OperatorAttention_*`（算子注意力/路由）
- `Logic_inference.py`（若 pilot/后续 paper 需要；注意 CPU-safe device 处理）

目标落位与阶段拆分：@`paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`
执行细化：@`paper/UXFD_paper/merge_uxfd/12_23/ops_library_completion_plan.md`

### WP1b：TSPN_UXFD 的增强壳（HookStore / registry / adapters）

目标：不改变核心数学结构前提下，补齐 final plan 对“可追溯/可解释”的工程壳（operator_id、layout、HookStore）。

参考：@`paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`

---

## P2（中优先：论文表格/对比实验/解释闭环）

### WP2：baselines 扩展（torch-only 优先）

目标：逐个移植上游 `model_collection` 的对比模型，注意额外依赖（如 `torch_geometric`）必须 optional 或不注册。

参考：@`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`
缺口清单：@`paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`

### WP3：explain_factory 真正执行（产出 summary）

当前：eligibility 已落盘，但 explainer 执行未接入（缺 `artifacts/explain/summary.json`）。

参考：@`paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`

---

## P3（可选：不阻塞当前里程碑）

### WP5：agent_factory（TODO-only 蒸馏落盘）

策略：先做 “TODO-only evidence 落盘”，默认不开网络、不调用 LLM。

参考：@`paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`

---

## 已完成（可复用的基础闭环）

- `artifacts/manifest.json` 固定产出：`src/trainer_factory/extensions/manifest.py`
- manifest → CSV：`scripts/collect_uxfd_runs.py`
- post-run 检查/离线绘图（独立脚本 + 配置）：`scripts/uxfd_postrun.py` + `paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`


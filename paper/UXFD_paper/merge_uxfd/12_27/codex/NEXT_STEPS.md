# 下一步建议顺序（12_27）

目标：用最少步骤把 UXFD merge 推进到“可验证、可扩展”的状态（KISS / 渐进式）。

---

## Step 1（P0）：先做 1 篇 pilot submodule（WP0）

按清单执行：@`paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`

最低验收：
- `python main.py --config paper/UXFD_paper/<pilot>/configs/vibench/min.yaml --override trainer.num_epochs=1`
- 产物存在：`<run_dir>/artifacts/manifest.json`

建议同时做 post-run 检查（不改训练流程）：
- `python scripts/uxfd_postrun.py --config paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`

---

## Step 2（P1）：补齐 pilot 所需的 UXFD 通用组件（WP1）

按依赖排序做最小可用 Copy+Adapter：
1) 2D 时频（`Signal_processing_2D.py`）
2) 1D↔2D fusion（`Fusion1D2D*.py`）
3) operator attention（如 pilot 需要）
4) fuzzy / logic inference（如 pilot 需要）

执行细化：@`paper/UXFD_paper/merge_uxfd/12_23/ops_library_completion_plan.md`
缺口清单：@`paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`

最低验收：
- pilot `min.yaml` 可跑通并产出 `manifest.json`（不要求解释/论文级全量）

---

## Step 3（P1→P2）：TSPN_UXFD 增强壳 + explain 执行（WP1b/WP3）

先做“壳”，再做“解释执行”：
1) HookStore/registry/layout adapters（不改变 forward 结果）
2) 接入至少 1 个 explainer 的实际执行（产出 `artifacts/explain/summary.json` 并写入 manifest）

参考：@`paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`

---

## Step 4（P2）：baselines 扩展（WP2）

torch-only 优先，重依赖默认不注册（或 optional import）。

参考：@`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`


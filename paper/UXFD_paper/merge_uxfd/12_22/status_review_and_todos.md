# Final Plan 执行复盘与 TODO（2025-12-22）

本文对照 `paper/UXFD_paper/merge_uxfd/12_18temp/codex/final_plan.md` 检查当前落地进度，并列出剩余 TODO。

## 0) 最终验收（DoD）对照

已完成（主仓库不依赖 submodule）：
- ✅ `python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`
- ✅ `python -m scripts.validate_configs`
- ✅ `python -m pytest test/`
  - 说明：MultiTaskPHM/重集成类测试已迁移到 `test/TODO/`，默认测试只保留最基础回归用例

未完成（依赖 paper submodule）：
- ❌ 任意 1 个 submodule 的 `paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml` 最小配置跑通
  - 当前仓库内尚未创建 7 个 paper submodule 目录与对应 configs/VIBENCH.md（见 WP0）

## 1) Work Packages（WP0–WP5）状态

### WP0：Submodule 落位与入口文档（未开始）

目标（final plan 1.1）：
- `paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml`
- `paper/UXFD_paper/<paper_id>/VIBENCH.md`

现状：
- 7 个 paper 仓库已作为 submodule 初始化到 `paper/UXFD_paper/*`（见 `paper/UXFD_paper/README.md`）
- 但各 submodule 内尚未补齐 vibench 入口文件（`configs/vibench/min.yaml` 与 `VIBENCH.md`）

TODO：
- 在每个 submodule 内按模板写最小 config + `VIBENCH.md`
  - 模板：`paper/UXFD_paper/merge_uxfd/12_21/codex/VIBENCH_MAPPING_TEMPLATE.md`
  - 规范：`paper/UXFD_paper/merge_uxfd/12_21/codex/submodule_config_conventions.md`

### WP1：主仓库 UXFD 通用模块整理（部分完成：骨架已落位）

已完成：
- ✅ `src/model_factory/X_model/UXFD/` 目录骨架（含 README）
- ✅ `signal_processing_2d` 提供纯 PyTorch `STFTTimeFrequency`（作为可用最小 2D 时频算子）

未完成（核心 TODO）：
- ❌ 从上游 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/` 移植并整理通用模块：
  - `Fusion1D2D*.py` → `UXFD/fusion/`
  - `FuzzyLogic*.py` → `UXFD/fuzzy/`
  - `operator_attention*.py` → `UXFD/operator_attention/`
  - `Signal_processing_2D.py`（更贴近上游版本）→ `UXFD/signal_processing_2d/`
- ❌ `TSPN_UXFD` 的增强能力（final plan 3.1 要求）尚未补齐：
  - 稳定 `operator_id` / registry（对应上游 `ALL_SP/ALL_FE` 的“稳定命名+注册”）
  - layout adapter（BLC/BCL/BTFC 明确化）
  - HookStore（统一收集中间量）

当前折中实现：
- `src/model_factory/X_model/TSPN_UXFD.py` 只是对现有 `TSPN.py` 的稳定别名（不改变数学结构，但缺少 HookStore 等增强壳）

### WP2：对比 baselines 整理（部分完成：最小基线已接入）

已完成：
- ✅ `src/model_factory/X_model/baselines/` 落位 + README
- ✅ `BASE_ExplainableCNN`（torch-only）已可被 `model_factory` 加载
- ✅ `src/model_factory/model_registry.csv` 已登记 `BASE_ExplainableCNN`

TODO：
- ❌ 按 `paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md` 继续移植更多 baseline
- ❌ 明确哪些 baseline 需要额外依赖（如 `torch_geometric`）并做 optional-import 或不注册

### WP3：explain_factory 落地（部分完成：eligibility+1 个 explainer skeleton）

已完成：
- ✅ `src/explain_factory/` 落位（含 `eligibility.py` / `metadata_reader.py`）
- ✅ `gradcam_xfd.py`（最小 1D Grad-CAM）已存在
- ✅ 默认 pipeline 会写：
  - `artifacts/data_metadata_snapshot.json`
  - explain.enable=true 时写 `artifacts/explain/eligibility.json`

TODO：
- ❌ 真正执行 explainer（生成 `artifacts/explain/summary.json` 等）尚未接入
  - 当前仅有 eligibility 与示例 explainer，实现未被 pipeline/trainer 调用
- ❌ 按 explainer 类型定义 required meta（例如采样率、窗长、stride）并形成稳定 schema

### WP4：report+manifest + collect→CSV（完成：闭环可用）

已完成：
- ✅ 每次 run 产出 `artifacts/manifest.json`
- ✅ `scripts/collect_uxfd_runs.py` 可将 manifests 扁平化到 CSV
- ✅ 本次样例与字段说明已整理到：
  - `paper/UXFD_paper/merge_uxfd/12_22/scripts_and_outputs.md`
  - `paper/UXFD_paper/merge_uxfd/12_22/results/README.md`

注意：
- manifest 在 `fit_end` 与 `test_end` 都会写；pipeline 会在 `test_result_*.csv` 落盘后再补写一次，确保 `metrics_path` 不为空

### WP5：agent_factory（未开始，允许保持 TODO）

现状：
- 未实现 `src/agent_factory/`（按 final plan 可选）

TODO：
- 后续如果要做 LLM 解释/蒸馏：建议先做 “TODO-only evidence 落盘”，默认不启用网络/不调用 LLM

## 2) 当前“最重要的阻塞 TODO”（建议优先级）

P0（阻塞 final plan 1.1/DoD 的）：
1) WP0：至少完成 1 个 paper submodule 的 `configs/vibench/min.yaml` + `VIBENCH.md` 并跑通

P1（核心科研复用能力）：
2) WP1：把上游 UXFD 模块真正移植进 `src/model_factory/X_model/UXFD/**`（不是占位）
3) WP1：给 `TSPN_UXFD` 增加 HookStore/registry/adapters（不改变上游计算范式）

P2（论文表格/对比实验）：
4) WP2：补齐更多 baseline（尽量 torch-only），并给每个 baseline 提供最小 vibench config（放在各自 paper submodule 内）

P3（解释增强）：
5) WP3：把至少一个 explainer 的实际执行接入到 pipeline/trainer（产出 `summary.json` 并写入 manifest）

## 3) 相关材料索引

- Final SSOT：`paper/UXFD_paper/merge_uxfd/12_18temp/codex/final_plan.md`
- 本次脚本/样例：`paper/UXFD_paper/merge_uxfd/12_22/scripts_and_outputs.md`
- 本次失败分析：`paper/UXFD_paper/merge_uxfd/12_22/failures_report.md`
- 上游缺口分析（Unified_X_fault_diagnosis）：`paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`

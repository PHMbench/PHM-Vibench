# UXFD 合并文档索引

## 文档结构
- `12_18temp/`: 最初规划文档（历史存档）
- `12_21/`: 最终实施文档（当前版本）
- `12_22/`: 本次实现产物整理 + 失败分析报告
- `12_23/`: 信号处理/特征提取/逻辑推理算子库补全计划（待确认后实施）

## 快速导航

### 🎯 核心文档
- **[最终计划](12_18temp/codex/final_plan.md)**: UXFD 合并的完整决策文档
- **[逐步操作](12_21/codex/step_by_step_ops.md)**: 可执行的详细步骤
- **[配置规范](12_21/codex/submodule_config_conventions.md)**: submodule 配置文件规范
- **[本次落地整理](12_22/README.md)**: 脚本/产物样例/失败分析（2025-12-22）
- **[Final Plan 复盘+TODO](12_22/status_review_and_todos.md)**: 对照 final plan 的完成度与剩余任务
- **[算子库补全计划](12_23/ops_library_completion_plan.md)**: 12/23 待确认的实现计划（确认后再改代码）

### 📋 技术文档
- [模型对齐方案](12_21/codex/model_alignment_plan.md): TSPN_UXFD 设计
- [对比模型集成](12_21/codex/model_collection_integration_plan.md): 基线模型移植
- [Manifest 规范](12_21/codex/manifest_to_csv_spec.md): 证据链索引格式

### 📝 模板
- [VIBENCH 映射模板](12_21/codex/VIBENCH_MAPPING_TEMPLATE.md): 每篇 paper 的使用指南

## 执行顺序
1. 阅读 `final_plan.md` 了解整体方案
2. 按照 `step_by_step_ops.md` 执行具体步骤
3. 参考各技术文档进行专项实施

## 文件说明

### 12_18temp/codex/
- `init_plan.md`: 最初的规划草案，已标记为收敛版本
- `final_plan.md`: 最终的完整执行计划（SSOT）

### 12_21/codex/
- `step_by_step_ops.md`: 详细的逐步操作指南，本科生可照做
- `submodule_config_conventions.md`: submodule 内配置文件的写作规范
- `model_alignment_plan.md`: 如何保持与上游模型的范式一致
- `model_collection_integration_plan.md`: 对比基线模型的集成方案
- `manifest_to_csv_spec.md`: manifest.json 转换为 CSV 的规范
- `VIBENCH_MAPPING_TEMPLATE.md`: 每篇 paper 的 VIBENCH.md 模板

### 12_22/
- `scripts_and_outputs.md`: 本次新增脚本/产物清单（以及如何复现）
- `failures_report.md`: `pytest test/` 失败与问题分析（含修复建议）
- `results/`: 一次 smoke run 的“证据链闭环”样例（小文件，可直接打开）

## 7 篇 Paper 列表
1. `paper/UXFD_paper/1D-2D_fusion_explainable` - 1D-2D 融合可解释
2. `paper/UXFD_paper/Explainable_FD_Toolkit` - 可解释故障诊断工具包
3. `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` - LLM 增强的可解释工具包
4. `paper/UXFD_paper/MOE_explainable` - MoE 可解释方法
5. `paper/UXFD_paper/Paper_fuzzy_XFD` - 模糊逻辑可解释方法
6. `paper/UXFD_paper/Neuralsymbolic_theory` - 神经符号理论
7. `paper/UXFD_paper/TII_operator_attention` - 算子注意力机制

## 联系方式
如有疑问，请参考各文档中的详细说明或查看项目主 README。

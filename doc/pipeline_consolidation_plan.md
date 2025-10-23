# Pipelines 02/03/04 合并计划（两阶段统一管线）

本方案回答：Pipeline_02_pretrain_fewshot、Pipeline_03_multitask_pretrain_finetune、Pipeline_04_unified_metric 实现的核心功能是否一致？如何在不破坏现有工作流的前提下，逐步合并为一个可配置的两阶段统一管线，并保留原有入口作为“兼容外壳（thin wrappers）”。

## 1. 现状评估（是否一致？）

- 共同点
  - 两阶段流程：Stage 1 预训练 → 生成/选择最佳 checkpoint → Stage 2 下游/适配/微调。
  - 统一的工厂架构：data_factory / model_factory / task_factory / trainer_factory。
  - 相同的编排动作：load_config → build_data → build_model → build_task → build_trainer → fit → test。
  - 产物一致性：保存 checkpoint、metrics、日志，以及配置备份。
- 差异点
  - 预训练范式：
    - 02：多源预训练 + few-shot 适配（K‑shot/episode 语义）。
    - 03：无监督/重构/对比式预训练 + 单/多任务微调，支持骨干对比与多任务头。
    - 04：统一度量学习（多数据集联合预训练）+ 数据集级微调，配套 unified_metric 的脚本。
  - 采样与任务：few-shot（02）/ multi-task（03）/ unified-metric（04），具体采样器与任务头不同。
  - 配置结构：Stage 1/2 的键名与组织形式略有差异，04 有配套脚本生态（quick_validate / run_unified_experiments）。

结论：三者在“编排层面的核心功能是一致的”，差异主要体现在“任务/采样/损失/评估”层，完全可以通过“统一编排 + 任务适配”的方式合并。

## 2. 统一设计（Two‑Stage Orchestrator）

- 核心抽象
  - Orchestrator：`run_stage(stage_cfg) → (ckpt_path, metrics)`；`run_pipeline(cfg) → summary`。
  - Stage 类型：`pretrain_mode ∈ {contrastive, reconstruction, unified_metric, custom}`；`adapt_mode ∈ {fewshot, finetune, multitask}`。
  - 任务/采样插件：通过已有工厂（task_factory/data_factory）的注册机制切换行为。
- 配置统一
  - 推荐统一键空间：
    - `training.stage_1_pretraining.*`（或 `stage_1.*`），包含：backbones、datasets、epochs、losses、sampler。
    - `training.stage_2_finetuning.*`（或 `stage_2.*`），包含：模式（fewshot/finetune/multitask）、目标系统、冻结策略、lr 等。
  - 兼容层（adapters）：为 02/03/04 旧配置生成上述统一结构（只在编排入口转换，不触及任务实现）。
- 产物/命名统一
  - 统一保存：`save/{metadata}/{model}/{task_trainer_timestamp}/`，并确保 checkpoint 命名可被 Stage 2 稳定解析。

## 3. 合并步骤（渐进式）

### Phase A（抽取与落地）
1. 抽取通用编排逻辑 → `src/utils/training/two_stage_orchestrator.py`
   - `run_pretrain(cfg) / run_adapt(cfg, ckpt) / run_complete(cfg)`
   - 复用已有 `TwoStageController.py` 中能重用的接口/日志格式。
2. 统一 checkpoint 选择逻辑（ModelCheckpoint.best_model_path）与 metrics 提取函数。
3. 引入“本机覆盖”入口，使用 `merge_with_local_override()`（已在 `src/configs/config_utils.py`）。

### Phase B（外壳重用）
4. 将 `Pipeline_02_pretrain_fewshot` 改为薄外壳：
   - 读取 02 的 YAML → adapter → 统一 cfg → 调用 orchestrator。
5. 将 `Pipeline_03_multitask_pretrain_finetune` 改为薄外壳：
   - 读取 03 的 YAML → adapter → 统一 cfg → 调用 orchestrator。
6. 将 `Pipeline_04_unified_metric` 改为薄外壳：
   - 读取 04 的 YAML → adapter → 统一 cfg → 调用 orchestrator（仍保留 `quick_validate/run_unified_experiments` 脚本）。

### Phase C（一致性与回归）
7. 测试矩阵：
   - 单元：orchestrator 的 `run_pretrain/run_adapt` 接口（mock trainer/data）。
   - 集成：用 02/03/04 的最小 YAML 各跑 1 epoch，比较指标与旧版一致。
   - 性能：确保统一层不引入额外 O(n²) 复杂度。
8. 文档与去重：
   - `src/readme.md` 与 `doc/*` 更新统一入口与差异点。
   - 标注 02/03/04 为“兼容外壳”，新功能优先在 orchestrator 实现。

## 4. 配置适配（Adapters）
- 目标：`old_cfg → unified_cfg`
- 方案：新增 `src/utils/config/pipeline_adapters.py`：
  - `adapt_p02(cfg) → unified_cfg`
  - `adapt_p03(cfg) → unified_cfg`
  - `adapt_p04(cfg) → unified_cfg`
- 适配点：
  - Stage 键名映射（如 `stage_1_pretraining` / `training_stage: pretrain`）。
  - 少量默认值补齐（backbone 列表/epochs/seed）。
  - few‑shot/统一度量的采样与损失字段映射到 task_factory 的已注册任务。

## 5. 目录与命名
- 新增：`src/utils/training/two_stage_orchestrator.py`
- 新增：`src/utils/config/pipeline_adapters.py`
- 保留：`Pipeline_02_* / Pipeline_03_* / Pipeline_04_*`（薄外壳）
- 逐步将新实验推荐入口汇总为：`python -m src.Pipeline_TwoStage --config_path ... [--local_config ...]`（等稳定后再引入）

## 6. 兼容与迁移
- 原入口仍可用（语义不变），内部实现改为：旧 YAML → adapter → orchestrator。
- 新增统一 YAML 模板示例（`configs/experiments/two_stage_*`）。
- 2 个小版本的迁移期：鼓励用户逐步迁移至统一 YAML；文档提供“前后对照”与一键转换脚本（可选）。

## 7. 风险与回滚
- 风险：
  - 旧脚本依赖特殊 side‑effects（如自定义日志名/路径） → 通过 adapter 注入保持一致。
  - 指标名不一致导致外部分析脚本断裂 → 提供向后兼容 aliases（统一导出键名 + 旧键名）。
- 回滚：
  - 保留旧版 pipeline 的 tag；orchestrator 引入开关以禁用新路径，临时回退。

## 8. 测试计划（最小集）
- 单元：
  - `test/test_orchestrator.py`：mock 训练一次，断言 ckpt 路径和 metrics 字段存在。
  - `test/test_pipeline_adapters.py`：输入旧 YAML 片段，输出 unified cfg 有预期键。
- 集成：
  - 1‑epoch CPU 跑 02/03/04 的最小配置，对比“训练是否完成、是否生成 ckpt、是否有 metrics.json”。

## 9. 里程碑与时间线
- W1：抽取 Orchestrator + 统一 ckpt/metrics 接口（A1~A3）。
- W2：02/03 适配为薄外壳（B4~B5）+ 单元/集成测试（C7）。
- W3：04 接入（B6），保留脚本生态，完成文档（C8）。
- W4：统一 YAML 示例与迁移指南，标记稳定版本。

## 10. 验收标准
- 以统一 Orchestrator 跑通 02/03/04 的 1‑epoch 样例，无需更改任务实现。
- 新旧入口的指标与产物路径一致或在文档中给出清晰映射。
- PR 中附带：测试结果截图/metrics、配置对照表、迁移说明。

---

该合并计划在不牺牲灵活性的前提下，最大化去重与一致性。合并后，团队只需维护一个“两阶段编排内核”，具体“预训练/适配”差异由任务/采样插件层解决，后续扩展与对比实验成本显著下降。

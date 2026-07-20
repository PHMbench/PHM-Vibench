## CDDG_config 目录说明（Work A 视角）

本目录规划为 **Work A – Foundation Model for Fleet-Level Generalization（CDDG）** 的专用配置集合。
目标是为 Experiment 0–7 提供面向 CDDG 任务的清晰入口，同时与统一的
`configs/experiment_X_unified.yaml` 保持一一对应关系。

> 当前阶段：命名规范与结构已确定，部分 YAML 仍直接复用顶层 unified 配置，后续可按需要拆出
> 独立副本或通过覆盖机制精简字段。

### 1. 计划中的配置文件命名

- `experiment_0_cddg_baseline.yaml`  
  - 对应 `configs/experiment_0_backbone_head.yaml`  
  - 单系统基线（Backbone+Head），CDDG 视角的独立训练。

- `experiment_1_cddg_hse.yaml`  
  - 对应 `configs/experiment_1_unified.yaml`  
  - HSE 直接 CDDG 分类。

- `experiment_2_cddg_hse_pretrain.yaml`  
  - 对应 `configs/experiment_2_unified.yaml`（两阶段 unified）  
  - HSE 对比预训练 + CDDG 微调。

- `experiment_3_cddg_hse_prompt.yaml`  
  - 对应 `configs/experiment_3_unified.yaml`  
  - HSE-Prompt + CDDG，两阶段训练。

- `experiment_4_cddg_ablation.yaml`  
  - 来源于原 `CDDG+config/experiment_4_cddg.yaml` / `configs/experiment_4_unified.yaml`  
  - 组件消融（CDDG 任务）。

- `experiment_5_cddg_optional.yaml`（可选）  
  - CDDG 视角下的 few-shot 梯度扫描对照版本。

- `experiment_6_cddg_backbone.yaml`  
  - 来源于原 `CDDG+config/experiment_6_cddg.yaml` / `configs/experiment_6_unified.yaml`  
  - Backbone 普适性验证（CDDG 任务）。

- `experiment_7_cddg_noise.yaml`  
  - 来源于原 `CDDG+config/experiment_7_cddg.yaml` / `configs/experiment_7_unified.yaml`  
  - 噪声鲁棒性（CDDG 任务）。

### 2. 与 unified 配置的关系

- 顶层 `configs/experiment_X_unified.yaml` 仍是 Vbench 推荐的统一入口（单 YAML + stages）；  
- `CDDG_config/experiment_X_cddg_*.yaml` 可以：
  - 直接引用 unified 配置（通过路径或 `base_config` 语义，视后续实现而定），或  
  - 复制 unified 内容并针对 CDDG 视角做最小修改（如 task.type、metrics 聚合方式）。  

### 3. 后续演进建议

- 在完成 Work A 的整体验证前，建议优先将 Experiment 0–3 的 CDDG 配置稳定下来；  
- Experiment 4–7 的 CDDG 版本可在组件消融 / Backbone / 噪声实验稳定后再精简与固化；  
- 所有 CDDG_config 中的路径应与 README 的 Work A 表格（Table A1 / A2）保持一致，便于论文结果回填。


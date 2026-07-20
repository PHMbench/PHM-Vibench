## GFS_config 目录说明（Work B 视角）

本目录规划为 **Work B – Foundation Model for Fleet-Level Few-Shot Learning（GFS）** 的专用配置集合。
目标是为 Experiment 0–7 提供面向 GFS / Few-shot 任务的入口，同时复用统一的
`configs/experiment_X_unified.yaml` 中的模型与数据设定。

> 当前阶段：仅确定命名规范与目录结构。大部分 GFS 配置预计基于 unified YAML
> 通过少量字段覆盖（task.type、sampler、few-shot 超参）实现，无需完整复制。

### 1. 计划中的配置文件命名

- `experiment_0_gfs_baseline.yaml`（可选）  
  - GFS 视角下的 Backbone+Head few-shot baseline。

- `experiment_1_gfs_hse.yaml`（可选）  
  - HSE 直接 few-shot / GFS 分类。

- `experiment_2_gfs_hse_pretrain.yaml`（可选）  
  - HSE 对比预训练 + GFS 下游任务。

- `experiment_3_gfs_hse_prompt.yaml`  
  - 对应 `configs/experiment_3_unified.yaml`，HSE-Prompt Few-shot 主方法。

- `experiment_4_gfs_ablation.yaml`  
  - 对应 `configs/experiment_4_unified.yaml`，GFS 视角下的组件消融。

- `experiment_5_gfs_shot_sweep.yaml`  
  - 对应 `configs/experiment_5_unified.yaml`，Few-shot 梯度扫描主实验。

- `experiment_6_gfs_backbone.yaml`  
  - 对应 `configs/experiment_6_unified.yaml`，Backbone 普适性（GFS 任务）。

- `experiment_7_gfs_noise.yaml`  
  - 对应 `configs/experiment_7_unified.yaml`，噪声鲁棒性（GFS 任务）。

### 2. 与 unified 配置的关系

- `experiment_X_unified.yaml` 负责描述两阶段训练逻辑（预训练 + 下游任务）以及共享的 embedding / backbone。  
- `GFS_config/experiment_X_gfs_*.yaml` 主要负责：
  - 选择 GFS / FS / GFS+CDDG 等 task.type 与 sampler；  
  - 设定 episodic 超参（num_episodes, num_ways, num_shots 等）；  
  - 保持与 Work A 的模型容量一致，便于对比 few-shot 与 CDDG 的样本效率。

### 3. 文档与表格对齐

- 所有 GFS_config 中的文件名应与 README 中 Work B 部分的 Table B1 / B2 一一对应；  
- 在 `docs/training_loop_validation_plan_0-7.md` 中，为每个 Experiment 提供一条 GFS 验证命令，引用本目录下的 YAML。


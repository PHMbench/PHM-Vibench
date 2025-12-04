# Config Registry & Demo Validation Plan (v0.1.0)

本文件记录两件事：

1. `configs/config_registry.csv` 的设计方案（仅针对 base 与 demo，不包含 reference，方便后续你删除 reference 目录）。
2. v0.1.0 期望的 6 个代表性实验的验证计划（通过后可以在 CSV 和 README 中标记为已验证）。

-----

## 一、configs/config_registry.csv 设计（仅 base + demo）

### 1.1 文件位置与作用

- 路径：`configs/config_registry.csv`
- 作用：作为 v0.1.0 配置层的唯一索引，回答两个问题：
  - 这份 YAML 是 base 还是 demo？
  - 对于 demo，它是由哪些 base environment / data / model / task / trainer 组合而成的？

### 1.2 字段设计（不影响 YAML，仅作元信息）

建议表头：

```text
id,category,path,description,pipeline,base_environment,base_data,base_model,base_task,base_trainer,status
```

字段含义：

- `id`: 简短 ID，例如：
  - `base_env_default`, `base_data_cross_domain`, `demo_01_cross_domain`。
- `category`: 固定枚举：
  - `base_environment` / `base_data` / `base_model` / `base_task` / `base_trainer` / `demo`。
- `path`: 配置文件路径（相对仓库根），例如：
  - `configs/base/data/base_cross_domain.yaml`
  - `configs/demo/02_cross_system/multi_system_cddg.yaml`
- `description`: 一句话用途说明。
  - base：比如 “单数据集分类/DG data base”；
  - demo：比如 “多系统 CDDG demo（v0.1.0 示例）”。
- `pipeline`: 仅对 `category=demo` 填写，对 base 行留空。
  - 与 YAML 顶层 `pipeline.name` 一致，例如 `Pipeline_01_default` / `Pipeline_02_pretrain_fewshot`。
- `base_environment` / `base_data` / `base_model` / `base_task` / `base_trainer`:
  - 对 `category=demo` 填写它通过 `base_configs` 引用的 base 路径；
  - 对 base 行留空或 `/`。
- `status`:
  - `/`：初始状态，尚未验证；
  - `sanity_ok`：已按下面的验证计划完成 1 轮本地 sanity；
  - `deprecated`：未来如果某个 demo/base 不再推荐使用，可以标这里。

### 1.3 填表策略（仅 base + demo）

1. **为每个 base 填一行**：

   示例：

   ```text
   base_env_default,base_environment,configs/base/environment/base.yaml,"通用 environment base（PROJECT_HOME + iterations）",,,,,,/ 
   base_data_cross_domain,base_data,configs/base/data/base_cross_domain.yaml,"单数据集 cross-domain DG data base",,,,,,/ 
   base_model_isfm_hse,base_model,configs/base/model/backbone_dlinear.yaml,"M_01_ISFM + E_01_HSE + B_04_Dlinear + H_01_Linear_cla",,,,,,/ 
   base_task_cddg,base_task,configs/base/task/cddg.yaml,"多系统 CDDG 分类任务 base",,,,,,/ 
   base_trainer_single_gpu,base_trainer,configs/base/trainer/default_single_gpu.yaml,"单 GPU 默认 Trainer",,,,,,/ 
   ```

2. **为每个 v0.1.0 demo 填一行**（不记录 reference，便于将来删除）

   示例（cross-system CDDG demo）：

   ```text
   demo_02_cross_system,demo,configs/demo/02_cross_system/multi_system_cddg.yaml,"多系统 CDDG demo（v0.1.0）",Pipeline_01_default,configs/base/environment/base.yaml,configs/base/data/base_cross_system.yaml,configs/base/model/backbone_dlinear.yaml,configs/base/task/cddg.yaml,configs/base/trainer/default_single_gpu.yaml,/
   ```

   其他 demo 按同样模式填充，只改变 path 与描述。

> 注意：**不在 CSV 中记录 `configs/reference/experiment_*.yaml`**，以便你后续清理 reference 时不会引入额外维护负担。

-----

## 二、6 个代表性实验的验证计划（写给 v0.1.0 使用）

下列 6 个实验均基于 v0.1.0 的 base + demo 体系，统一使用：

- `data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"`
- `metadata_file: "metadata.xlsx"`
- 模型：

  ```yaml
  model:
    name: "M_01_ISFM"
    type: "ISFM"
    embedding: "E_01_HSE"
    backbone: "B_04_Dlinear"
    task_head: "H_01_Linear_cla"
  ```

**通用前提**：

- 确保本地存在 `"/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx"`，并与当前 data_factory 管道兼容；
- 推荐入口命令模式：

  ```bash
  python main.py --config <demo_yaml> --override trainer.num_epochs=1
  ```

### 2.1 实验 #1：Cross-domain DG demo（单数据集 DG）

- 配置文件：`configs/demo/01_cross_domain/cwru_to_ottawa_dg.yaml`
- pipeline：`pipeline.name = "Pipeline_01_default"`
- 基础组件：
  - environment: `configs/base/environment/base.yaml`（demo 中覆盖 project/output_dir/seed/notes）
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/dg.yaml`（`task.type="DG"`）
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- 验证步骤：

  ```bash
  python main.py \
    --config configs/demo/01_cross_domain/cwru_to_ottawa_dg.yaml \
    --override trainer.num_epochs=1
  ```

- 验证要点：
  - `load_config()` 能正常解析 base + demo，`cfg.task.type == "DG"`；
  - `Pipeline_01_default` 完成一轮 train/val/test，不因 config 结构报错；
  - 输出结果写入 `results/demo/cwru_to_ottawa_dg/...`。

> 验证通过后，可在 `configs/config_registry.csv` 对应行的 `status` 列填 `sanity_ok`。

### 2.2 实验 #2：Cross-system CDDG demo（多系统 CDDG）

- 配置文件：`configs/demo/02_cross_system/multi_system_cddg.yaml`
- pipeline：`Pipeline_01_default`
- 基础组件：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_cross_system.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/cddg.yaml`（`task.type="CDDG"`）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 验证步骤：

  ```bash
  python main.py \
    --config configs/demo/02_cross_system/multi_system_cddg.yaml \
    --override trainer.num_epochs=1
  ```

- 验证要点：
  - CDDG 路径可跑通（不因 task.type/name 导致 task 导入失败）；
  - `target_system_id` 等字段被正常使用；
  - 结果写入 `results/demo/multi_system_cddg/...`。

### 2.3 实验 #3：Few-shot demo（单系统 few-shot）

- 配置文件：`configs/demo/03_fewshot/cwru_protonet.yaml`
- pipeline：`Pipeline_01_default`
- 基础组件：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_fewshot.yaml`（`window_size=1024`）
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/fewshot.yaml`（`task.type="FS"`）
  - trainer: `base/trainer/default_single_gpu.yaml`（demo 覆盖 `num_epochs=1`）
- 验证步骤：

  ```bash
  python main.py \
    --config configs/demo/03_fewshot/cwru_protonet.yaml \
    --override trainer.num_epochs=1
  ```

- 验证要点：
  - `task.type == "FS"`，能正确选择 few-shot 任务模块（FS 目录下）；
  - 支持 few-shot episode 的数据加载流程不报 config 相关错误。

### 2.4 实验 #4：Cross-system few-shot demo

- 配置文件：`configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
- pipeline：`Pipeline_01_default`
- 基础组件：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_cross_system_fewshot.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/cddg_fewshot.yaml`（`task.type="FS"`，用于跨系统 few-shot 设置）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 验证步骤：

  ```bash
  python main.py \
    --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
    --override trainer.num_epochs=1
  ```

- 验证要点：
  - 在跨系统 few-shot 配置下（`task` 中指定 `source_system_id` / `target_system_id`）仍能正常构建数据与任务；
  - 不因 task.type/name 不匹配导致任务模块导入失败。

### 2.5 实验 #5：Pretrain + few-shot 两阶段 demo

- 配置文件：`configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- pipeline：`Pipeline_02_pretrain_fewshot`
- 基础组件：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_classification.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/pretrain.yaml`（demo 覆盖 `name: "pretrain_then_fewshot"`, `type: "pretrain"`）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 验证步骤（先以“快速预训练”视角）：

  ```bash
  python main.py \
    --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
    --override trainer.num_epochs=1
  ```

- 验证要点：
  - `pipeline.name` 能正确选中 `Pipeline_02_pretrain_fewshot`；
  - 至少预训练阶段能完整跑完一轮；
  - few-shot 阶段如果当前配置未完整接好，可以先接受“跳过/告警”的行为，但不得因为 config 结构直接崩溃。

### 2.6 实验 #6：Pretrain HSE for CDDG demo

- 配置文件：`configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`
- pipeline：`Pipeline_01_default`（单阶段 HSE CDDG 预训练）
- 基础组件：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_cross_system.yaml`（demo 覆盖 `train_ratio` / `stride`）
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/pretrain.yaml`（demo 覆盖 `name: "hse_contrastive"`, `type: "CDDG"`）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 验证步骤：

  ```bash
  python main.py \
    --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
    --override trainer.num_epochs=1
  ```

- 验证要点：
  - `task.type == "CDDG"`, `task.name == "hse_contrastive"` 能成功选中 `task/pretrain/hse_contrastive.py`；
  - 至少完成 1 轮 HSE 风格的训练，不因配置缺失或类型不匹配崩溃。

-----

## 三、验证与 CSV 状态联动建议

当某个 demo 按上述步骤完成 sanity 验证后，可以：

1. 在 `configs/config_registry.csv` 中，将对应 demo 行的 `status` 从 `/` 更新为 `sanity_ok`；
2. 在 `docs/v0.1.0/v0.1.0_update.md` 或 `release_v0.1.0_planning.md` 中简单记一行，例如：

   ```markdown
   - [x] demo_02_cross_system — sanity_ok @ 2025-xx-xx （命令：python main.py --config ...）
   ```

这样，后续你在真正合并 `release/v0.1.0` 时，可以一眼看出 “哪些 YAML 已经跑过、哪些还只是设计稿”。***


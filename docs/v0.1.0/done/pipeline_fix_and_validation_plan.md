# Pipeline接口修复与v0.1.0验证计划

## 问题概述（已修复的历史问题）

早期在执行 `python main.py --config <demo_yaml>` 时，`Pipeline_01_default.py` 曾因为访问 `VBENCH_HOME` 而触发：

```text
AttributeError: 'ConfigWrapper' object has no attribute 'VBENCH_HOME'
```

根本原因当时是：

- pipeline 接口期望 `args_environment.VBENCH_HOME` 等字段；
- 新的配置体系里，environment 不再保证提供 VBENCH\_* 字段，且未来希望统一为 `PROJECT_HOME`。

当前 v0.1.0 设计中，这个问题已经通过“移除硬依赖 VBENCH\_HOME”解决。

## 第一阶段：Pipeline 接口与环境变量策略（现状说明）

### 1.1 main → pipeline 的接口约定

- `main.py`：
  - 解析 CLI：推荐入口为 `--config <yaml 或预设名>`，兼容旧参数 `--config_path`；
  - 从 YAML 顶层读取 `pipeline`：
    - 若写成 `pipeline: "Pipeline_01_default"`，则导入 `src.Pipeline_01_default`；
    - 若写成：

      ```yaml
      pipeline:
        name: "Pipeline_02_pretrain_fewshot"
      ```

      则使用 `name` 字段；
    - 未指定时默认使用 `Pipeline_01_default`；
  - 将最终解析出的路径写回 `args.config_path`，传给 `pipeline(args)`。

- 各 pipeline（例如 `Pipeline_01_default`、`Pipeline_02_pretrain_fewshot`）统一接口：

  ```python
  def pipeline(args: argparse.Namespace):
      ...
  ```

### 1.2 环境变量：统一为 PROJECT_HOME，不再依赖 VBENCH\_HOME

- `configs/base/environment/base.yaml` 提供统一的环境字段：

  ```yaml
  environment:
    PROJECT_HOME: "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench"
    project: "demo_project"
    seed: 42
    output_dir: "results/demo"
    notes: ""
    iterations: 1
  ```

- `Pipeline_01_default.py` 现状：
  - 使用 `transfer_namespace(configs.environment)` 得到 `args_environment`；
  - 将 environment 中的大写字段写入 `os.environ`（例如 `PROJECT_HOME`）；
  - **不再访问 `args_environment.VBENCH_HOME` 或 `VBENCH_DATA`**，目录创建依赖 `environment.output_dir` + `path_name(configs, it)`。

- 其他 Pipeline（例如 `Pipeline_02_pretrain_fewshot`、`Pipeline_04_unified_metric`）：
  - 已统一按 “args → load_config → ConfigWrapper” 的方式读取配置；
  - 如仍有对 `VBENCH_HOME` 的访问，也应逐步迁移到 `PROJECT_HOME` 或完全移除硬依赖。

> 结论：当前 v0.1.0 下，入口接口已经统一，Pipeline 不再直接依赖 VBENCH\_* 字段；后续只需在新增/修改 Pipeline 时继续遵守这套约定。

## 第二阶段：执行 6 个代表性实验验证（细化版）

### 2.1 验证准备

**通用前提**：
- 确保数据路径存在：`/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx`
- 所有 demo 共享统一模型配置（由 base model 提供）：

  ```yaml
  model:
    name: "M_01_ISFM"
    type: "ISFM"
    embedding: "E_01_HSE"
    backbone: "B_04_Dlinear"
    task_head: "H_01_Linear_cla"
  ```

- 统一验证命令模式：

  ```bash
  python main.py --config <demo_yaml> --override trainer.num_epochs=1
  ```

- 验证标准：
  - 每个实验至少完成 1 轮训练（train/val/test 或预训练阶段）；
  - 不因 YAML 结构、`task.type/name`、`pipeline` 决策错误而崩溃；
  - 若失败仅由于数据文件缺失、网络下载失败等“环境问题”，记录为环境异常，不计入 config/pipeline BUG。

### 2.2 验证顺序（按复杂度递增）

#### 实验 #1：Cross-domain DG（单数据集 DG 示例）
```bash
# 配置：configs/demo/01_cross_domain/cwru_to_ottawa_dg.yaml
python main.py \
  --config configs/demo/01_cross_domain/cwru_to_ottawa_dg.yaml \
  --override trainer.num_epochs=1
```
**验证要点（通过后在 config_registry.csv 中将 demo_01_cross_domain 标记为 sanity_ok）：**

- `load_config()` 能正常解析 `base_configs + override`：
  - `cfg.data.data_dir == "/home/user/data/PHMbenchdata/PHM-Vibench"`；
  - `cfg.data.metadata_file == "metadata.xlsx"`；
  - `cfg.task.type == "DG"`；
  - `cfg.model.*` 为统一的 ISFM 配置。
- `Pipeline_01_default`：
  - 完成 1 次 train/val/test 流程；
  - `results/demo/cwru_to_ottawa_dg/` 下生成对应的 `test_result_0.csv` 等文件。

#### 实验 #2：Cross-system CDDG
```bash
# 配置：configs/demo/02_cross_system/multi_system_cddg.yaml
python main.py \
  --config configs/demo/02_cross_system/multi_system_cddg.yaml \
  --override trainer.num_epochs=1
```
**验证要点（对应 config_registry 中 demo_02_cross_system）：**

- `cfg.task.type == "CDDG"`，`cfg.task.name == "classification"`；
- 任务模块解析正确：应选中 `task/CDDG/classification.py`。  
- 运行 1 轮训练不会报 “Task 模块导入失败” 或 “缺少字段”；
- `task.target_system_id` 被用于数据划分（具体表现可通过日志或结果检查）；
- 结果写入 `results/demo/multi_system_cddg/`。

#### 实验 #3：Few-shot（单系统 few-shot）
```bash
# 配置：configs/demo/03_fewshot/cwru_protonet.yaml
python main.py \
  --config configs/demo/03_fewshot/cwru_protonet.yaml \
  --override trainer.num_epochs=1
```
**验证要点（对应 config_registry 中 demo_03_fewshot）：**

- `cfg.task.type == "FS"`；
- Few-shot 任务模块解析正确（`task/FS` 目录下实现）；
- 支持 few-shot episode 的数据加载逻辑能够运行：
  - 例如可以在日志中看到 episode 级别的迭代打印；
  - 不因 batch 形状或缺失字段崩溃。

#### 实验 #4：Cross-system few-shot
```bash
# 配置：configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml
python main.py \
  --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
  --override trainer.num_epochs=1
```
**验证要点（对应 config_registry 中 demo_04_cross_system_fewshot）：**

- `cfg.task.type == "FS"`（跨系统 few-shot 仍使用 FS 类型）；
- `task` 中的 `source_system_id` / `target_system_id` 会影响采样（可通过日志或 debug 检查一次）；
- 训练阶段能够跑完 1 轮，不因多系统 few-shot 的配置不一致而崩溃。

#### 实验 #5：Pretrain + few-shot（两阶段 demo）
```bash
# 配置：configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml
python main.py \
  --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
  --override trainer.num_epochs=1
```
**验证要点（对应 config_registry 中 demo_05_pretrain_fewshot）：**

- `pipeline.name == "Pipeline_02_pretrain_fewshot"`，`main.py` 能正确导入 `src.Pipeline_02_pretrain_fewshot`；
- 预训练阶段：
  - 能加载配置并构建模型/任务；
  - 至少跑完 1 轮 epoch；
- few-shot 阶段：
  - 即使当前配置较简化，也不因 config 结构问题直接崩溃；
  - 如果 few-shot 阶段暂未完全打通，可以在验证表中记录具体行为（例如“预训练 OK，few-shot 阶段暂时跳过”）。

#### 实验 #6：Pretrain CDDG（HSE 预训练）
```bash
# 配置：configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml
python main.py \
  --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
  --override trainer.num_epochs=1
```
**验证要点（对应 config_registry 中 demo_06_pretrain_cddg）：**

- `cfg.task.type == "CDDG"`, `cfg.task.name == "hse_contrastive"`；
- 任务模块解析正确：选中 `task/pretrain/hse_contrastive.py`；
- 至少完成 1 轮 HSE 风格训练：
  - 不因 task / model / data 字段缺失崩溃；
  - 学习率、weight decay 等基本优化参数来自 `task/pretrain.yaml` + demo override。

### 2.3 验证结果记录与 config_registry.csv 联动

- 验证结果建议记录在两个地方：

  1. 本文档或 `release_v0.1.0_planning.md` 中的表格（人工阅读用）；
  2. `configs/config_registry.csv` 中对应 demo 行的 `status` 列：
     - `/` → 尚未验证；
     - `sanity_ok` → 已通过上述命令完成至少一轮本地 sanity。

- `configs/config_registry.csv` 的字段设计与样例，详见 `docs/v0.1.0/codex/config_registry_plan.md`。该 CSV 已创建并填入 base 与 6 个 demo 条目，只需在验证后更新 `status`。

## 第三阶段：文档与发布（简要）

- 配置系统与 Pipeline 接口的变更已在以下文件中记录：
  - `docs/v0.1.0/configupdate.md`
  - `docs/v0.1.0/v0.1.0_update.md`
  - `docs/v0.1.0/codex/config_registry_plan.md`
  - `src/model_factory/README*.md`
  - `src/task_factory/readme.md`

- 待 6 个代表性 demo 标记为 `sanity_ok` 后，可在 v0.1.0 发布说明中：
  - 列出“官方支持的 demo 列表 + 启动命令”；
  - 简要说明配置体系（base + demo + registry）与 Pipeline 入口的统一方式。

> 成功标准（与 v0.1.0_update.md 中的 checklist 对齐）：
>
> - [x] Pipeline 接口统一为 `pipeline(args: Namespace)`，不再依赖 VBENCH\_*；
> - [x] `main.py` 通过 YAML 顶层 `pipeline` 选择具体 Pipeline；
> - [x] base/demo/config registry 文档与 CSV 均已搭好；
> - [ ] 6 个代表性 demo 完成 sanity 验证并在 `configs/config_registry.csv` 中标记为 `sanity_ok`；
> - [ ] BUG 列表中与 Pipeline / Config 相关的问题在 v0.1.0 范围内有清晰的“已修/延后”说明。

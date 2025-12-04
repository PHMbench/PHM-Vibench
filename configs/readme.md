# Configs 目录结构与组合说明（v0.1.0）

本目录包含三类配置：

- `base/`：data/model/task/trainer/environment 的基础模板；
- `demo/`：轻量级示例实验（v0.1.0）；
- `reference/`：与论文实验对应的完整配置（experiment_0~7）。

---

## 1. Base 配置模板

**data base（均使用 metadata.xlsx + 统一 data_dir）**

| 文件路径                                   | 作用场景                  | 核心字段（示意）                                                    |
| ------------------------------------------ | ------------------------- | ------------------------------------------------------------------- |
| `configs/base/data/base_classification.yaml`      | 单数据集分类 / DG         | `data_dir=/home/user/data/PHMbenchdata/PHM-Vibench`, `metadata_file=metadata.xlsx` |
| `configs/base/data/base_cross_domain.yaml`       | 单数据集 DG（cross-domain） | 同上，batch/window 等与 reference 实验对齐                         |
| `configs/base/data/base_cross_system.yaml`       | 多系统 CDDG               | 同上                                                                |
| `configs/base/data/base_fewshot.yaml`            | 单系统 few-shot           | `window_size=1024`, 其余字段同上                                   |
| `configs/base/data/base_cross_system_fewshot.yaml` | 跨系统 few-shot           | 与 `base_fewshot` 相同结构                                         |

**model base**

| 文件路径                                      | 来源参考                         | 说明                         |
| --------------------------------------------- | -------------------------------- | ---------------------------- |
| `configs/base/model/backbone_dlinear.yaml`    | `experiment_1_cddg_hse.yaml` 的 model 段 | ISFM + HSE + DLinear 组合    |
| `configs/base/model/backbone_transformer.yaml`| 早期 Single_DG Transformer demo  | 简化版 Transformer backbone  |

**task base**

| 文件路径                             | 类型    | 典型用途               |
| ------------------------------------ | ------- | ---------------------- |
| `configs/base/task/classification.yaml` | `DG`    | 单数据集分类 / 简单 DG |
| `configs/base/task/dg.yaml`             | `DG`    | cross-domain DG        |
| `configs/base/task/cddg.yaml`           | `CDDG`  | 多系统 CDDG            |
| `configs/base/task/fewshot.yaml`        | `FS`    | 单系统 few-shot (ProtoNet) |
| `configs/base/task/cddg_fewshot.yaml`   | `FS`    | 跨系统 few-shot（仍用 `FS` 类型） |
| `configs/base/task/pretrain.yaml`       | `pretrain` | HSE / ISFM 预训练       |

**trainer base**

| 文件路径                                      | 说明                    |
| --------------------------------------------- | ----------------------- |
| `configs/base/trainer/default_single_gpu.yaml`| 单 GPU 默认训练器       |

**environment base**

| 文件路径                                   | 说明                                      |
| ------------------------------------------ | ----------------------------------------- |
| `configs/base/environment/base.yaml`       | 提供 `PROJECT_HOME` / 默认 `iterations` 等 |

---

## 2. Demo 配置（如何由 base 组合而成）

**1）Cross-domain DG（单数据集 DG）**

| Demo 文件路径                                     | 使用的 base 组件                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------------- |
| `configs/demo/01_cross_domain/cwru_to_ottawa_dg.yaml` | `environment/base.yaml` + `data/base_cross_domain.yaml` + `model/backbone_dlinear.yaml` + `task/dg.yaml` + `trainer/default_single_gpu.yaml` |

- YAML 顶层通过 `base_configs` 引用上述 base；
- `task` 中使用 `source_domain_id` / `target_domain_id`（来自旧版 CWRU demo）表达不同 domain；
- data 仅使用统一的 `metadata.xlsx` / `data_dir`，不再引入 `source_dataset` / `target_dataset` 字段。

**2）Cross-system CDDG**

| Demo 文件路径                                     | 使用的 base 组件                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------------- |
| `configs/demo/02_cross_system/multi_system_cddg.yaml` | `environment/base.yaml` + `data/base_cross_system.yaml` + `model/backbone_dlinear.yaml` + `task/cddg.yaml` + `trainer/default_single_gpu.yaml` |

- `task.target_system_id` 用于选择目标系统 ID；
- 其他训练超参由 `task/cddg.yaml` 与 `trainer/default_single_gpu.yaml` 提供。

---

## 3. Config Registry（config_registry.csv）

- `configs/config_registry.csv` 提供了对 base 与 demo 配置的统一索引：

  - 每一行对应一个 base 或 demo；
  - 对于 demo 行，`base_environment/base_data/base_model/base_task/base_trainer` 列列出了它通过 `base_configs` 组合的各个组件；
  - `status` 列用于标记是否已经做过最小 sanity 验证：
    - `/`：尚未验证；
    - `sanity_ok`：已按 `docs/v0.1.0/codex/config_registry_plan.md` 中的 6 个实验计划完成至少一次本地跑通。

- 后续新增 demo 或调整 base 时，建议同步更新该 CSV，保持它作为 config 层的单一索引表。 

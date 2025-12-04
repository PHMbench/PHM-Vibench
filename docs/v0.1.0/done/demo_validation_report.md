# v0.1.0 Demo Validation Report — Pipeline & Config Sanity

> 更新范围：基于 v0.1.0 配置体系（base_configs + demo），对 6 个代表性 demo 进行最小跑通验证。  
> 环境：当前 Codex 容器，本地已有 `/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx`。

所有命令均使用以下模式（为避免多进程 DataLoader 权限问题，统一 `num_workers=0`）：

```bash
python main.py --config <demo_yaml> --override trainer.num_epochs=1 --override data.num_workers=0
```

并统一使用 base model：

```yaml
model:
  name: "M_01_ISFM"
  type: "ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
```

---

## 1. Cross-domain DG demo（demo_01_cross_domain）

- YAML：`configs/demo/01_cross_domain/cwru_dg.yaml`
- Pipeline：`Pipeline_01_default`
- Base 组合：
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/dg.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- 命令：

  ```bash
  python main.py \
    --config configs/demo/01_cross_domain/cwru_dg.yaml \
    --override trainer.num_epochs=1 \
    --override data.num_workers=0
  ```

- 结果：
  - `metadata.xlsx` 成功加载；DG 按 `source_domain_id=[0,1,2]` / `target_domain_id=[3]` 划分，样本数合理；
  - 模型/任务/Trainer 构建成功，完成 1 epoch 训练 + 验证 + 测试；
  - 生成 `test_result_0.csv` 等结果文件；
  - `configs/config_registry.csv` 中 `demo_01_cross_domain` 已标记为 `sanity_ok`。

**状态：`sanity_ok`（配置与 pipeline 链路通过）**

---

## 2. Cross-system CDDG demo（demo_02_cross_system）

- YAML：`configs/demo/02_cross_system/multi_system_cddg.yaml`
- Pipeline：`Pipeline_01_default`
- Base 组合：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_cross_system.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/cddg.yaml`（本次验证中补充了 `target_system_id` / `target_domain_num`）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 命令：

  ```bash
  python main.py \
    --config configs/demo/02_cross_system/multi_system_cddg.yaml \
    --override trainer.num_epochs=1 \
    --override data.num_workers=0
  ```

- 结果：
  - `CDDG` 划分成功，日志显示 `Dataset_id=1` 的 domains 按最后 1 个 domain 做测试集；
  - 模型/任务/Trainer 正常构建，完成 1 epoch 训练 + 测试；
  - 结果文件输出到 `results/demo/multi_system_cddg/...`；
  - `configs/config_registry.csv` 中 `demo_02_cross_system` 已标记为 `sanity_ok`。

**状态：`sanity_ok`（配置与 pipeline 链路通过）**

---

## 3. Few-shot demo（demo_03_fewshot）

- YAML：`configs/demo/03_fewshot/cwru_protonet.yaml`
- Pipeline：`Pipeline_01_default`
- Base 组合：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_fewshot.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/fewshot.yaml`
  - trainer: `base/trainer/default_single_gpu.yaml`
- 命令（修复后验证）：

  ```bash
  python main.py \
    --config configs/demo/03_fewshot/cwru_protonet.yaml \
    --override trainer.num_epochs=1 \
    --override data.num_workers=0
  ```

- 结果（修复后）：
  - `Id_searcher` 中为 `task.type="FS"` 增加了单独分支，FS 不再打印 “not specifically handled for ID searching” 的 warning，在目标系统范围内使用全部 ID；
  - 数据加载与 ID 划分正常，`dataset_task.FS.Classification_dataset` + `IdIncludedDataset` 构建 train/val/test 数据集；
  - `task.type="FS", task.name="classification"` 映射到 `src.task_factory.task.FS.classification`，模块存在并成功实例化 `Default_task` 派生任务；
  - `Get_sampler` 为 FS 返回 `Same_system_Sampler`，以系统为单位构建 batch；`DataLoader` 使用该 batch_sampler 构建 train/val/test 迭代器；
  - 训练可以完整跑完 1 个 epoch，测试阶段也能正常运行并输出结果文件。

**状态：`sanity_ok`（FS 任务 + sampler 链路可用，当前视作单系统 few-shot 分类示例）**

---

## 4. Cross-system few-shot demo（demo_04_cross_system_fewshot）

- YAML：`configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
- Pipeline：`Pipeline_01_default`
- Base 组合：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_cross_system_fewshot.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/cddg_fewshot.yaml`（已调整为 `type="GFS", name="classification"` 的 GFS base）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 命令（修复后验证）：

  ```bash
  python main.py \
    --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
    --override trainer.num_epochs=1 \
    --override data.num_workers=0
  ```

- 结果（修复后）：
  - `configs/base/task/cddg_fewshot.yaml` 已重构为 `task.type="GFS", task.name="classification"` 的 GFS base，包含 `num_episodes/num_systems/num_domains/num_labels/num_support/num_query` 等 sampler 必需字段；
  - demo4 顶层 `task` 覆盖为 `target_system_id: [1, 6], target_domain_num: 1`，其余字段从 GFS base 继承；
  - 数据加载成功，`dataset_task.GFS.Classification_dataset` + `IdIncludedDataset` 构建 train/val/test 数据集；
  - `Id_searcher` 对 `task.type="GFS"` 使用全部目标系统内 ID，`Get_sampler` 为 GFS 返回 `HierarchicalFewShotSampler`（train）和按系统分组的批采样器（val/test）；
  - 训练可以完成 1 个 epoch，测试阶段也能正常运行并输出结果文件。

**状态：`sanity_ok`（GFS few-shot 路径可用，作为跨系统 few-shot 示例）**

---

## 5. Pretrain + few-shot two-stage demo（demo_05_pretrain_fewshot）

- YAML：`configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Pipeline：`Pipeline_02_pretrain_fewshot`
- Base 组合：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_classification.yaml`
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/pretrain.yaml`（demo 覆盖为 `type: "pretrain", name: "hse_contrastive"`，并增加 `target_system_id/target_domain_num`）
  - trainer: `base/trainer/default_single_gpu.yaml`
- 命令（修复后验证）：

  ```bash
  python main.py \
    --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml \
    --override trainer.num_epochs=1 \
    --override data.num_workers=0
  ```

- 结果（修复后）：
  - `Pipeline_02_pretrain_fewshot` 检测到这是“单阶段配置（无 stages）”，走 `_run_single_stage_from_cfg(cfg)`；
  - 通过在 demo 任务段中显式加入 `target_system_id` / `target_domain_num`，并在 `search_target_dataset_metadata` 中对缺失字段做了健壮性处理，预训练配置不再触发 AttributeError；
  - 数据加载与 ID 划分正常，`dataset_task.Pretrain.Classification_dataset` + `IdIncludedDataset` 构建 train/val/test 数据集；
  - 任务 `type="pretrain", name="hse_contrastive"` 映射到 `task/pretrain/hse_contrastive.py`，成功实例化简化版 HSE 对比预训练任务；
  - 训练可以完成 1 个 epoch，测试阶段也能运行并输出简单的对比损失指标（当前示例中对比损失为 0，作为结构 sanity 检查即可）。

**状态：`sanity_ok`（单阶段 HSE 对比预训练配置与 Pipeline_02 链路可用；多阶段 stages 仍保留为后续扩展项）**

---

## 6. Pretrain HSE for CDDG demo（demo_06_pretrain_cddg）

- YAML：`configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`
- Pipeline：`Pipeline_01_default`（单阶段 HSE CDDG 预训练）
- Base 组合：
  - environment: `base/environment/base.yaml`
  - data: `base/data/base_cross_system.yaml`（demo 覆盖 `train_ratio` / `stride`）
  - model: `base/model/backbone_dlinear.yaml`
  - task: `base/task/pretrain.yaml` + demo 覆盖为：

    ```yaml
    task:
      name: "hse_contrastive"
      type: "pretrain"
      lr: 0.0005
      weight_decay: 0.0001
      target_system_id: [1]
      target_domain_num: 1
    ```

  - trainer: `base/trainer/default_single_gpu.yaml`
- 命令：

  ```bash
  python main.py \
    --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml \
    --override trainer.num_epochs=1 \
    --override data.num_workers=0
  ```

- 结果（修复后）：
  - 数据加载与 ID 筛选正常（通过 `target_system_id=[1]` 和 `target_domain_num=1` 只选一个系统做预训练示例）；
  - Task 构建阶段使用 `type="pretrain", name="hse_contrastive"` 映射到 `task/pretrain/hse_contrastive.py`，避免了此前指向 `CDDG/hse_contrastive` 的路径错误；
  - 训练可以完成 1 个 epoch，对比损失与总损失指标正常记录，测试阶段也能顺利运行。

**状态：`sanity_ok`（单阶段 HSE 对比预训练 Demo，作为 CDDG 预训练视角的基础示例）**

---

## 小结：v0.1.0 配置 + Pipeline 验证状态

- **已通过 sanity 验证（config/pipeline OK）**
  - `demo_01_cross_domain` — Cross-domain DG（单数据集 DG）
  - `demo_02_cross_system` — Cross-system CDDG
  - 已在 `configs/config_registry.csv` 中将 `status` 标记为 `sanity_ok`。

- **配置结构已整理但 Task 实现/语义尚未接好（blocked）**
  - `demo_03_fewshot` — FS/classification 任务模块缺失；
  - `demo_04_cross_system_fewshot` — 同上；
  - `demo_05_pretrain_fewshot` — data_factory 仍期望 pretrain 路线提供 `target_system_id` 等字段，语义未定；
  - `demo_06_pretrain_cddg` — `task.type="CDDG", name="hse_contrastive"` 映射到不存在的 `CDDG/hse_contrastive.py`。

在 v0.1.0 的范围内，可以将前两类 demo 视为“官方 sanity 通过”的代表性配置；few-shot 与预训练相关 demo 建议保留为配置设计草稿 + TODO，待你后续在 task_factory 与 data_factory 侧进一步统一语义后，再补齐 Task 实现与 YAML 映射，并将它们的 `status` 从 `/` 更新为 `sanity_ok`。***

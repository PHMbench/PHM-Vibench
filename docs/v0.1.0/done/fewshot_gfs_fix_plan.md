# v0.1.0 Few-shot / GFS 修复计划（demo3 & demo4）

本计划只描述 **配置 & pipeline 行为的修复方案**，暂不直接改代码，用来统一设计思路，确认后再实施。  
本版计划已经按照你的要求做了两点重要调整：

- `prototypical_network.py` / `matching_network.py` 被视为 **few-shot 模型族**，概念上属于模型层（`model.type/model.name`），  
  不再作为 v0.1.0 对外公开的 `task.name` 组合来使用；
- FS 被视为 **GFS 的简化版：没有跨数据集 / 多系统，只在单系统内做“few-shot 风格”采样**。

---

## 1. 现状回顾（基于当前代码）

### 1.1 demo3：Few-shot（单系统）

- 配置：`configs/demo/03_fewshot/cwru_protonet.yaml`，通过 `base/task/fewshot.yaml` 提供 `task`。
- 当前 `configs/base/task/fewshot.yaml`：
  - `task.type: "FS"`, `task.name: "classification"`；
  - few-shot 相关字段：`n_way / k_shot / q_query / episodes_per_epoch / lr`。
- 任务实现（代码状态）：
  - `src/task_factory/task/FS/classification.py`：继承 `Default_task`，代表“基于 few-shot 采样的普通分类任务”；
  - `src/task_factory/task/FS/FS.py`：一个带 `support/query` 结构的 few-shot 风格任务；
  - `src/task_factory/task/FS/prototypical_network.py` / `matching_network.py` / `knn_feature.py` / `finetuning.py`：
    - 代码里实现了 ProtoNet / MatchingNet 等多种 few-shot 算法；
    - 但从架构设计角度，更适合未来作为 **模型层的 few-shot 模型** 暴露（`model.type/model.name`），
      而不是在 v0.1.0 demo 中通过 `task.name` 使用。
- 数据 & sampler：
  - Dataset：`dataset_task.FS.Classification_dataset.set_dataset` + `IdIncludedDataset`，输出窗口级样本；
  - `dataset_task.FS.episode_dataset.set_dataset` 可以生成 `support_x/support_y/query_x/query_y` 风格的 episode，
    但目前 **没有接入主 data_factory 流程**；
  - `Get_sampler` 对 `task.type == "FS"` 直接返回 `None`，即使用 PyTorch 默认 sampler。
- 结果：demo3 当前更接近“普通分类 + FS 超参字段”的状态，few-shot 采样 / episode 结构在 v0.1.0 中还不是公开接口。

### 1.2 demo4：Cross-system few-shot（GFS 视角）

- 配置：`configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`；
- 当前 `configs/base/task/cddg_fewshot.yaml`：
  - `task.type: "FS"`, `task.name: "classification"`；
  - 暂无 GFS sampler 所需字段（如 `num_episodes / num_systems / num_domains / num_labels / num_support / num_query`）。
- 实现侧：
  - GFS 任务：
    - `src/task_factory/task/GFS/classification.py` / `matching.py`；
  - GFS 数据 & sampler：
    - `src/data_factory/dataset_task/GFS/Classification_dataset.py`；
    - `src/data_factory/samplers/FS_sampler.HierarchicalFewShotSampler`；
    - `Get_sampler` 对 `task.type == "GFS"` 调用 `_get_gfs_sampler(...)`。
- 参考配置：
  - `configs/reference/experiment_1_gfs_hse.yaml` 已给出 GFS 标准字段；
  - v0.1.0 中我们只拿它做“schema 参考”，不会在 CSV 中长期依赖。
- 结果：demo4 想表达的是“跨系统 few-shot / GFS baseline”，但当前 task 仍是 `FS` 类型，与 GFS sampler 不对齐。

### 1.3 ID 搜索行为

- `src.data_factory.ID.Id_searcher.search_ids_for_task`：
  - 对 `task.type == "DG"` / `"CDDG"` / `"GFS"` 有专门分支；
  - 对 `task.type == "FS"` 没有专门逻辑，落入默认分支并打印 warning：

    > Task type FS not specifically handled for ID searching...

---

## 2. demo3：单系统 Few-shot（FS）修复方案

目标：让 demo3 走一条**语义清晰的“单系统 few-shot 采样 + 分类”路径**，并与 v0.0.9 配置习惯保持一致：

- `task.type="FS"`，`task.name="classification"`；
- few-shot 的“算法类型”（ProtoNet / MatchingNet 等）从概念上归入模型层，  
  在 v0.1.0 demo 不通过 `task.name` 暴露；
- v0.1.0 只确保：
  - FS 任务可以正常构建并跑通；
  - FS 的 ID 搜索 / sampler 行为与“单系统 few-shot”语义不冲突；
  - 不强制把 `support/query` 结构暴露为公共 batch 接口。

### 2.1 配置层调整（不改代码）

1. **对齐 base fewshot 定义**

   - 文件：`configs/base/task/fewshot.yaml`
   - 计划（保守）：
     - 保持 `task.type: "FS"` 不变；
     - 保持 `task.name: "classification"`，语义上就是“FS 风格场景下的分类任务”；
     - few-shot 超参字段保持现状：
       - `n_way`, `k_shot`, `q_query`, `episodes_per_epoch`, `lr`；
     - 在文件注释中明确：
       - ProtoNet / MatchingNet 等 few-shot 算法属于**模型层设计**（未来可能对应 `model.type: "FewShot"` 等），
       - v0.1.0 demo 使用 ISFM，不在 `task.name` 中出现这些名字。

2. **在 demo3 中显式说明“FS + classification”语义**

   - 文件：`configs/demo/03_fewshot/cwru_protonet.yaml`
   - 计划：
     - 继续通过 `base_configs.task: "configs/base/task/fewshot.yaml"` 使用 few-shot base；
     - 顶层 `task` 段可显式写出（即使只是在 base 上重复一遍）：

       ```yaml
       task:
         type: "FS"
         name: "classification"
         # Few-shot 超参（与 base 保持一致或适度修改）
         n_way: 5
         k_shot: 5
         q_query: 15
         episodes_per_epoch: 100
         lr: 0.001
         # 可选：target_system_id: [1]  # 视需要沿用 CDDG/CWRU 的系统编号
       ```

   - 这样，config 接口与 v0.0.9 的 FewShot demo 一致：`type="FS"`, `name="classification"`，
     few-shot 算法不会通过 `task.name` 这一层暴露。

### 2.2 Data & sampler 设计（FS 视为“单系统版 GFS”）

> 本小节是设计意图说明，暂不在 v0.1.0 中大改代码，只做最小必要修改。

3. **FS 的数据形态：继续使用窗口级样本**

   - v0.1.0 中：
     - 继续使用 `dataset_task.FS.Classification_dataset.set_dataset` + `IdIncludedDataset`；
     - DataLoader 返回的 batch 形态仍为：
       - `{'x', 'y', 'file_id', ...}`（普通分类 batch）；
     - `FS.py` / `prototypical_network.py` / `episode_dataset.py` 等提供的 `support/query` episode 结构：
       - 视为**内部实验能力**；
       - 不在 v0.1.0 的 demo / 文档中作为对外承诺的接口。

4. **FS 的 sampler：概念上与 GFS 统一，但本轮只保证不会出错**

   - 现状：
     - `Get_sampler`：
       - `task.type == 'GFS'` → 使用 `HierarchicalFewShotSampler`（真正的 few-shot-style 采样）；
       - `task.type == 'FS'` → 返回 `None`，即 DataLoader 使用 PyTorch 默认 sampler；
   - 设计上的目标（记录为 TODO，不在本轮强行实现）：
     - 将 FS 看成“单系统的 GFS”：
       - 仍使用 `HierarchicalFewShotSampler` 框架；
       - 但限制在单一 `target_system_id` / 更少的 domain 上；
       - 输出依然是“窗口级 batch”，而不是强制 support/query 结构；
     - 未来可能的方向：
       - 在 `Get_sampler` 里为 `task.type == 'FS'` 挂接一个简化版的 `HierarchicalFewShotSampler`；
       - 通过 `task.num_systems/num_domains/num_labels/num_support/num_query` 控制采样；
       - 与 GFS 在 CSV / 文档层统一接口。
   - v0.1.0 先不动 sampler 主逻辑，只在 ID 搜索处为 FS 补上清晰的分支（见下一节），确保配置不会触发奇怪 warning。

### 2.3 ID 搜索行为修复（FS 分支）

5. **为 FS 单独定义 ID 搜索策略**

   - 计划修改位置：
     - `src.data_factory.ID.Id_searcher.search_ids_for_task`：
       - 在现有分支基础上新增：

         ```python
         elif args_task.type == 'FS':
             # Few-shot 使用筛选后的全部 ID（与 GFS 保持一致）
             train_val_ids, test_ids = list(metadata_accessor.keys()), list(metadata_accessor.keys())
         ```

   - 目标：
     - 消除 “Task type FS not specifically handled for ID searching” 的 warning；
     - 使 FS 与 GFS 一样，在目标系统范围内使用全部可用 ID；
     - 为未来“FS 使用与 GFS 统一的层次化采样器”打好数据筛选基础。

---

## 3. demo4：Cross-system few-shot（GFS）修复方案

目标：让 demo4 走 **GFS 任务 + GFS sampler** 的完整路径：

- 配置上使用 `task.type="GFS"`, `task.name="classification"`；
- 数据使用 `dataset_task.GFS.Classification_dataset`；
- sampler 使用 `HierarchicalFewShotSampler` 生成跨 system/domain 的 episodes；
- YAML 字段完全参考 `configs/reference/experiment_1_gfs_hse.yaml` 的 schema，不新造 key。

### 3.1 base task：从 FS 调整为 GFS

6. **重构 `configs/base/task/cddg_fewshot.yaml` 为 GFS base**

   - 文件：`configs/base/task/cddg_fewshot.yaml`
   - 计划调整：

     ```yaml
     task:
       name: "classification"
       type: "GFS"

       # CDDG/GFS 相关字段（参考 experiment_1_gfs_hse.yaml）
       target_system_id: [1, 13, 6, 12, 19]
       target_domain_num: 1

       optimizer: "adamw"
       lr: 0.001
       weight_decay: 0.0001
       early_stopping: true
       es_patience: 15

       loss: "CE"
       metrics: ["acc", "f1", "precision", "recall"]

       # GFS sampler 所需参数
       num_episodes: 100
       num_systems: 1
       num_domains: 1
       num_labels: 3
       num_support: 5
       num_query: 15
     ```

   - 所有字段均来自现有 reference 配置，不新增新 key。

### 3.2 demo4：覆盖与参数微调

7. **更新 demo4 的任务段以匹配 GFS**

   - 文件：`configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
   - 计划调整：
     - 继续通过 `base_configs.task: "configs/base/task/cddg_fewshot.yaml"` 引用 GFS base；
     - 顶层 `task` 段只做**最小覆盖**，例如：

       ```yaml
       task:
         target_system_id: [1, 13, 6, 12, 19]  # 或按你的实际需求调整
         target_domain_num: 1
         # 如需，可在 demo 里单独改 num_episodes / num_support / num_query 等
       ```

   - 这样，demo4 的 `task.type` / `task.name` 完全对齐：
     - `task.type: "GFS"`, `task.name: "classification"` → `task/GFS/classification.py`；
     - dataset → `dataset_task/GFS/Classification_dataset.py`；
     - sampler → `Get_sampler` 中的 `_get_gfs_sampler(...)` + `HierarchicalFewShotSampler`。

### 3.3 GFS sampler 激活验证（计划）

8. **最小运行命令（之后执行）**

   - 命令（单 epoch + 单进程）：

     ```bash
     python main.py \
       --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
       --override trainer.num_epochs=1 \
       --override data.num_workers=0
     ```

   - 验证要点（执行后会记录到 demo_validation_report 中）：
     - 构建数据时成功加载 `GFS` 对应 Dataset；
     - `Get_sampler` 为 train/val/test 返回 GFS sampler 或普通 batch sampler（按当前实现）；
     - 训练过程能完成 1 轮，不因 sampler/任务配置崩溃。

---

## 4. 多阶段 Pipeline（demo5/6）高层设计（只做规划）

> 下述只描述 future plan，v0.1.0 当前阶段的实际修改重点仍然是 demo3/4。  
> 多阶段 `stages` 设计会等你确认后，再在 YAML 中具体落地。

### 4.1 demo5：`pretrain_hse_then_fewshot.yaml` → 单 YAML 两阶段

9. **保持当前 base 组合 + 增加 `stages`**

   - 文件：`configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
   - 基本思路：
     - 顶层继续使用 `base_configs.environment/data/model/task/trainer`；
     - 顶层 `task` 可保持 `type: "pretrain"` 以兼容单阶段 fallback；
     - 新增 `stages` 字段，交给 `Pipeline_02_pretrain_fewshot` / `TwoStageOrchestrator` 解析：

       ```yaml
       stages:
         - name: "pretrain_hse"
           overrides:
             task:
               type: "pretrain"
               name: "hse_contrastive"
             trainer:
               num_epochs: 20

         - name: "fewshot_adapt"
           overrides:
             task:
               type: "FS"
               name: "classification"
               # few-shot 超参沿用 base/task/fewshot.yaml 或在此覆盖
       ```

   - 不新增新字段，仅用现有 `task.type/name` 和 trainer/data/environment 字段。

### 4.2 demo6：`pretrain_hse_cddg.yaml` 的阶段化预期

10. **长期方向（本轮不改 pipeline，仅做说明）**

   - 目标（参考 `experiment_2_cddg_hse_pretrain.yaml` 思路）：
     - Stage 1：`task.type="pretrain"`, `task.name="hse_contrastive"`，做统一 HSE 预训练；
     - Stage 2：`task.type="CDDG"`, `task.name` 为 CDDG 分类相关任务，加载 stage1 的 checkpoint 进行适配。
   - YAML 结构可与 demo5 的 `stages` 类似，只是 task/type/value 不同。
   - 当前 v0.1.0 重点：先保证 `pretrain_hse_cddg.yaml` 在 `Pipeline_01_default` 下单阶段能稳定跑通，多阶段重构留到后续版本。

---

## 5. 后续执行顺序（待你确认后进行）

1. **确认本 plan（核心设计点）**

   - 对 demo3：
     - 继续使用 `task.type="FS"`, `task.name="classification"`；
     - few-shot 算法（ProtoNet / MatchingNet 等）视为**模型层概念**，不在 v0.1.0 demo 的 `task.name` 中出现；
     - sampler 先按“普通分类 + FS 超参”跑通，未来再看是否需要接入层次化 few-shot sampler。
   - 对 demo4：
     - 切换为 `task.type="GFS"`, `task.name="classification"`；
     - 使用 GFS 对应的 Dataset + HierarchicalFewShotSampler，体现“跨系统 few-shot”的采样逻辑。

2. **按本 plan 实施配置 & 最小代码修改**

   - 修改 YAML：
     - `configs/base/task/cddg_fewshot.yaml` → 明确为 GFS base；
     - `configs/demo/03_fewshot/cwru_protonet.yaml` → 显式 FS + classification few-shot 配置（但不引入新 key）；
     - `configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml` → 切换到 GFS 任务，并按需微调超参；
     - `configs/base/task/fewshot.yaml` → 在注释中明确“FS + classification + 统一 ISFM 模型”的定位。
   - 最小代码修改（不引入新字段，只修复行为）：
     - `src/data_factory/ID/Id_searcher.py`：
       - 为 `task.type == 'FS'` 增加明确分支（使用全部 key），消除 warning，并统一 FS/GFS ID 搜索逻辑。

3. **运行最小验证并更新验证记录**

   - 按计划跑：

     ```bash
     # demo3 FS
     python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml \
       --override trainer.num_epochs=1 --override data.num_workers=0

     # demo4 GFS
     python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml \
       --override trainer.num_epochs=1 --override data.num_workers=0
     ```

   - 若两者均能完成 1 轮 train/val/test：
     - 在 `configs/config_registry.csv` 中将 `demo_03_fewshot` 和 `demo_04_cross_system_fewshot` 的 `status` 标为 `sanity_ok`；
     - 在 `docs/v0.1.0/codex/demo_validation_report.md` 中补充对应的验证结果条目；
     - 在 `docs/v0.1.0/cc/pipeline_fix_and_validation_plan.md` 中把 demo3/4 状态从 TODO 改为已完成（带上命令记录）。


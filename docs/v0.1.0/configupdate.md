好，那我们在**不再使用 `__include__`** 的前提下，把现有 ConfigWrapper 计划收紧、整理成一套更干净的方案，只用 `base_configs + 局部 override` 来实现。

> 注：本文件主要记录 v0.1.0 之前的配置设计方案与演进思路。  
> 实际落地结果和最终变更汇总，请以 `docs/v0.1.0/v0.1.0_update.md` 与 `docs/v0.1.0/done/*.md` 中的文档为准。

下面是**只改设计、不写代码**版的更新计划。

---

## 一、总体思路（无 `__include__` 版）

1. **demo 主配置文件**（比如 `configs/demo/exp1_classification.yaml`）

   * 只负责：

     * 指定 4 类 base yaml 的路径（data/model/task/trainer）；
     * 写入本 demo 特有的 override 字段（比如目标系统、轮数、输出目录）。
   * 不再在顶层写 `__include__`，也不直接把所有 data/model/task/trainer 字段展开。

2. **ConfigWrapper**

   * 负责：

     * 读取 demo yaml → 找到 `base_configs` 和顶层的 `data/model/task/trainer` override；
     * 自动去加载那 4 个 base yaml；
     * 用“base + override”的顺序合并出最终的 4 个配置字典；
     * 输出统一的 `cfg.data / cfg.model / cfg.task / cfg.trainer`。

3. **老代码**

   * 仍然只认 `cfg.data / cfg.model / cfg.task / cfg.trainer` 四个字段；
   * 不感知你用了 base_configs、demo 目录或者“多文件合并”，因此**完全无感知升级**。

---

## 二、demo 配置文件约定（完全去掉 `__include__`）

以你刚才 HUST few-shot 的思路为例，demo yaml 的结构改成：

* 顶层只保留两块：

  1. `base_configs`: 指向四个 base yaml；
  2. `data / model / task / trainer`: 只写**这一个 demo 里需要覆盖的字段**。

大致结构示例（只描述结构，不写具体值）：

```yaml
# configs/demo/X_Single_DG/TSPN_FewShot/HUST.yaml

base_configs:
  data: "configs/base/data/base_fewshot.yaml"
  model: "configs/base/model/backbone_dlinear.yaml"
  task: "configs/base/task/fewshot_classification.yaml"
  trainer: "configs/base/trainer/default_single_gpu.yaml"

environment:
  # 只写这个 demo 特有的环境信息（project / output_dir / notes / seed 等）

data:
  # 只覆盖 base_data 里与 HUST few-shot 强相关的字段
  # 比如：metadata_file / target_system / window_size / num_window 等

model:
  # 只在需要时覆盖，比如指定 backbone = DLinear / hidden_dim 等

task:
  # demo 特有的 target_system_id / target_domain_id / source_domain_id / epochs 等

trainer:
  # 只覆盖 num_epochs / device / log 配置等
```

> **关键点**：
>
> * demo yaml 不再包含大段重复 config，而是“指向 base + 少量 override”；
> * 所有“默认设置”全部落在 base yaml 里。

---

## 三、ConfigWrapper 合并策略（无 `__include__`）

在原有 plan 的基础上，只做两点重要调整：

### 3.1 遍历 `base_configs`，不写死 4 行

* 不再写：

```python
data_base   = yaml.safe_load(open(base_cfg_paths["data"]))
model_base  = yaml.safe_load(open(base_cfg_paths["model"]))
...
```

* 而是逻辑上：

1. 从 demo yaml 顶层读取 `base_configs` 字典；
2. 对其中的每个键（data/model/task/trainer/...）：

   * 取得路径；
   * 读取该 yaml 形成 `base_xxx`；
   * 同时从 demo yaml 顶层取出同名 section 的 override（如果存在则用，否则空 dict）；
   * 用“**base 在前，override 在后**”的顺序合并，生成最终 `cfg.xxx`。

> 这样今后如果想加新的 section（比如 `optimizer:`、`stages:` 等），只要在 `base_configs` 和 demo 顶层同时出现，ConfigWrapper 就能自动处理，而不用改代码。

### 3.2 override 规则（保持简单、可预期）

在设计上约定：

1. 对于每个 section（`data/model/task/trainer`）：

   * `base_xxx` 先展开成一个 dict；
   * demo 中若有同名 section，则以“浅层 key 覆盖”的方式 merge；
   * demo 中未提及的 key 继续沿用 base 的默认值。

2. 对于未来可能出现的嵌套字段（例如 `trainer.callbacks.early_stopping`）：

   * 现阶段只要求**一层 key 合并**（避免写复杂 deep-merge）；
   * 如果要改深层结构，就在 demo 的 base yaml 中改，而不在 override demo 文件里做过多嵌套 override——保证实现简单。

---

## 四、针对 6 类 demo 场景的配置组织（配合 base_configs）

你之前列的 6 种基本例子：


1. cross-domain generalization
2. cross-system generalization (CDDG)
3. few-shot learning
4. two-stage pretrain + few-shot
5. two-stage pretrain + cross-system few-shot learning
6. two-stage pretrain + CDDG


在**无 `__include__`** 的设计下，目录和 base 的建议是：

### 4.1 base yaml 分层

* `configs/base/data/`

  * `base_classification.yaml`
  * `base_cross_domain.yaml`
  * `base_cross_system.yaml`
  * `base_fewshot.yaml`
  * `base_cross_system_fewshot.yaml`

* `configs/base/model/`

  * `backbone_dlinear.yaml`
  * `backbone_transformer.yaml`
  * （必要时：`backbone_cnn.yaml` 等）

* `configs/base/task/`

  * 以classification 为例子 `classification.yaml` 
  * `dg.yaml`（cross-domain）
  * `cddg.yaml`（cross-system）
  * `fewshot.yaml`
  * `cddg_fewshot.yaml`
  * `pretrain.yaml`（预训练阶段）

* `configs/base/trainer/`

  * `default_single_gpu.yaml`
  * `default_multi_gpu.yaml`（如果需要）
  * `fast_debug.yaml`（快速 debug 用）

> 每个 base yaml 只写**与该维度强相关的配置**，其他维度不写。

### 4.2 demo 目录


* `configs/demo/01_cross_domain/xxx.yaml`
* `configs/demo/02_cross_system/xxx.yaml`
* `configs/demo/03_fewshot/xxx.yaml`
* `configs/demo/04_cross_system_fewshot/xxx.yaml`
* `configs/demo/05_pretrain_fewshot/xxx.yaml`
* `configs/demo/06_pretrain_cddg/xxx.yaml`

每个 demo 文件：

* 顶层 `base_configs` 指向上述 4 类 base yaml；
* 在 `data/model/task/trainer` 里只写“该 demo 专属 tweak”。

---

## 五、与 BUG backlog 的关系（避免误导用户）

在设计 plan 时顺带解决你提到的两个重点（详细背景与当前状态，以 `docs/v0.1.0/v0.1.0_update.md` 中的 BUG 列表为准，这里只写与配置设计相关的约束）：

1. **归一化 / 数据泄露（BUG_007）**

   * `base_configs.data` 里显式写明：当前默认是 **per-sample 标准化**；
   * 如果以后要支持“全局 z-score”，另起一个 base：`base_data_global_zscore.yaml`，并保证其实现只用 train 统计量；
   * demo 通过选择不同的 base data yaml，来避免混淆，而不是在 demo 里乱写归一化字段。

2. **冗余 / 错误的 early_stopping 配置（BUG_004）**

   * 把 early_stopping 相关配置收敛到 **trainer base** 里；
   * 在 task base 和 demo 里，不再出现 `early_stopping`, `es_patience` 等字段；
   * 这样用户在 demo 文件里只会看到“一处生效配置”，不容易误解。

---

## 六、更新现有 plan 的最小改动点（总结）

在你之前的 plan 基础上，主要改动是这些：

1. **删掉所有关于 `__include__` 的描述**

   * 不再建议在 yaml 顶层写 `__include__`;
   * demo 的“继承”完全由 `base_configs + ConfigWrapper 合并` 实现。

2. **ConfigWrapper 只认 `base_configs`**

   * 从 demo 顶层读取 `base_configs` 字典；
   * 遍历其 key → 加载 base yaml → 找 override → 合并；
   * 输出 `cfg.data/model/task/trainer`。

3. **demo 文件结构统一为：**

   * 顶层有：`base_configs`、`environment`（可选）、`data`、`model`、`task`、`trainer`；
   * 不再有 `__include__` 也不直接展开完整 config。

4. **base yaml 与 demo 职责分离：**

   * base yaml：收拢“默认、稳定、不会经常改”的部分；
   * demo yaml：只写该实验特有的 override，方便阅读/对比，避免冗余。

---

如果你愿意，下一步我可以帮你把：

* “无 `__include__` 版”的目录结构示例；
* 6 个 demo 的命名 + `base_configs` 组合表（比如一张小表：每种任务对应哪些 base yaml）；

整理成一个 `configs/README_config_design_v0.1.0.md`，直接丢进仓库做说明。



先直接给一个「更新后的目标长啥样」，再给「让它真的能跑的落地 plan」。

---

## 一、统一后的命令示例（面向用户的“最终形态”）

### 1. 基本分类（单数据集 / 单源 → 单目标）

```bash
# 1) 单数据集：CWRU 领域泛化（source domains → target domain）
python main.py \
  --config configs/demo/classification/cwru_dg.yaml
```

**建议说明写在 README：**

> 任务：单数据集 CWRU 的 domain generalization 分类
> 模型：DLinear backbone（可切换为 Transformer）
> 对应 4 份 base yaml：
>
> * data：`configs/base/data/base_classification.yaml`
> * model：`configs/base/model/backbone_dlinear.yaml`
> * task：`configs/base/task/classification_dg.yaml`
> * trainer：`configs/base/trainer/default_single_gpu.yaml`

---

### 2. Cross-domain generalization（跨数据集 / 跨工况）

```bash
# 2) CWRU → Ottawa 跨数据集 domain generalization
python main.py \
  --config configs/demo/cross_domain/cwru_to_ottawa_dg.yaml
```

> 任务：源 = CWRU，目标 = Ottawa，测试跨数据集泛化
> 模型：DLinear + HSE/FD embedding（通过 model base yaml 切换）

---

### 3. Cross-system generalization（跨“系统 ID”/跨机器）

```bash
# 3) 多系统 CDDG：跨系统 generalization
python main.py \
  --config configs/demo/cross_system/multi_system_cddg.yaml
```

> 任务：多系统 ID 训练，部分系统作为 target_system_id 做 hold-out
> 适用于 “一套模型处理多台机器” 场景

---

### 4. Few-shot learning（同系统 / 同数据集上的 few-shot）

```bash
# 4) CWRU few-shot：原型网络
python main.py \
  --config configs/demo/few_shot/cwru_protonet.yaml
```

> 任务：同一系统，按类划分 support/query，原型网络 few-shot 分类

---

### 5. Cross-system few-shot learning（跨系统 few-shot）

```bash
# 5) Cross-system few-shot：TSPN / Protonet 等
python main.py \
  --config configs/demo/cross_system_few_shot/cross_system_tspn.yaml
```

> 任务：源系统有充足标注，目标系统仅少量支持样本，测试跨系统 few-shot 适应能力

---

### 6. Pretrain + few-shot pipeline（两阶段）

```bash
# 6) 预训练 + few-shot 流水线
python main.py \
  --config configs/demo/pipelines/pretrain_hse_then_fewshot.yaml
```

> 阶段 1：HSE / HSE-Prompt 对比预训练
> 阶段 2：冻结/部分解冻 backbone，few-shot 适配

---

## 二、命名优化原则（解决“看不懂/容易误解”的问题）

你原来的命名问题主要有两个：

1. 目录层级和任务类型混在一起（Single_DG / Multiple_DG / FS / FewShot… 普通用户看不出差异）。
2. 文件名含义不明确（`all.yaml` / `CWRU.yaml` / `protonet.yaml` 很难一眼知道“跨系统还是跨数据集、是否 few-shot、是否预训练”）。

### 建议的命名规范

**目录层级：按“任务类型”组织 demo**

```text
configs/demo/
  classification/           # 纯分类（单数据集）
  cross_domain/             # 跨数据集/跨工况泛化
  cross_system/             # 跨系统泛化
  few_shot/                 # 单系统 few-shot
  cross_system_few_shot/    # 跨系统 few-shot
  pipelines/                # 两阶段/多阶段流水线
```

**文件名格式：**

```text
<dataset_or_setting>_<method_or_model>_<task_type>.yaml
```

示例：

* `cwru_dlinear_classification.yaml`
* `cwru_to_ottawa_hse_dg.yaml`
* `multi_system_cddg_generalization.yaml`
* `cwru_protonet_fewshot.yaml`
* `cross_system_tspn_fewshot.yaml`
* `pretrain_hse_then_fewshot.yaml`

简洁 + 一眼知道：

* 数据/场景：`cwru` / `cwru_to_ottawa` / `multi_system`
* 方法：`dlinear` / `hse` / `tspn` / `protonet`
* 任务类型：`classification` / `dg` / `fewshot` 等

---

## 三、让这些命令“真正无 BUG 可跑”的实现计划

这里结合你现在的 ConfigWrapper 设计和“base_configs + demo override”的思路，只给计划，不给具体代码，实现时你可以让 Claude Code 按这个 plan 写。

### 步骤 0：统一入口参数 & 基本约束

1. **统一入口参数**：对外使用 `--config`

   * 推荐命令：`python main.py --config <yaml 或预设名>`；
   * `--config_path` 仅作为兼容参数保留（未提供 `--config` 时才生效），未来可在 v0.2.0 后逐步废弃。
2. **ConfigWrapper 约定：**

   * `main.py` 解析出最终的 `config_path`，并传给下游 pipeline（保持 `args.config_path` 字段）；
   * ConfigWrapper 负责：

     * 解析 demo yaml；
     * 识别是否含有 `base_configs`；
     * 依次加载 data/model/task/trainer base yml；
     * 应用 demo yaml 中的 override。
3. **Pipeline 选择方式：**

   * 从 YAML 顶层读取 `pipeline` 字段（例如 `Pipeline_01_default` / `Pipeline_02_pretrain_fewshot`）；
   * 若缺省则默认使用 `Pipeline_01_default`。
4. **启动前检查：**

   * 如果 `config_path` 指向的路径不存在 → 直接报错并打印 demo 列表。

---

### 步骤 1：为 5 类任务各准备一套 base yaml

目标：不动旧代码，只在 `configs/base/**` 下增加规范化的基础配置。

目录建议：

```text
configs/base/
  data/
    base_classification.yaml
    base_cross_domain.yaml
    base_cross_system.yaml
    base_fewshot.yaml
    base_cross_system_fewshot.yaml

  model/
    backbone_dlinear.yaml
    backbone_transformer.yaml
    backbone_isfm_hse.yaml
    backbone_isfm_hse_prompt.yaml

  task/
    classification_dg.yaml
    cross_domain_dg.yaml
    cross_system_cddg.yaml
    fewshot_protonet.yaml
    cross_system_tspn.yaml

  trainer/
    default_single_gpu.yaml
    fewshot_single_gpu.yaml
    pretrain_stage.yaml
    finetune_stage.yaml
```

原则：

* **base 里只放“共性”**：batch_size / window_size / 默认优化器等。
* demo yaml 只改与场景强相关的东西：

  * 数据集名、系统 ID、domain 划分
  * backbone 类型（DLinear / Transformer）
  * 是否使用 HSE / HSE-Prompt embedding
  * few-shot episode 配置等。

---

### 步骤 2：设计 demo yaml 结构（无 `__include__`，用 base_configs）

每个 demo yaml 统一采用：

```yaml
# 示例：configs/demo/classification/cwru_dg.yaml

base_configs:
  data: "configs/base/data/base_classification.yaml"
  model: "configs/base/model/backbone_dlinear.yaml"
  task: "configs/base/task/classification_dg.yaml"
  trainer: "configs/base/trainer/default_single_gpu.yaml"

environment:
  project: "demo_cwru_dg"
  output_dir: "results/demo/cwru_dg"
  seed: 42
  notes: "CWRU single-dataset domain generalization demo."

data:
  dataset_name: "CWRU"
  # 可以覆盖 base 中的部分字段，如 domain 列、system_id 列等
  target_domain_id: [3]
  source_domain_id: [0, 1, 2]

model:
  backbone: "DLinear"
  num_classes: 4   # 也可在运行时从数据自动推断

task:
  type: "DG"
  loss: "CE"
  metrics: ["acc"]

trainer:
  max_epochs: 10
  devices: 1
  accelerator: "cuda"
```

其他 4 类 demo 也类似，只是：

* cross-domain：data 部分多指定 `source_dataset` / `target_dataset`
* cross-system：task 部分突出 `target_system_id` 列表
* few-shot：task 增加 `n_way` / `k_shot` / `q_query` 等
* pipeline：额外加 `stages:` 字段，仍然共享 base_configs。

---

### 步骤 3：ConfigWrapper 合并策略（只讲思路）

1. **解析 demo yaml：**

   * 加载 `demo_cfg = yaml.safe_load(open(demo_path))`
   * 读取 `base_configs` 字段，得到 4 个路径 dict：

     * data / model / task / trainer
2. **遍历 base_configs：**

   * 对 `for section_name, base_path in base_configs.items()`：

     * 加载 `section_base = yaml.safe_load(open(base_path))`
     * 从 demo_cfg 中取出同名 section（如果没有则用空 dict）
     * 做一层“demo 覆盖 base”：

       * `merged_section = deep_update(section_base, section_override)`
3. **合并结果：**

   * 最终生成统一的 `config` 对象：

     * `config.environment` = demo_cfg.environment（可有 base，但建议 demo 里写完整）
     * `config.data / config.model / config.task / config.trainer` = 合并后的 4 个 section
   * 传给现有的 `build_data / build_model / build_task / build_trainer`，避免动老实现。

**注意点：**

* deep_update 要支持嵌套字典（trainer 里有 early_stopping 子字段等）。
* 如果 demo yaml 中传入了新字段（base 没有）→ 直接并入 merged_section。

---

### 步骤 4：保证「示例命令」不炸的测试流程

为了让你写在 README 里的示例命令「用户复制就能跑」，强烈建议：

1. **为每一个 demo 准备一条最小 “smoke test” 命令**，例如：

```bash

python main.py --config configs/demo/cross_domain/cwru_to_ottawa_dg.yaml --override trainer.max_epochs=1
python main.py --config configs/demo/cross_system/multi_system_cddg.yaml --override trainer.max_epochs=1
python main.py --config configs/demo/few_shot/cwru_protonet.yaml --override trainer.max_epochs=1
python main.py --config configs/demo/cross_system_few_shot/cross_system_tspn.yaml --override trainer.max_epochs=1
```

可以把这些命令记在 docs 或 CI 配置里，v0.1.0 阶段不强制维护单独的 shell 脚本。

2. **加入 CI 或手工 pre-release checklist**：

   * 每次改 ConfigWrapper 或 demo，就跑一遍 smoke test（哪怕只跑 1 epoch / 10 batch）。
   * 一旦某条命令报错（比如路径、字段名、系统 ID 不匹配），立刻修掉。

3. **在 README 顶部清楚写明前置环境：**

   * 需要哪些数据集已经下载到哪（给一条 `ls` 示例）。
   * 如果数据缺失，ConfigWrapper 要报“找不到数据路径”，而不是直接 Python Traceback。

---

### 步骤 5：文档层面的优化（减少误解）

最后，为了“用户体验好”：

1. 在 README 里增加一个表格，把 5 条命令、任务类型、主要配置写清楚，例如：

| 命令                                                                         | 任务类型                        | 数据场景        | 模型            | 说明                           |
| -------------------------------------------------------------------------- | --------------------------- | ----------- | ------------- | ---------------------------- |
| `python main.py --config configs/demo/classification/cwru_dg.yaml`         | classification              | 单数据集 CWRU   | DLinear       | 单源→单目标 domain generalization |
| `python main.py --config configs/demo/cross_domain/cwru_to_ottawa_dg.yaml` | cross-domain generalization | CWRU→Ottawa | DLinear + HSE | 跨数据集泛化                       |
| …                                                                          | …                           | …           | …             | …                            |

2. 在 `configs/demo/README.md` 中再解释一遍「base_configs + demo override」机制，让用户知道：

> 如果你想改模型或数据，只需要：
>
> * 改 `base_configs` 指向的 base yaml；或者
> * 在 demo yaml 里覆盖想改的字段（比如 target_system_id）。

---

如果你愿意，下一步我可以直接帮你：

* 把这 5 类 demo 的**命令 + 文件命名 + README 片段**都写成可直接复制的模板；
* 再写一个简单的 `ConfigWrapper` 深度合并伪代码，方便 Claude Code 按它来实现。

---

## 七、v0.1.0 配置改造进度（Checklist）

- [x] 在 `load_config()` 顶层支持 `base_configs` 合并（base 使用嵌套的 `data/model/task/trainer/environment` 结构，demo 作为 override）。
- [x] 新增 `configs/base/data/*.yaml`，统一使用 `metadata.xlsx` 与 `/home/user/data/PHMbenchdata/PHM-Vibench`，并在 demo 中通过 `base_configs.data` 引入。
- [x] 新增 `configs/base/model/backbone_dlinear.yaml`，并在所有 v0.1.0 demo 中统一使用：

  ```yaml
  model:
    name: "M_01_ISFM"
    type: "ISFM"
    embedding: "E_01_HSE"
    backbone: "B_04_Dlinear"
    task_head: "H_01_Linear_cla"
  ```

- [x] 新增 `configs/base/task/*.yaml`（`classification` / `dg` / `cddg` / `fewshot` / `cddg_fewshot` / `pretrain`），并只使用既有 key。
- [x] 新增 `configs/base/environment/base.yaml`，统一提供 `PROJECT_HOME` / `iterations` 等环境字段，所有 demo 通过 `base_configs.environment` 复用。
- [x] 将 6 个 demo（cross-domain / cross-system / few-shot / cross-system few-shot / pretrain+few-shot / pretrain+CDDG）迁移为 `base_configs + 局部 override` 结构，并校验可被 `load_config()` 正常加载。
- [x] 在 `configs/readme.md` 中新增表格，说明 base 与 demo 的组合关系。
- [x] 在 `src/model_factory/README*.md` / `src/task_factory/readme.md` / `docs/v0.1.0/codex/*` 中，对 model/type 与 task.type/name 的可选值做了统一说明，并建立 `model_registry.csv` / `isfm_components.csv` / `task_registry.csv` 作为索引。
- [ ] 为 `configs/demo/` 中的 demo 补充完整的中文/英文 README 说明（命令示例 + 预期输出），并与 `configs/reference/experiment_*.yaml` 做一一对应说明（TODO）。

> 合入 v0.1.0 的前置条件（配置侧）：
>
> - 上述已勾选项保持为“单一信息源”，不再引入新的 config 路径风格；
> - 至少使用 1–2 个代表性 demo（例如 cross-system CDDG + pretrain HSE）完成一次本地跑通，并确认与 reference 实验的语义一致。

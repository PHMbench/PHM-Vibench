## Task Factory 配置映射整理计划（草稿）

版本：v0.1.0 / codex 记录  
目标：整理一份清晰文档，让用户**只看一份说明或一张表**，就能从 config 正确写到对应的 task，避免在 `main-config-pipeline` 与 `task_factory` 之间反复翻源码。

---

### 1. 梳理现状：列出所有 Task 与入口字段

- 浏览范围（至少）：
  - `src/task_factory/task_factory.py`
  - `src/task_factory/Default_task.py`
  - `src/task_factory/task/ID/ID_task.py`
  - `src/task_factory/task/MT/multi_task_lightning.py`
  - `src/task_factory/task/`（包含 `DG/`, `CDDG/`, `pretrain/`, `FS/`, `GFS/`, `ID/`, `MT/` 等子目录）
  - `src/task_factory/utils/`
- 输出一个内部草表csv，字段建议：
  - `config 关键字段`（例如 `task.name` / `task.type` 等，需按实际代码确认）
  - `task.type`（任务大类：DG / CDDG / pretrain / FS / GFS / ID / MT / Default_task 等）
  - `task.name`（具体任务名：classification / hse_contrastive / masked_reconstruction / FS / matching 等）
  - `Python 模块路径`（如 `src.task_factory.task.CDDG.classification`）
  - `主要适用场景`（type 侧：单域DG / 多域DG / ID / Multi-task / Few-shot / pretrain 等；name 侧：分类 / 预测 / 异常检测）
  - `典型使用的 pipeline`（例如 `Pipeline_01_default` 等，如有）

> 目标：搞清楚“**config 的哪几个字段最终决定选到哪个 Task**”，并把所有可选 Task 都枚举出来。

---

### 2. 设计统一说明模板：所有 Task 按相同结构描述

为每一种可被 factory 构建的 Task 定义统一信息项（后续写到文档中）：

- 标识信息：
  - `task.type`（任务大类，如 DG / CDDG / pretrain / FS / GFS / ID / MT / Default_task）
  - `task.name`（具体任务名，如 classification / hse_contrastive / masked_reconstruction 等）
  - 对应 Python 模块路径（例如 `src.task_factory.task.DG.classification`）
- 功能与场景：
  - 任务类型（分类 / 预测 / DG / ID / Multi-task / Few-shot 等）
  - 典型使用数据集 / pipeline（如果有）
- 配置依赖：
  - 对 `data_factory` / `model_factory` / `trainer` 的前置要求
  - 必需的 config 字段（缺失会直接报错的）
- 常见错误：
  - 常见配置错误模式 + 报错信息简要说明

> 目标：以后新增 Task 按同一个模板补充描述，避免“隐式知识”只在代码里。

---

### 3. 重构 `src/task_factory/readme.md`（面向使用者）

规划将 `readme.md` 重构为 4 个核心部分：

1. **Task 是怎么被选中的**
   - 描述真实调用链（按代码确认准确命名）：
     - `main.py` → `load_config()` → `pipeline_xx` → `task_factory.build_task(config)`
   - 明确列出：哪些 config 字段参与选择 Task，它们的默认值和优先级。

2. **Task 索引表（核心总览）**
   - 表头示例：
     - `task.type | task.name | 场景 | 模块路径 | 典型 pipeline | 关键 config 示例`
   - 每一行对应一个可选 Task；用户可以通过表格快速从需求定位到该写什么 config。

3. **常用场景配方（recipes）**
   - 按场景给出几种“配方级”示例（不必是完整 YAML，只写与 Task 相关部分）：
     - 单域/单任务分类（例如 CWRU demo）
     - 多任务 / `multi_task_lightning`
     - ID 相关 Task
   - 每个配方都标明对应的 `task.type` 与 `task.name` 组合，并链接到索引表中的那一行。

4. **新增 Task 的规范**
   - 简要说明：
     - 在哪里注册新的 Task（registry 位置）
     - 建议的命名规则（`task.type` / `task.name` 与模块路径的关系）
     - 新增 Task 时必须在索引表补充一行，并按第 2 步模板写清信息

---

### 4. 将实现细节沉淀到 `src/task_factory/CLAUDE.md`

- `CLAUDE.md` 更偏架构 / 内部实现：
  - Task factory 的设计思路与 registry 结构
  - Task 与 `data_factory` / `model_factory` / `trainer` 的解耦与协作方式
  - 复杂 pipeline（如 `multi_task_lightning`）的流程图、调用顺序等
- `readme.md` 只保留“如何从 config 写到正确 Task”的使用层面说明，减少新用户的认知负担。

---



---

### 6. 选取代表性 config 示例并对齐说明

- 选 2–3 个重要配置文件（例如：`configs/demo/Single_DG/CWRU.yaml` 等）
- 对这些配置文件中与 Task 相关的字段进行标注，并在文档中给出“示意性拆解”：
  - 指出：哪些字段共同决定了 Task
  - 这些字段对应 Task 索引表中的哪一行
- 在 `readme.md` 中以“带注释的 config 片段”的形式展示，方便用户照抄改参数。

---

### 7. 后续落地顺序建议

1. 实际走完第 1 步（梳理 registry 和 config 字段），在此文件或单独表格中放出初版 Task 索引草稿。
2. 根据草稿微调第 2–4 步的结构（尤其是 `readme.md` 的目录和索引表表头）。
3. 最后再决定是否实现第 5 步的辅助脚本，以及在文档中如何引用。

> 本文件仅作为 codex 生成的规划草稿，你可以直接修改此文档，或将其中的部分内容复制到最终的 `src/task_factory/readme.md` / `CLAUDE.md` 中使用。

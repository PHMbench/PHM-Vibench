# Model Factory 配置文档整理方案（v0.1.0 草稿）

> 目标：让使用者一眼看懂 config 里的 `model` 段如何映射到具体模型类、需要哪些字段、有哪些可调参数；同时为新增模型提供统一规范。

## 1. 梳理现有注册与命名规则

- 阅读并总结：
  - `src/model_factory/model_factory.py`
  - `src/model_factory/CLAUDE.md`
- 列出所有可选的 **model 类型与子模块**，按 `model.type` 现有取值分类：
  - 例如：`ISFM`、`CNN`、`RNN`、`Transformer`、`ISFM_Prompt` 等（不单独把 FewShot 当成类型）。
  - 对每个标识符记录：
    - 对应 Python 类名
    - 模块路径
    - 适用任务类型（如 cla/pred/fs/gfs 等）一般都适用
- 同时扫描示例配置：
  - `configs/demo/`
  - `configs/experiments/`
  - 理清当前 `model` 段使用的字段（如 `model.type`、`model.embedding`、`model.backbone`、`model.task_head` 等）。

## 2. 定义统一的 model 配置 schema

- 不改变现有配置方式，只是**明确和文档化**现有字段，重点梳理：
  - `model.type`：模型类型，例如 `ISFM`、`CNN`、`RNN`、`Transformer` 等。
  - `model.embedding`：嵌入模块标识，例如 `E_01_HSE`（仅在部分类型下生效，如 `ISFM`）。
  - `model.backbone`：骨干网络标识，例如 `B_04_Dlinear`、`B_08_PatchTST`。
  - `model.task_head`：任务 head 标识，例如 `H_01_Linear_cla`、`H_03_Linear_pred`。
  - 通用超参数：如 `d_model`、`n_layers`、`dropout` 等。
- 明确两类字段：
  - **通用字段**：所有模型都支持或至少会忽略无害，例如 `dropout`。
  - **类型/模型专用字段**：仅在特定 `model.type` 或具体模型中有意义（例如 ISFM 专有的配置）。
- 在 schema 中明确：
  - 每个字段的类型、默认值、是否必填。
  - 字段命名风格与现有 YAML 保持一致（全部小写、使用下划线）。

## 3. 更新 `src/model_factory/README.md`：总览与映射表

- 新增「如何通过 config 选择模型」章节：
  - 说明 pipeline 如何读取 `config.model.*`。
  - 说明这些字段如何传入 `model_factory`（例如 `get_model(cfg.model)`）。
- 增加「模型注册表」，但**以 CSV 文件形式管理**，便于维护，例如：
  - 新建 `src/model_factory/model_registry.csv`，列出每一行一个可用模型（`model.type` + `model.name`）。
  - 建议 CSV 列包含（可根据实际情况调整）：
    - `model.type` (CNN/RNN/ISFM/Transformer 等)
    - `model.name` (具体模型类名)
    - `module_path`
    - `args`（该模型常用/关键配置字段的简要列举）
    - `notes`（简短备注 / 推荐用途）
    - `test_status`（记录测试结果，例如 `/` / `pass` / `fail` / `partial`）
  - README 中只需要说明这个 CSV 的用途和字段含义，不必维护 Markdown 表格。
  - 对于 ISFM 的 `embedding` / `backbone` / `task_head` 组件，单独在 `src/model_factory/ISFM/isfm_components.csv` 中维护组件 registry（见第 4 节）。

- 定义「标准 model 配置 schema」小节：
  - 列出通用字段及说明。
  - 指向各 `model.type` 对应目录下的详细参数说明文档。
- 提供 2–3 个完整 YAML 例子：
  - ISFM + 分类任务。
  - Transformer/Backbone + 预测任务。

## 4. 按 `model.type` 整理参数说明（含 ISFM 组件 CSV）

- 在每个类型对应的子目录（如 `src/model_factory/ISFM`、`CNN`、`Transformer` 等）新增或更新文档（例如 `README.md` 或 `CONFIG.md`）：
  - 简要介绍该类型的用途和典型任务。
  - 提供一个最小可运行 YAML 片段示例，仅包含与该类型相关的关键字段：

    ```yaml
    model:
      type: ISFM
      embedding: E_01_HSE
      backbone: B_04_Dlinear
      task_head: H_01_Linear_cla
      d_model: 256
      n_layers: 4
      dropout: 0.1
    ```

  - 给出「可配置参数表」，包括：
    - 字段名
    - 类型
    - 默认值
    - 含义
    - 是否必填
  - 特别是列出该类型下所有合法的候选项，例如：
    - `embedding` 可选值（如 `E_01_HSE` 等）。 也需要进一步给出E_01_HSE 需要的参数配置；**完整列表和路径可从 `src/model_factory/ISFM/isfm_components.csv` 中读取**；
    - `backbone` 可选值（如 `B_04_Dlinear` 等）。
    - `task_head` 可选值（如 `H_01_Linear_cla`、`H_03_Linear_pred` 等）。
- 对于差异较大的特殊模型（例如 Prompt 型 ISFM）：
  - 在对应类型文档中开单独小节，列出附加字段和专有配置。
  - 或在具体模型类的 docstring 中补充详细配置说明，并从类型文档中链接过去。

## 5. 统一整理 model_factory 文档结构（README / README_CN / CLAUDE）

- 目前存在的文档文件：
  - `src/model_factory/README.md`（英文，偏对外介绍与示例）
  - `src/model_factory/readme.md`（英文，偏实现/使用细节，内容与 README 有部分重叠）
  - `src/model_factory/CLAUDE.md`（给 agent 的内部工作说明）
- 统一规划：
  - 保留 `README.md` 作为**唯一的英文总览文档**：
    - 将 `readme.md` 中有价值的说明（模块结构、配置示例、workflow 等）合并进 `README.md`。
    - 按本 plan 中的结构整理章节：模型类型概览、配置方式（type/embedding/backbone/task_head）、CSV 注册表说明、各类型 README 链接等。
  - 新增 `README_CN.md` 作为**完整的中文对外文档**：
    - 内容与 `README.md` 对齐，但使用中文描述，适合中文用户直接阅读。
    - 可以优先覆盖：整体介绍、模型类型/组件说明、典型配置示例（尤其 ISFM + E_01_HSE + B_04_Dlinear + H_01_Linear_cla）、如何查 CSV 和各类型 README 获取详细参数。
  - 精简 `readme.md`：
    - 将其中内容全部迁移/合并到 `README.md` 与 `README_CN.md` 后，删除 `readme.md` 文件，避免双重入口。
  - 保留 `CLAUDE.md` 作为**面向智能代理/维护者的内部笔记**：
    - 在开头增加一句说明：「本文件主要给自动化 agent 使用，普通用户请阅读 README / README_CN」。
    - 可适当链接回 `README.md` 中的相关章节，而不用重复用户向文档内容。


## 6. 串联到主 pipeline 与任务文档

- 在顶层文档（如 `README.md` 或 `src/configs/CLAUDE.md`）中新增「模型选择与任务搭配」小节：
  - 说明不同任务类型推荐的 model 组合：

- 文本形式给出若干示例组合（具体表格可后续从 CSV 或 config 中导出），例如：
  - 单数据 DG 分类：`model.type = ISFM`，`embedding = E_01_HSE`，`backbone = B_04_Dlinear`，`task_head = H_01_Linear_cla`，示例配置见 `configs/demo/Single_DG/CWRU.yaml`。
  - 预测任务：`model.type = Transformer`，`backbone = B_08_PatchTST`，`task_head = H_03_Linear_pred`，示例配置见 `configs/experiments/...`。
  - FewShot / GFS 任务：强调这是**训练/任务范式**，通常复用上述模型类型（例如 ISFM/Transformer），在任务和 trainer 侧做 few-shot 配置，而不是新增一个 `model.type = FewShot`。
  - 预训练 + 微调：说明如何在多阶段 pipeline 中复用相同 `model.type`/`backbone`，参考 `src/configs/CLAUDE.md` 中的多阶段配置示例。

- 确保官方 demo 配置（例如 `configs/demo/Single_DG/CWRU.yaml`）：
  - 使用前面定义的标准 schema。
  - 能作为后续所有配置的参考模板。

## 7. 维护与贡献规范

- 更新 `src/model_factory/contributing.md`，约定新增模型的步骤：
  - 在 `model_factory` 中注册模型。
  - 在 central registry CSV（`docs/v0.1.0/codex/model_registry.csv`）中增加一行说明。
  - 在对应 `model.type` 目录文档中补充：
    - 最小 YAML 示例。
    - 参数表或指向模型 docstring 的链接。
- 对 PR 要求：
  - 新增/修改模型时，附带对应 config 示例（可以放在 `configs/experiments/` 或 demo 下）。
  - 简要说明该模型适用的任务类型以及推荐配置范围。

---

> 说明：本文件为 v0.1.0 的初稿规划，后续你可以直接在这里修改字段命名、表格结构和示例 YAML，以便逐步收敛为最终的官方文档和贡献规范。

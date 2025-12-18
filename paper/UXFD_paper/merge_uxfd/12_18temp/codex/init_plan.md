# UXFD 合并研究/重构计划（init_plan）

目标：把 UXFD 框架下的 **7 篇 Paper 子项目**（来自 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper`）以“**Paper 资产 submodule + 主仓库统一运行框架**”的方式并入 PHM‑Vibench，并确保：
- 7 篇 Paper 的**语义、配置、复现入口**不漂移
- PHM‑Vibench 的**单一入口**与 **5‑block 配置模型**不被破坏
- 研究闭环可跑：`train/eval/explain/collect/report`（agent 先只落盘 TODO 蒸馏，不接 LLM）

---

## 0) 真相源（SSOT）与硬约束

### 0.1 真相源（必须对齐的文档）
- 主仓库运行/架构约束：`README.md`、`AGENTS.md`、`CLAUDE.md`
- UXFD 7 篇 Paper 总览（上游）：`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/README.md`
- 上游的 paper 证据链与协议（可参考）：`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/doc/**`

### 0.2 硬约束（不满足就不合并）
- **单一入口不变**：`python main.py --config <yaml> [--override key=value ...]`
- **5‑block 配置不变**：`environment/data/model/task/trainer`
- **工厂与 registry 驱动**：新增能力要走 `src/*_factory/` 的注册/装配路径
- **Paper 工作流隔离**：paper‑grade 实验/脚本放 `paper/`（submodule），主仓库的验收/测试不依赖 submodule 初始化
- **不重造 task 体系**：优先复用现有 `task_factory`/`trainer_factory`；仅做“paper config → vibench config”映射与组件增补
- **metadata = data metadata**：解释模块读取的是 dataset/batch 的元信息（采样率/工况/domain/通道等），不是 run_meta
- **可复现三件套**：配置文件、运行命令、输出目录（日志/表格/图）要能闭环追溯

---

## 1) 7 篇 Paper 清单（目录路径作为唯一 ID）

上游 `Paper/README.md` 的“官方顺序”是合并时的唯一编号口径；在主仓库侧落为 `paper/UXFD_paper/<paper_id>/` 的 submodule。

| # | paper_id（建议） | 上游目录名 | 角色（并入主仓库后） |
|---:|---|---|---|
| 1 | `fusion_1d2d` | `1D-2D_fusion_explainable` | 模态融合/对齐相关 operator + 数据元信息需求定义 |
| 2 | `xfd_toolkit` | `Explainable_FD_Toolkit` | explain_factory 的 API/协议/可视化规范来源 |
| 3 | `llm_xfd_toolkit` | `LLM_Explainable_FD_Toolkit` | agent_factory 的“解释文本/对话”消费方（先不接 LLM） |
| 4 | `moe_xfd` | `MOE_explainable` | MoE 路由/专家结构相关 operator（路径级可解释） |
| 5 | `fuzzy_xfd` | `Paper_fuzzy_XFD` | 规则/隶属度/审计相关 explainer 或 baseline 模块 |
| 6 | `nesy_theory` | `Neuralsymbolic_theory` | 跨层抽象/命题/口径统一（主要沉淀为 docs） |
| 7 | `op_attention_tii` | `TII_operator_attention` | 算子级注意力相关 operator + 理论/合成验证（docs+可选 operator） |

> `paper_id` 只用于主仓库“映射与配置默认值”的稳定索引；论文内容真相源仍在 submodule 内的 README 与文档。

---

## 2) 目录落位（主仓库 vs Paper submodule）

### 2.1 Paper submodule（只放论文资产/实验资产）

目标：把 7 篇 paper 的 README/文稿/实验脚本/图表等“论文资产”保持原样引入，但不让主仓库依赖它们才能运行/测试。

建议结构（示意）：
```
paper/UXFD_paper/
  README.md                  # 总入口：7 篇索引 + 与主仓库映射说明
  README_SUBMODULE.md         # submodule 初始化/更新说明（对齐 paper/README_SUBMODULE.md 风格）
  fusion_1d2d/                # git submodule
  xfd_toolkit/                # git submodule
  llm_xfd_toolkit/            # git submodule
  moe_xfd/                    # git submodule
  fuzzy_xfd/                  # git submodule
  nesy_theory/                # git submodule
  op_attention_tii/           # git submodule
```

submodule 原则：
- submodule 内部 README/命令是“论文复现语义真相源”，主仓库只做“如何用 vibench 跑”的映射补充
- 不把 submodule 的依赖装进主仓库的 `requirements.txt`，避免污染主流程

### 2.2 主仓库（只放可复用的统一框架能力）

目标：把 7 篇 paper 共同需要的“运行框架能力”沉淀到主仓库，保证 config‑first 与可复用。

建议落位（与现有架构对齐）：
- 模型统一入口：`src/model_factory/X_model/`
  - 优先方案：在现有 `src/model_factory/X_model/TSPN.py` 基础上“增量扩展”
  - 兼容方案：新增一个不破坏旧行为的模型名（例如 `TSPN_UXFD`），并在 `src/model_factory/model_registry.csv` 注册
- explain 统一入口：新增 `src/explain_factory/`（主仓库可复用）
- agent 统一入口：新增 `src/agent_factory/`（先只落盘 TODO 蒸馏）
- paper→vibench 映射文档：`docs/paper_hub/<paper_id>.md`（主仓库侧“怎么跑”）

---

## 3) 统一框架设计：TSPN（算子可插拔）→ Explain → Collect/Report → Agent(TODO)

### 3.1 模型层：统一为“可插拔算子”的 TSPN

核心原则：不再“每篇一个模型文件”，而是 **一个 TSPN + 一个 OperatorRegistry**；7 篇 paper 的差异体现在：
- `operator_graph`（算子集合/拓扑）
- `operator_args`（每个算子参数）
- （可选）`hooks`（为 explain 暴露可解释中间节点）

建议把 Operator 抽象成最小接口（只约束张量契约 + 可选解释 hook）：
- `forward(x, meta) -> x'`
- `explain_hooks()`（可选）：暴露关键中间量名/shape/含义（供 explain_factory 统一消费）

### 3.2 配置层：paper config → vibench 5‑block config 的“稳定映射”

目标：7 篇 paper 的复现入口最终都能落到：
`python main.py --config <yaml> --override ...`

约定（建议）：
- **主仓库侧**只提供“UXFD 组件能力 + 映射适配器”，不强行把 paper configs 写进 `configs/config_registry.csv`
- **paper submodule 侧**提供 paper‑grade configs（引用主仓库的模型/解释能力）

建议新增一个 `paper_id` 映射表（实现层可以是 `config_adapter` 或纯 YAML 预设）：
- `paper_id -> default operator_graph + default operator_args + explain requirements`

### 3.3 Explain 层：必须读取 data metadata（不是 run_meta）

`src/explain_factory/` 的两个硬能力：
- `metadata_reader`：从 dataset/batch 提取统一 schema 的 data metadata
- `ExplainReady(m, x, μ) -> (bool, reasons[])`：解释可用性门控 + 可审计原因输出

推荐的 metadata schema（按可用性递增）：
- `sensor`: 通道名称/物理量/单位/安装位置
- `sampling_rate`, `window_length`, `stride`
- `operating_condition`: 转速/负载/工况标签
- `domain`: domain_id / domain_name（DG 用）
- `transform`: STFT/Envelope/Filter 等预处理说明（若在 data 或 task 中做过）

### 3.4 Collect/Report：结果闭环（不绑定外部服务）

目标：把 `train/eval/explain` 的输出整理成可被论文引用的“证据链产物”：
- `metrics.json/csv`（指标表）
- `figures/`（解释图、关键可视化）
- `config_snapshot.yaml`（运行时 resolved config）

主仓库只提供“收集脚本/规范”；paper submodule 可以提供“论文写作需要的汇总表/图”的高级脚本。

### 3.5 Agent：先只落盘 TODO 蒸馏内容（不接 LLM）

`src/agent_factory/` 在本阶段只做两件事：
- 规定“蒸馏内容”的落盘路径与 frontmatter schema
- （可选）把 `metrics + explain_summary` 拼成“蒸馏草稿”（仍不调用 LLM）

---

## 4) 合并路线（按 PR 闭环拆分）

每个 PR 都要满足：
- 不破坏主仓库 demos 与 tests
- 有明确验收命令（优先使用 `AGENTS.md` 里的命令）
- 文档口径可追溯（能回答“值从哪来/在哪消费”）

### PR0：UXFD Paper submodule 落位 + 文档入口对齐

改动（主仓库侧）：
- 建立 `paper/UXFD_paper/` 的入口结构（README 与 submodule 指南）
- 约定 7 个 submodule 的目录名与 `paper_id`
- 增加 `docs/paper_hub/README.md`（说明“paper 资产 vs 主仓库能力”的边界）

验收：
- `paper/UXFD_paper/README.md` 里能点击到 7 篇（哪怕 submodule 还未 init，也不应破坏主仓库）
- 主仓库 `README.md`/`paper/README_SUBMODULE.md` 的口径不冲突（必要时只补链接，不改主叙事）

### PR1：TSPN “算子可插拔”骨架（不搬 paper‑specific 代码）

改动：
- 在 `TSPN` 现有实现基础上抽出/新增 `OperatorRegistry + operator_graph` 的装配路径
- 为避免破坏旧行为：优先新增 `TSPN_UXFD`（或同等命名）并注册到 `src/model_factory/model_registry.csv`
- 最小可用 operator 集（先覆盖 1 篇 paper 的最稳定子集）

验收：
- `python -m scripts.config_inspect --config <uxfd_demo.yaml>` 能看到模型 target 指向正确模块
- 最小 smoke：能前向 + 能跑 1 个 batch（不要求指标）

### PR2：7 篇 paper 的 config 映射（paper_id → operator_graph）

改动：
- 增加 `config_adapter`（把上游 paper 风格配置键翻译成 vibench 5‑block + `operator_graph/args`）
- 产出 7 篇的“映射卡片”：`docs/paper_hub/<paper_id>.md`

验收：
- 每篇 `docs/paper_hub/<paper_id>.md` 都提供一条可复制命令：`python main.py --config ...`
- 7 篇至少能完成：构建模型 + 走到训练 step（不要求指标对齐）

### PR3：Explain_factory（metadata_reader + ExplainReady + 最小 explainer）

改动：
- `src/explain_factory/metadata_reader.py`：从 dataset/batch 抽取 metadata（统一 schema）
- `src/explain_factory/eligibility.py`：ExplainReady 门控（返回 reasons，日志可审计）
- `src/explain_factory/explainers/`：先落 1–2 个最小解释器（例如 operator‑level attribution + 时域/频域重要性）

验收：
- explain 开启：能生成 `artifacts/explain/*`
- explain 关闭：不影响主流程
- metadata 缺失：明确报缺哪些键（不 silent fail）

---

### PR4：Collect/Report 与“论文证据链”产物规范

改动：
- 若已有收集能力：补“UXFD 7 篇汇总视图”
- 若没有：新增 `scripts/collect_uxfd_runs.py`（只读 `save/`，生成 CSV/JSON 总表）
- 给 paper submodule 的“结果表模板/图表索引”提供稳定输入格式

验收：
- 汇总脚本能生成总表（CSV/JSON），并能被 paper submodule 的写作脚本消费

### PR5：agent_factory（仅落盘 TODO 蒸馏内容）

改动：
- 定义 `distilled/` 的路径规范与 frontmatter（paper_id / dataset_task / operator_graph / findings / issues / next）
- （可选）生成蒸馏草稿的轻量脚本

---

## 5) 三个关键“判定/接入点”（必须写死成规范）

### 5.1 “解释模块能不能用”的唯一入口

- `ExplainReady(...) -> (bool, reasons[])`
- reasons 必须可审计（日志/产物里能看到）

### 5.2 “data metadata 从哪里来”（优先级）

1. dataset 对象提供 `.metadata` / `.get_metadata()`
2. dataloader/batch 返回 `(x, y, meta)`
3. data_factory 返回值附带（例如 `DataBundle{..., metadata}`）

### 5.3 “paper 兼容”怎么保证不漂

- 每篇 paper 的 `docs/paper_hub/<paper_id>.md` 固化：`operator_graph + 关键超参 + explain metadata 要求`
- `config_adapter` 必须有 `paper_id -> 默认映射`，并带版本号/变更记录（避免复现漂移）

---

## 6) Prompt Pack（用于“让模型生成更好的重构/优化计划”）

把下面 prompt 复制给任意 LLM；把 `{...}` 用你的真实内容替换即可。

### Prompt A：生成“可执行 PR 计划”（推荐）
```
你是资深研究工程负责人，要把 7 篇 UXFD Paper 子项目并入 PHM‑Vibench 主仓库。

输入材料：
1) 主仓库约束：README.md / AGENTS.md / CLAUDE.md 的关键规则（粘贴要点）
2) 上游 Paper 总览：/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/README.md（粘贴 7 篇列表）
3) 当前代码现状：仓库树（src/*_factory/, configs/, paper/）（粘贴关键目录）
4) 你的硬约束：
   - 单一入口：python main.py --config ...
   - 5-block：environment/data/model/task/trainer
   - paper 内容必须以 submodule 形式引入到 paper/UXFD_paper/
   - 不让主仓库测试依赖 submodule

任务：
输出一个“闭环合并计划”，必须包含：
1) 目标与非目标（明确不做什么）
2) 7 篇 paper -> 主仓库模块映射表（operator/explain/agent/docs/only-doc）
3) 分 PR 列表（PR0..PRN），每个 PR：改动点、文件落位、验收命令、回滚策略
4) 配置迁移方案：paper config -> vibench config 的稳定映射（paper_id->默认值）
5) 风险清单（至少 8 条）+ 缓解措施
6) Definition of Done（工程 + 论文证据链）

输出格式：严格 Markdown；先表格后列表；每个 PR 必须给出 1 条可复制的验收命令。
```

### Prompt B：生成“paper config → vibench 5‑block 映射”
```
你是 PHM‑Vibench 的配置系统专家。

给定：某篇 paper 的 config 键（粘贴 YAML/argparse 参数），以及这篇 paper 的目标任务（DG/FS/ID/Pretrain）。
要求：把它映射到 vibench 的 5‑block：environment/data/model/task/trainer，并给出：
1) 对应的 registry 组件（model/task/trainer/data reader）
2) operator_graph/operator_args 的落点（若需要）
3) 需要的数据 metadata 键（解释依赖）
4) 最小可跑命令：python main.py --config ... --override ...
输出：一个可直接落地的 vibench YAML 模板 + 映射解释表（字段->来源->消费方）。
```

### Prompt C：生成“ExplainReady 门控与 metadata schema”
```
你要为工业振动信号做 explain_factory。
约束：解释必须基于 data metadata（采样率/工况/domain/通道等），并提供 ExplainReady(m,x,μ)->(bool,reasons[])。
请输出：
1) metadata schema（字段名/类型/可选性/缺失时策略）
2) ExplainReady 的规则集合（每条规则给出 reason 文本）
3) 2 个最小 explainer 的接口与产物规范（文件路径/JSON schema）
4) 与主仓库 pipeline 的接入点（在哪个 factory/哪个阶段调用）
```

### Prompt D：生成“docs/paper_hub/<paper_id>.md 模板 + README 对齐清单”
```
你要把 7 篇 UXFD paper 的复现入口与主仓库对齐，但不修改 paper submodule 内的 README（它们是真相源）。

请输出：
1) 一个 docs/paper_hub/<paper_id>.md 的统一模板（包含：paper_id、上游链接、vibench config 路径、operator_graph 摘要、data metadata 要求、一键命令、常见坑）
2) 一个“README 对齐清单”：需要改/需要新增/禁止改的 README 列表（含理由）
3) 一个“复现证据链落盘规范”：输出目录下必须出现哪些文件（config snapshot、metrics、figures、explain artifacts）

要求：模板必须兼容 `python main.py --config ... --override ...` 的工作流；并显式写出“submodule 未初始化时主仓库不应报错”的处理策略。
```

---

## 7) 下一步（执行建议）

如果要立刻开始落地，建议先做 PR0（submodule 落位 + docs 入口），然后用 Prompt A 生成一个“可审计的 PR 级计划”，把输出贴回本文件作为 v2。

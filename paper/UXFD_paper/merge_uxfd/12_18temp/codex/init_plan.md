# UXFD 合并研究/重构计划（init_plan，已收敛）

本文件作为“背景与草案区”保留；最终执行以最终版为准：
- @`paper/UXFD_paper/merge_uxfd/12_18temp/codex/final_plan.md`

拆分文档（按职责解耦）：
- @`paper/UXFD_paper/merge_uxfd/12_21/codex/step_by_step_ops.md`
- @`paper/UXFD_paper/merge_uxfd/12_21/codex/submodule_config_conventions.md`
- @`paper/UXFD_paper/merge_uxfd/12_21/codex/model_alignment_plan.md`
- @`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`
- @`paper/UXFD_paper/merge_uxfd/12_21/codex/manifest_to_csv_spec.md`

目标：把 UXFD 框架，及其之下的 **7 篇 Paper 子项目**（来自 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper`）以“**Paper 资产 submodule + 主仓库统一运行框架**”的方式并入 PHM‑Vibench，并确保：

- 7 篇 Paper 的**语义、配置、复现入口**不漂移
- PHM‑Vibench 的**单一入口**与 **5‑block 配置模型**不被破坏
- 研究闭环可跑：`train/eval/explain/report`（agent 先只落盘 TODO 蒸馏，不接 LLM）

---

## 0) 真相源（SSOT）与硬约束

### 0.1 真相源（必须对齐的文档）

- 主仓库运行/架构约束：`README.md`、`AGENTS.md`、`CLAUDE.md`
- UXFD 7 篇 Paper 总览（上游）：`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/README.md`
- 上游的 paper 证据链与协议（可参考）：`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/doc/**`
- 映射文档（合并后口径）：每个 submodule 内的 `VIBENCH.md`（例如 `paper/UXFD_paper/<paper_id>/VIBENCH.md`）

### 0.2 硬约束（不满足就不合并）

- **单一入口不变**：`python main.py --config <yaml> [--override key=value ...]`
- **5‑block 配置不变**：`environment/data/model/task/trainer`
- **不新增第 6 个一级 block**：Explain/Collect/Report/Agent 等扩展配置必须挂在已有 block 内（优先 `trainer.*`）
- **扩展位建议统一**：优先采用 `trainer.extensions.*`（如现有 schema 不支持，在 PR3 引入兼容字段与校验）
- **工厂与 registry 驱动**：新增能力要走 `src/*_factory/` 的注册/装配路径
- **Paper 工作流隔离**：paper‑grade 实验/脚本放 `paper/`（submodule），主仓库的验收/测试不依赖 submodule 初始化
- **不重造 task 体系**：优先复用现有 `task_factory`/`trainer_factory`；仅做“paper config → vibench config”映射与组件增补
- **metadata = data metadata**：解释模块读取的是 dataset/batch 的元信息（采样率/工况/domain/通道等），不是 run_meta
- **可复现三件套**：配置文件、运行命令、输出目录（日志/表格/图）要能闭环追溯
- **依赖防污染**：主仓库 `requirements.txt` 为硬上限；Explain/Agent 的额外依赖必须 optional import + 可审计降级（不能因为缺依赖崩溃）

示例（扩展位，仅作建议口径）：

```yaml
trainer:
  name: DefaultTrainer
  extensions:
    explain:
      enable: true
      explainer: "router_weights"    # 或 "gradients" / "timefreq" / "fuzzy_rules" / ...
      save_artifacts: true
      llm: {enable: false}          # 仅做自然语言解释时开启；默认关闭
    report:
      enable: true
      manifest: true                # 生成 artifacts/manifest.json
    collect:
      enable: true                  # 汇总当前 run 或多 run（可选；默认也可只提供脚本）
    agent:
      enable: false
      mode: "todo_only"             # 默认只落盘 evidence，不调用 LLM
      llm: {enable: false}          # LLM 开关集中在 agent（推荐），避免 explain 逻辑混杂
```

### 0.3 术语与命名口径（避免漂移与歧义）

- `paper_id`：7 篇 paper 的稳定索引（只用于 preset 与各 submodule 内的 `VIBENCH.md`）
- `preset_id`：配置预设 id（默认 `preset_id == paper_id`；允许 `paper_id@v1` 版本化）
- `operator_id`：算子注册名（稳定字符串，可版本化；禁止用类名自动生成）
- `layout`：张量布局（例如 `BLC` / `BCL` / `BCHW`），贯穿 operator 的输入输出契约

---

## 1) 7 篇 Paper 清单（目录路径作为唯一 ID）

上游 `Paper/README.md` 的“官方顺序”是合并时的唯一编号口径；在主仓库侧落为 `paper/UXFD_paper/<paper_id>/` 的 submodule。

| # | paper_id（建议）     | 上游目录名                     | 角色（并入主仓库后）                                            |
| -: | -------------------- | ------------------------------ | --------------------------------------------------------------- |
| 1 | `fusion_1d2d`      | `1D-2D_fusion_explainable`   | 模态融合/对齐相关 operator + 数据元信息需求定义                 |
| 2 | `xfd_toolkit`      | `Explainable_FD_Toolkit`     | explain_factory 的 API/协议/可视化规范来源                      |
| 3 | `llm_xfd_toolkit`  | `LLM_Explainable_FD_Toolkit` | agent_factory 的“解释文本/对话”消费方（先不接 LLM）           |
| 4 | `moe_xfd`          | `MOE_explainable`            | MoE 路由/专家结构相关 operator（路径级可解释）                  |
| 5 | `fuzzy_xfd`        | `Paper_fuzzy_XFD`            | 规则/隶属度/审计相关 explainer 或 baseline 模块                 |
| 6 | `nesy_theory`      | `Neuralsymbolic_theory`      | 跨层抽象/命题/口径统一（主要沉淀为 docs）                       |
| 7 | `op_attention_tii` | `TII_operator_attention`     | 算子级注意力相关 operator + 理论/合成验证（docs+可选 operator） |

> `paper_id` 只用于主仓库“映射与配置默认值”的稳定索引；论文内容真相源仍在 submodule 内的 README 与文档。

---

## 2) 目录落位（主仓库 vs Paper submodule）

### 2.1 Paper submodule（只放论文资产/实验资产）

目标：把 7 篇 paper 的 README/文稿/实验脚本/图表等“论文资产”保持原样引入，并且让每篇 paper 的**配置文件与个性化实验**
都留在各自 submodule 内；同时不让主仓库 demos/tests 依赖它们才能运行/测试。

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
- 映射文档写在 submodule 内：`paper/UXFD_paper/<paper_id>/VIBENCH.md`（避免污染主仓库 `docs/`）
- 该 paper 的配置文件也写在 submodule 内：`paper/UXFD_paper/<paper_id>/configs/vibench/*.yaml`
- 个性化/临时/ablation 配置放：`paper/UXFD_paper/<paper_id>/configs/personal/`（可选）
- 不把 submodule 的依赖装进主仓库的 `requirements.txt`，避免污染主流程

### 2.2 主仓库（只放可复用的统一框架能力）

目标：把 7 篇 paper 共同需要的“运行框架能力”沉淀到主仓库，保证 config‑first 与可复用。

建议落位（与现有架构对齐）：

- 模型统一入口：`src/model_factory/X_model/`
  - 建议新增：`src/model_factory/X_model/uxfd_tspn/`（TSPN_UXFD：保持 `SignalProcessingLayer → FeatureExtractorlayer → Classifier`，外加 Registry/Adapters/HookStore）
  - UXFD 通用模块（建议按主题有序组织，避免 7 篇 paper 互相污染）：
    - `src/model_factory/X_model/UXFD/`：
      - `signal_processing_1d/`（对齐上游 `Signal_processing.py`）
      - `signal_processing_2d/`（对齐上游 `Signal_processing_2D.py`）
      - `fusion/`（对齐 `Fusion1D2D*.py`）
      - `fuzzy/`（对齐 `FuzzyLogic*.py`）
      - `operator_attention/`（对齐 `operator_attention*.py`）
      - `tspn/`（TSPN_UXFD 与其依赖：HookStore/Adapters/Registry）
  - 对比模型（来自 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection`）：
    - `src/model_factory/X_model/baselines/`（只保留模型 forward 与 vibench 适配封装）
  - 兼容策略：保留现有 `src/model_factory/X_model/{TSPN,Signal_processing,Feature_extract}.py` 行为不变（逐步迁移）
  - 注册策略：新增模型名（例如 `TSPN_UXFD` / `TFN` / `MCN` / `WKN` 等）注册到 `src/model_factory/model_registry.csv`
- explain 统一入口：新增 `src/explain_factory/`（主仓库可复用）
- agent 统一入口：新增 `src/agent_factory/`（先只落盘 TODO 蒸馏）
- paper→vibench 映射文档：放在各自 submodule 内的 `paper/UXFD_paper/<paper_id>/VIBENCH.md`

---

## 3) 统一框架设计：TSPN（算子可插拔）→ Explain → Collect/Report → Agent(TODO)

### 3.1 模型层：统一为“可插拔算子”的 TSPN

核心原则：不再“每篇一个模型文件”，而是 **一个 TSPN‑Compatible 容器（保持上游 TSPN 的层级结构）+ 一个稳定的 OperatorRegistry**。
这里的“图容器/可配置”并不是引入全新范式，而是把上游已经存在的“配置驱动组网”
（`signal_processing_configs` + `feature_extractor_configs`）工程化为 vibench 可追溯的 YAML 入口。

推荐阅读（解释为什么不偏离上游）：`paper/UXFD_paper/merge_uxfd/12_21/codex/model_alignment_plan.md`

7 篇 paper 的差异体现在：

- `operator_graph`（算子集合/拓扑）
- `operator_args`（每个算子参数）
- （可选）`hooks`（为 explain 暴露可解释中间节点）

建议把 Operator 抽象成最小接口（只约束张量契约 + 可选解释 hook）：

- `forward(x, meta) -> x'`
- `capabilities() -> dict`：声明可解释能力（是否有 operator 权重/路由概率/可命名特征等）
- `explain_hooks() -> dict`（可选）：暴露关键中间量（name/shape/语义/轴含义/单位），供 explain_factory 消费

#### 3.1.0 LayoutSpec + AdapterOperators（把高风险“形状不一致”降到可控）

上游实现中最常见的隐性风险是布局/形状混乱（`(B,L,C)`、`(B,C,L)`、`(B,C,H,W)` 混用）。
为避免 PR2 爆炸，建议在 operator 规范中强制声明 layout：

- 每个 operator 必须声明 `input_layout` / `output_layout`
- GraphConfigurableModel 负责在拼接时自动插入 AdapterOperator（或抛出可审计报错）

可审计的布局映射可写成：
`x_out = Permute(layout_in -> layout_out, x_in)`（实现上就是 `permute/reshape` 等）

AdapterOperator 建议至少包含：

- `PermuteAdapter`：`BLC <-> BCL`
- `ToTimeFreq2D` / `ToTimeSeries`：`BLC <-> BTFC`（当启用 `SP_2D/*` 时，贴近 `Signal_processing_2D.py` 的输出）

#### 3.1.1 结合上游 `Unified_X_fault_diagnosis/model/` 的统一收敛策略

上游目录（`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model`）已经覆盖 UXFD 7 篇 paper 的主要模型族与变体：

- **TSPN 家族**：`TSPN.py`、`TSPN_explainable.py`、`TSPN_sparse.py`、`TSPN_KAN.py`、
  `TSPN_OperatorAttention.py`、`TSPN_LLM_Enhanced.py`
- **注意力/路由/融合**：`operator_attention*.py`、`MoE*.py`、`Fusion1D2D*.py`
- **规则/逻辑/NeSy 基类**：`FuzzyLogic*.py`、`Logic_inference.py`、`explainable_base.py`、
  `llm_explainable_base.py`
- **信号处理/特征**：`Signal_processing.py`、`Signal_processing_2D.py`、`Feature_extract.py`、
  `FeatureExtractor.py`
- **其他模型族**（可作为 operator/head/backbone 纳入统一框架）：`DEN.py`、`TFON.py`、`NNSPN.py`、`kan.py`
- **配置与组网**：`config.py`、`parse_network.py`

并入 PHM‑Vibench 时，不建议把这些文件按“paper 维度”搬进主仓库，而是按“可复用能力维度”收敛：

1) 主仓库沉淀 `TSPN_UXFD`（或同等命名）+ `OperatorRegistry` + 最小必要算子实现
2) paper submodule 保留原始实现作为“语义真相源/复现对照”
3) 通过 `paper_id -> operator_graph preset` 把 paper 的差异固定为配置差异（避免代码分叉）

#### 3.1.2 把“文件级实现差异”降维为“operator 类型差异”

> 目标：7 篇 paper 最终应该只是 7 个 `operator_graph` preset，而不是 7 套模型代码。

建议 operator 类型（与上游文件对齐）：

- `SP_1D/*`：1D 信号处理算子（对应 `Signal_processing.py`）
- `SP_2D/*`：2D/时频算子（对应 `Signal_processing_2D.py`）
- `FE/*`：可解释特征算子（对应 `Feature_extract.py` / `FeatureExtractor.py`）
- `FUSION/*`：多分支/多模态融合算子（对应 `Fusion1D2D*.py`）
- `ROUTER/OP_ATT`：算子级注意力（对应 `operator_attention*.py`）
- `ROUTER/MOE`：专家路由（对应 `MoE*.py`）
- `REASONER/FUZZY`：规则/隶属度推理（对应 `FuzzyLogic*.py`）
- `REASONER/LOGIC`：逻辑推理链（对应 `Logic_inference.py`）
- `HEAD/*`：任务输出头（分类/回归/多任务；可吸纳 `DEN/TFON/NNSPN/KAN` 的“头/骨干”能力）

#### 3.1.3 stage 固化：让 graph 可控、可解释、可汇总

建议把 `operator_graph` 固化为固定 stage（便于兼容与解释产物结构化）：

1) `preprocess_1d`（可空）
2) `preprocess_2d`（可空；若启用需定义轴语义）
3) `feature`
4) `fusion`（可空）
5) `router`（可空）
6) `reasoner`（可空）
7) `head`

Explain 层的 hook 约定按 stage 输出，天然支持：

- operator‑level：哪个算子/专家最重要
- stage‑level：哪个阶段贡献最大
- sample/domain‑level：跨工况/跨域的解释一致性统计

#### 3.1.4 HookStore（统一抓取点，避免解释产物散落在各 operator 里）

建议把“中间量/权重/路由概率”等 hooks 的收集职责集中在 GraphConfigurableModel：

- operator 只负责通过 `explain_hooks()` 暴露中间量（不负责落盘）
- GraphConfigurableModel 在一次 forward 中统一收集 hooks，写入一个 `HookStore`（内存结构）
- explain_factory 只读 `HookStore + data metadata`，并生成 explain artifacts

这样可以保证：

- hooks 命名与 layout 可统一规范（便于跨 paper 汇总）
- 解释产物的落盘路径在一个地方控制（便于 manifest/collect/report）

### 3.2 配置层：paper config → vibench 5‑block config 的“稳定映射”

目标：7 篇 paper 的复现入口最终都能落到：
`python main.py --config <yaml> --override ...`

约定（建议）：

- **主仓库侧**只提供“UXFD 组件能力 + 映射适配器”，不强行把 paper configs 写进 `configs/config_registry.csv`
- **paper submodule 侧**提供 paper‑grade configs（引用主仓库的模型/解释能力）

建议新增一个 `paper_id` 映射表（实现层可以是 `config_adapter` 或纯 YAML 预设）：

- `paper_id -> default operator_graph + default operator_args + explain requirements`

#### 3.2.1 统一模型与组网配置（建议 YAML 约定）

目标：同时兼容

- PHM‑Vibench 的 5‑block config（`environment/data/model/task/trainer`）
- 上游 `model/config.py` + `parse_network.py` 的“组网思路”
- TSPN 的 operator_graph（stage 化）

建议把“graph 与算子参数”放在 `model` block 内（并确保可被 CLI `--override` 覆盖）：

```yaml
model:
  type: "X_model"
  name: "TSPN_UXFD"                 # 新增/兼容命名（避免破坏既有 TSPN）

  # 基本形状/输出（task 可覆盖，但建议模型侧也能独立跑通）
  in_channels: 2
  num_classes: 5

  # stage 化 graph（强约束字段，便于 explain/collect 对齐）
  operator_graph:
    preprocess_1d:
      - {op: "SP_1D/I", args: {}}
      - {op: "SP_1D/WF", args: {wavelet: "db4"}}
      - {op: "SP_1D/HT", args: {}}
      - {op: "SP_1D/FFT", args: {}}
    feature:
      - {op: "FE/Mean", args: {}}
      - {op: "FE/Std", args: {}}
    router:
      - {op: "ROUTER/OP_ATT", args: {temperature: 0.1}}
    head:
      - {op: "HEAD/Linear", args: {hidden_dim: 128}}

  # paper preset（可选）：只作为默认 graph 入口；显式 operator_graph 优先
  paper_id: "op_attention_tii"
  preset_version: "v1"
```

落地规则（写进 config_adapter 的 deterministic 逻辑）：

- 若用户只提供 `paper_id`：用 `(paper_id, preset_version)` 填充 `operator_graph` 与默认 args
- 若用户显式提供 `operator_graph`：以显式 graph 为准（paper_id 仅作为记录标签）
- 若上游 paper 的 config 是“层列表式”（例如 `signal_processing_configs` / `feature_extractor_configs`）：
  先翻译成 stage 化 graph，再进入统一装配（避免“同一个算子在不同论文里叫法不同”）

#### 3.2.2 Presets + 镜像配置自动生成（降低维护成本，CI 可用）

为避免“7 篇 paper 的关注点不同导致配置互相污染”，UXFD 的 paper configs **全部放在各自 submodule 内**，
主仓库只提供通用 base_configs 与通用组件（data/model/task/trainer/explain/report/collect/agent）。

推荐落位（在对应 submodule 内）：
- `paper/UXFD_paper/<paper_id>/configs/vibench/`：该 paper 的 vibench 入口 YAML（5‑block）
- `paper/UXFD_paper/<paper_id>/configs/personal/`：个性化/临时/ablation（可选，不进入主线）
- `paper/UXFD_paper/<paper_id>/VIBENCH.md`：该 paper 的唯一“怎么用 vibench 跑”的入口文档

submodule configs 的写法建议：
- 以主仓库 `configs/base/**` 作为 `base_configs`（减少重复、保持主仓库口径）
- paper 仅覆盖自身差异（operators、fusion、fuzzy、解释器选择、metadata 要求等）

说明：
- 主仓库不维护每篇 paper 的完整 configs（避免“主 docs 被 paper 细节弄乱”）
- 主仓库 demos/tests 只保证自身 `configs/demo/**` 可跑（不依赖 submodule 初始化）

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

#### 3.3.1 Explain 的目标：把“结构可解释”变成“可投稿证据链”

Explain 层的输出要同时服务三类需求：

1) 工程可审计：解释为什么能/不能用（ExplainReady reasons），并把解释结果结构化落盘
2) 研究可比较：跨模型/跨数据集/跨工况能做统一统计（faithfulness/stability/efficiency）
3) 论文可引用：自动生成可用的图表与表格输入（collect/report 可汇总）

因此解释对象不是 `x` 本身，而是三元组 `(x, y_hat, μ)`：输入信号、模型输出、data metadata。

#### 3.3.2 为什么必须用 data metadata（方法论依据）

工业振动信号的解释需要“物理语义对齐”，这一步只能靠 data metadata：

- 没有 `sampling_rate`：频域解释（FFT/STFT）无法标注 Hz，跨数据集无法比较
- 没有 `sensor`/单位/安装位置：通道重要性无法被人读懂，也无法做跨通道一致性
- 没有 `operating_condition/domain`：DG/跨工况解释无法按域分组统计（也无法做稳定性结论）
- 没有 `window_length/stride`：时序归因无法对应到物理时间尺度与滑窗位置

结论：ExplainReady 的第一门槛就是 metadata schema；缺失时必须“降级输出”而不是 silent fail。

#### 3.3.3 Metadata 穿透：最小侵入接入策略（尽量不改 task）

落地时最常见的阻塞点是：trainer/task 只接收 `(x, y)`，解释模块却需要 `meta`。
建议用“只改一处”的策略解决：

- 增加统一 batch 解包函数：`unpack_batch(batch) -> (x, y, meta)`
- 支持 batch 形态：`(x,y)` / `(x,y,meta)` / `{"x":..,"y":..,"meta":..}`
- 如果 meta 缺失：返回 `{}`，并标注 `meta_source="default"`（由 explain_factory 负责 reasons 与降级产物）

推荐在输出目录落盘一份可审计快照（无论 explain 是否启用都可生成）：

- `artifacts/data_metadata_snapshot.json`
  - `meta_source`: `"batch" | "dataset" | "default"`
  - `degraded`: `true/false`
  - `missing_keys`: `[...]`

#### 3.3.4 ExplainReady：可扩展的门控规则（建议）

ExplainReady 不是一个 if，而是一组可组合规则（输出 reasons，便于排错与审计）：

- `schema_ok`：metadata 是否包含 explainer 需要的 key（缺哪些 key）
- `model_hook_ok`：模型是否暴露 explain_hooks（缺哪些 stage/operator 输出）
- `input_ok`：输入 shape/通道数/窗口长度是否满足解释器约束
- `task_ok`：当前任务类型是否支持（分类/回归/多任务/异常检测）

建议 reasons 统一为 `短码:字段/模块` + 人类可读文本，例如：

- `MISSING_META:sampling_rate`（频域图与频带归因不可用）
- `MISSING_META:sensor`（只能输出 channel_index，不能输出物理通道名）
- `MODEL_NO_HOOK:router`（无法输出 operator 权重/路由概率）
- `UNSUPPORTED_TASK:anomaly_detection`（该解释器暂未覆盖）

建议把门控结果固定落盘为：

- `artifacts/explain/eligibility.json`（最小字段集）
  - `ok: bool`
  - `reasons: [{code, message, suggestion, missing_keys?}]`
  - `meta_source: "batch" | "dataset" | "default"`
  - `degraded: bool`

#### 3.3.5 解释方法族：内生（intrinsic）+ 后验（post‑hoc）

Explain 方法建议显式区分两类，并在产物中标注 `method_family`：

**A) 内生解释（最符合 UXFD 的“算子/规则可审计”主张）**

- Operator Importance / Router Weights
  - 原理：`ROUTER/OP_ATT` 的算子权重、`ROUTER/MOE` 的路由概率本身就是可解释证据
  - 输出：per‑sample/per‑domain 的 operator 权重分布；可汇总为 “Top‑k operators per dataset”
- Feature Evidence（统计/物理特征）
  - 原理：`FE/*` 的输出是可命名特征（均值、方差、谱能量、包络峰值等），可直接做贡献分解与排名
  - 输出：Top‑k 特征 + 值 + 贡献（对 logit/损失的边际影响）
- Rule / Logic Trace（Fuzzy/Logic/NeSy）
  - 原理：规则命中、隶属度、推理链天然可审计，解释能输出“为什么判为某类”
  - 输出：规则命中列表、隶属度曲线、推理路径（含中间命题/变量）

**B) 后验解释（对任意可微模型通用，用于对齐评估协议）**

- Gradient / Integrated Gradients（时间×通道）
  - 原理：计算 `∂y/∂x` 或积分梯度，得到每个时间点/通道的敏感度
  - 依赖：不强依赖 metadata，但 metadata 决定轴标签/单位/通道名
- Occlusion / Deletion（Faithfulness）
  - 原理：遮挡某段时间窗或某频带，观察输出变化，得到 deletion curve（faithfulness 指标）
  - 依赖：时间窗需要 `window_length/stride`；频带遮挡需要 `sampling_rate`
- Counterfactual（可选增强）
  - 原理：在物理可行约束下求最小扰动使预测翻转，解释“决策边界附近关键证据”
  - 依赖：物理/数据约束（来自 metadata 或数据规范）

说明（继承现有方法，避免重复造轮子）：
- 优先从 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection/GradCAM_XFD.py` 等现有实现移植/封装
  到 `src/explain_factory/explainers/`，保持接口统一
- 对常规梯度类方法（grad/IG/occlusion）尽量不e用纯 PyTorch 实现，避免强依赖第三方库；如需 Captum 等依赖必须
  optional import + eligibility 降级（不能 ImportError 崩溃）

#### 3.3.6 Explain 统一产物（建议落盘规范）

每次 explain 至少产出：

- `artifacts/explain/summary.json`：run 级总结（paper_id、operator_graph 摘要、ExplainReady reasons、方法族列表）
- `artifacts/explain/per_sample/*.json`：样本级解释（operator 权重/重要时间段/重要频带/规则命中等）
- `figures/explain/*.png`：论文级图（带通道名、单位、Hz 轴、工况/domain 标注）

并为 collect/report 提供可汇总输入：

- `reports/explain_metrics.csv`：faithfulness/stability/efficiency（按 dataset/domain/model 聚合）

#### 3.3.7 explainer 的 YAML 类别（建议：由 `trainer.extensions.explain.explainer` 选择）

为避免 explain_factory “一个入口塞所有解释器”，建议把解释器选择显式下沉到 YAML：

- `trainer.extensions.explain.explainer: <explainer_id>`
- `trainer.extensions.explain.explainer_args.*`（可选）

解释器建议最小集合（先覆盖 UXFD 核心路径）：

- `router_weights`：读取 HookStore 的 `ROUTER/*` 权重/路由概率（内生解释）
- `timefreq`：读取 `SP_2D/*` 的时频图（或由 metadata 决定是否生成），输出时频可视化与频带归因
- `gradients`：对任意可微模型做梯度/积分梯度（后验解释）
- `fuzzy_rules`：输出规则命中/隶属度/推理链（内生解释）

#### 3.3.8 LLM 自然语言解释的开关（建议：由 agent 承担）

自然语言解释（LLM）建议不要直接塞进 explain_factory 的核心逻辑里，而是作为“消费结构化解释”的后处理：

- explain_factory 负责：结构化解释 + 可视化 + eligibility
- agent_factory 负责：读取结构化解释与 metadata，生成自然语言解释（可选）

配置建议：

- `trainer.extensions.agent.enable: true/false`
- `trainer.extensions.agent.mode: "todo_only" | "llm"`
- `trainer.extensions.agent.llm.enable: true/false`（默认 false，避免引入网络/依赖）

这样做的好处：

- explain_factory 在 core requirements 环境中始终可运行（最重要）
- LLM 依赖与网络策略可完全隔离（符合“paper 工作流隔离”和“依赖防污染”）

### 3.4 Collect/Report：结果闭环（不绑定外部服务）

目标：把 `train/eval/explain` 的输出整理成可被论文引用的“证据链产物”：

- `metrics.json/csv`（指标表）
- `figures/`（解释图、关键可视化）
- `config_snapshot.yaml`（运行时 resolved config）

主仓库只提供“收集脚本/规范”；paper submodule 可以提供“论文写作需要的汇总表/图”的高级脚本。

#### 3.4.1 `artifacts/manifest.json`（强烈建议，供 paper 脚本稳定消费）

为避免“文件名/路径漂移”，建议主仓库在每次 run 结束时生成索引文件：

- `artifacts/manifest.json`（最小字段集）
  - `paper_id`, `preset_version`, `run_id`
  - `config_snapshot`
  - `metrics`（路径或列表）
  - `figures_dir`
  - `data_metadata_snapshot`
  - `eligibility`（若 explain 启用）
  - `explain_dir` / `explain_summary`（若 explain 启用）
  - `distilled_dir`（若 agent todo 启用）

可选增强（便于解释/调试复现）：

- `layout_chain`：GraphConfigurableModel 拼接时的 layout 流（含自动插入的 adapters）
- `hooks_index`：hooks 的索引文件路径（例如 `artifacts/hooks/index.json`）

#### 3.4.1.1 建议把 manifest 汇总为 CSV（更清晰、便于 Excel/论文表）

`manifest.json` 是“单次 run 的索引”。为了更易读、更易对齐论文表，建议在 collect 阶段把一批 runs 的
manifest 汇总为 CSV：
- 运行级总表：`reports/uxfd_runs.csv`（一行一个 run）
- （可选）解释级总表：`reports/uxfd_explain.csv`（一行一个 run 或一个 explainer）

CSV 字段建议稳定化（示例列，允许缺失）：
- 身份与版本：`run_id,paper_id,preset_version,config_snapshot`
- 数据与任务：`dataset_id,dataset_name,domain_id,operating_condition,task_name`
- 指标：`metric/acc,metric/f1,metric/auc,...`（按项目实际输出展开）
- explain：`explain_ok,explainer_id,explain_reasons,explain_summary_path`
- 产物路径：`manifest_path,metrics_path,figures_dir,explain_dir,distilled_dir`
- 审计：`meta_source,degraded,missing_keys`

扁平化规则（推荐写死在 collect 脚本里）：
- dict 用稳定分隔符展开（建议 `metric/acc` 这种“前缀/字段”风格，避免 `.` 与 override 冲突）
- list/复杂结构用 JSON 字符串写入单元格（必要时额外导出明细 CSV）

#### 3.4.2 Collect（为什么计划里要显式出现）

`collect` 的职责是“跨 run 的汇总与对齐”，它和 `report`（单次 run 的产物规范）不同：

- `report`：在**一次 run 内**生成 manifest、固定目录结构、把 explain/metrics/figures 写全
- `collect`：在**多个 run 之间**读取 manifest，合并成论文可用的总表（CSV/JSON），并按 `paper_id/dataset/domain/model` 分组

落地形态可以二选一（都不新增一级 block）：

1) 脚本形态（推荐先做）：`scripts/collect_uxfd_runs.py` 读取 `save/**/artifacts/manifest.json`
2) 扩展形态（可选）：`trainer.extensions.collect.enable=true` 时对当前 output_dir 做一次“局部 collect”

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
- 在每个 submodule 内新增 `VIBENCH.md`（或提供模板，初始化后在 submodule repo 内提交）

验收：

- `paper/UXFD_paper/README.md` 里能点击到 7 篇（哪怕 submodule 还未 init，也不应破坏主仓库）
- 主仓库 `README.md`/`paper/README_SUBMODULE.md` 的口径不冲突（必要时只补链接，不改主叙事）

### PR1：TSPN “算子可插拔”骨架（不搬 paper‑specific 代码）

改动：

- 实现/新增 GraphConfigurableModel（解析 `operator_graph`、传递 `meta`、统一收集 hooks）
- 引入 `OperatorRegistry` + 最小可用 operators（先覆盖 1 篇 paper 的最稳定子集）
- 引入 `LayoutSpec + AdapterOperators`（至少 `BLC <-> BCL`），降低形状不一致风险
- 为避免破坏旧行为：新增 `TSPN_UXFD`（或同等命名）并注册到 `src/model_factory/model_registry.csv`
- HookStore：由 GraphConfigurableModel 统一收集 hooks（operator 不允许自行落盘解释产物）
- 优先复用上游 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/{Signal_processing,Signal_processing_2D,TSPN}.py` 的接口与张量约定（见 `paper/UXFD_paper/merge_uxfd/12_21/codex/model_alignment_plan.md`）

验收：

- `python -m scripts.config_inspect --config <uxfd_demo.yaml>` 能看到模型 target 指向正确模块
- 最小 smoke：能前向 + 能跑 1 个 batch（不要求指标）

### PR1b：对比模型（`model_collection`）整理进 `X_model`（便于统一跑表）

改动：
- 从 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection/` 选择“对比基线”移植到主仓库：
  - 目标落位：`src/model_factory/X_model/baselines/`
  - 仅保留“模型定义 + forward”，删除/隔离其自带 trainer/dataset/脚本
  - 统一适配 vibench 模型签名（与现有 `model_factory` 一致）
- 在 `src/model_factory/model_registry.csv` 注册这些 baseline（避免与现有 `CNN/Transformer/...` 重名冲突）
 - 详细落位与命名建议见：`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`

验收：
- `python -m scripts.config_inspect --config <baseline_demo.yaml>` 能定位到正确模块
- 任选 1 个 baseline 跑 1 epoch（用 dummy 数据或你们的 smoke 数据）

### PR2：7 篇 paper 的 config 映射（paper_id → operator_graph）

改动：

- 每个 submodule 内落地该 paper 的 vibench configs：`paper/UXFD_paper/<paper_id>/configs/vibench/*.yaml`
- 产出/更新映射文档：`paper/UXFD_paper/<paper_id>/VIBENCH.md`（一键命令 + metadata 要求 + 产物规范）
- submodule configs 的目录与写法规范见：`paper/UXFD_paper/merge_uxfd/12_21/codex/submodule_config_conventions.md`
- （可选）增加 `config_adapter`：仅用于兼容“上游旧参数风格”（例如 `signal_processing_configs`）到
  `model.operator_graph`，让 submodule configs 可逐步从旧写法迁移到新写法

验收：

- 每个 submodule 的 `paper/UXFD_paper/<paper_id>/VIBENCH.md` 都提供一条可复制命令：`python main.py --config ...`
- 7 篇至少能完成：构建模型 + 走到训练 step（不要求指标对齐）

### PR3：Explain_factory（metadata_reader + ExplainReady + 最小 explainer）

改动：

- `src/explain_factory/metadata_reader.py`：从 dataset/batch 抽取 metadata（统一 schema）
- `src/explain_factory/eligibility.py`：ExplainReady 门控（返回 reasons，日志可审计）
- `src/explain_factory/explainers/`：先落 1–2 个最小解释器（例如 operator‑level attribution + 时域/频域重要性）
- 增加 `unpack_batch(batch)`（统一 meta 穿透；尽量不改 task 接口）
- 依赖降级策略：只装 core requirements 时也要能产出 `eligibility.json` / `data_metadata_snapshot.json`（不能 ImportError 崩溃）
- 增加 `trainer.extensions.explain.explainer` 的选择逻辑（explainer_id → explainer 实现）

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
- 生成 `artifacts/manifest.json`（最小字段集固定），供 paper submodule 稳定消费

验收：

- 汇总脚本能生成总表（CSV/JSON），并能被 paper submodule 的写作脚本消费

### PR5：agent_factory（仅落盘 TODO 蒸馏内容）

改动：

- 定义 `distilled/` 的路径规范与 frontmatter（paper_id / dataset_task / operator_graph / findings / issues / next）
- （可选）生成蒸馏草稿的轻量脚本
- LLM 开关预留：`trainer.extensions.agent.mode="llm"`（默认不启用，不让主流程依赖网络）

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

- 每篇 paper 的 `paper/UXFD_paper/<paper_id>/VIBENCH.md` 固化：`operator_graph + 关键超参 + explain metadata 要求`
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
   - 不新增第 6 个一级 block（explain/agent/report 必须挂在 trainer.extensions 或等价位置）
   - 模型必须保持上游 TSPN 的层级结构（SignalProcessingLayer/FeatureExtractorlayer/Classifier），并通过 TSPN_UXFD 形式接入 OperatorRegistry（含 LayoutSpec/AdapterOperators）
   - paper 内容必须以 submodule 形式引入到 paper/UXFD_paper/
   - 不让主仓库测试依赖 submodule
   - requirements.txt 为硬上限；可选依赖必须可审计降级（不可 ImportError 崩溃）

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
输出：
- 一个可直接落地的 vibench YAML 模板（含 `trainer.extensions` 的开关位）
- 该 paper 对应的 submodule 配置文件（建议落位：`paper/UXFD_paper/<paper_id>/configs/vibench/<name>.yaml`）
  -（可选）若要拆分差异片段，也应放在 submodule 内（例如 `paper/UXFD_paper/<paper_id>/configs/presets/*.yaml`）
- 映射解释表（字段->来源->消费方），并标注稳定 `operator_id`（可版本化）
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

硬约束补充：
- 必须说明 `layout(BLC/BCL/...)` 与 AdapterOperators（至少 `BLC <-> BCL`）
- 必须说明 `unpack_batch(batch)->(x,y,meta)` 的 meta 穿透策略（尽量不改 task 接口）
- 在仅安装 core requirements 的环境中：即使解释器不可用，也必须产出 `eligibility.json`（不能 ImportError 崩溃）
```

### Prompt D：生成“submodule 内 VIBENCH.md 模板 + README 对齐清单”

```
你要把 7 篇 UXFD paper 的复现入口与主仓库对齐，但不修改 paper submodule 内的 README（它们是真相源）。

请输出：
1) 一个 paper/UXFD_paper/<paper_id>/VIBENCH.md 的统一模板（包含：paper_id、上游链接、vibench config 路径、operator_graph 摘要、data metadata 要求、一键命令、常见坑）
2) 一个“README 对齐清单”：需要改/需要新增/禁止改的 README 列表（含理由）
3) 一个“复现证据链落盘规范”：输出目录下必须出现哪些文件（config snapshot、metrics、figures、explain artifacts）

要求：模板必须兼容 `python main.py --config ... --override ...` 的工作流；并显式写出“submodule 未初始化时主仓库不应报错”的处理策略。
```

---

## 7) 下一步（执行建议）

如果要立刻开始落地，建议先做 PR0（submodule 落位 + docs 入口），然后用 Prompt A 生成一个“可审计的 PR 级计划”，把输出贴回本文件作为 v2。

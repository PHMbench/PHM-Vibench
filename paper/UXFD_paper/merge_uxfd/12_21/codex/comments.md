好的，这是为您整合了所有高价值建议（Submodule 隔离、GraphModel 算子化、配置继承、扩展位规范、闭环证据链）的 **v1.1 正式版计划**。

这份计划现在**工程落地性极强**，既保护了主仓库的架构纯洁性，又给 7 篇 Paper 的多样性留足了“软着陆”空间。你可以直接复制全文覆盖原计划。

---

# UXFD 合并研究/重构计划 (v1.1)

**版本**：v1.1 (Final Engineering Plan)
**目标**：把 UXFD 框架下的 **7 篇 Paper 子项目** 以“**Paper 资产 submodule + 主仓库统一运行框架**”的方式并入 PHM‑Vibench，实现工程与科研的解耦。

---

## 0) 真相源 (SSOT) 与 硬约束

### 0.1 真相源 (SSOT)

* **主仓库架构约束**：`README.md`、`AGENTS.md`、`CLAUDE.md`
* **论文语义/复现标准**：Submodule 内部的 `README.md` 与原始脚本（位于 `paper/UXFD_paper/<paper_id>/`）
* **合并后的运行口径**：`docs/paper_hub/<paper_id>.md`（说明如何在 Vibench 中复现）

### 0.2 工程硬约束 (Critical Constraints)

1. **单一入口不变**：必须通过 `python main.py --config <yaml> [--override ...]` 启动。
2. **5-block 核心语义不变**：配置必须包含 `environment/data/model/task/trainer`。
3. **扩展位规范**：所有新增能力（explain/collect/report/agent）必须挂载于 **`trainer.extensions`**，**严禁**新增第 6 个一级 block。
```yaml
trainer:
  name: DefaultTrainer
  extensions:
    explain: {enable: true, ...}
    agent: {enable: false, mode: "todo_only"}

```


4. **工厂驱动**：新增能力必须走 `src/*_factory/` 的注册与装配路径。
5. **Paper 工作流隔离**：主仓库的 CI/CD、单元测试、Demos **绝不允许** 依赖 Submodule 初始化。
6. **Metadata 驱动解释**：解释模块只消费 `dataset/batch` 里的元信息（物理量/采样率等），不依赖训练日志。
7. **依赖防污染**：主仓库 `requirements.txt` 是上限；Paper 的冲突依赖只能在其独立容器/venv 中运行，不得污染主环境。

---

## 1) 7 篇 Paper 清单与角色

在主仓库侧，通过 `paper_id` 作为唯一索引；在文件系统侧，落位为 `paper/UXFD_paper/<paper_id>/` 的 submodule。

| # | paper_id | 角色与能力贡献 | 对应上游目录 |
| --- | --- | --- | --- |
| 1 | `fusion_1d2d` | **Operator**: 模态融合, 对齐 adapter | `1D-2D_fusion_explainable` |
| 2 | `xfd_toolkit` | **Explain**: 协议定义, 可视化规范 | `Explainable_FD_Toolkit` |
| 3 | `llm_xfd_toolkit` | **Agent**: 解释文本消费 (暂不接 LLM) | `LLM_Explainable_FD_Toolkit` |
| 4 | `moe_xfd` | **Operator**: MoE Router (路径可解释) | `MOE_explainable` |
| 5 | `fuzzy_xfd` | **Operator**: 模糊规则层, 隶属度计算 | `Paper_fuzzy_XFD` |
| 6 | `nesy_theory` | **Docs**: 跨层抽象理论 (仅文档) | `Neuralsymbolic_theory` |
| 7 | `op_attention_tii` | **Operator**: 算子级注意力 | `TII_operator_attention` |

---

## 2) 目录落位策略

### 2.1 Paper Submodule (资产区)

*位置*：`paper/UXFD_paper/`
*原则*：保持原样，不修改，作为“历史真相源”。

```text
paper/UXFD_paper/
  README.md                # 索引页
  README_SUBMODULE.md      # 初始化指南
  fusion_1d2d/             # git submodule
  ...

```

### 2.2 主仓库 (能力区)

*位置*：`src/` & `configs/`
*原则*：提供统一框架，**即使 Submodule 为空也能运行**。

* **模型层**：`src/model_factory/X_model/uxfd_tspn/` (含 GraphModel, Operators, Adapters)
* **配置层**：
* `configs/presets/uxfd/` (Paper 的配置差异/片段)
* `configs/reference/uxfd_min/<paper_id>.yaml` (**最小镜像配置**，主仓库 CI 用)


* **文档层**：`docs/paper_hub/<paper_id>.md` (复现指南)

---

## 3) 统一框架设计 (Core Architecture)

### 3.1 模型层：GraphConfigurableModel (算子流水线)

不再追求单一的“万能 TSPN”，而是实现一个 **图/流水线容器** + **算子库**。

* **容器 (`GraphConfigurableModel`)**：负责按 Config 实例化算子并串联 `forward`。
* **算子 (`Operators`)**：实现具体的逻辑 (Attention, Conv, Fusion)。
* **适配器 (`Adapters`)**：解决接口不一致问题 (e.g., `PermuteAdapter`, `MaskInjector`).

**Operator 接口规范 (ABC)**：

```python
class BaseOperator(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor: ...

    def explain_hooks(self) -> Dict[str, Any]:
        """返回中间变量: {name: tensor, semantic: 'attention', axes: ['time']}"""
        return {}

    def capabilities(self) -> Dict[str, Any]:
        """能力声明: {'needs_mask': True, 'layout': 'BCL'}"""
        return {}

```

### 3.2 配置层：Presets + Override

避免硬编码映射，采用“继承+覆盖”模式：

1. **Base**: 引用主仓库现有的 `configs/base/model/backbone_*.yaml`。
2. **Preset**: `configs/presets/uxfd/<paper_id>.yaml` 定义算子图 (`operator_graph`) 和特有参数。
3. **Reference**: `configs/reference/uxfd_min/<paper_id>.yaml` 组装完整配置，作为一键运行入口。

### 3.3 Explain 层：Metadata 穿透与门控

* **Metadata 来源优先级**：
1. `batch` (如果 `data_factory` 返回的是 dict 或 tuple 含 meta)。
2. `dataset.get_metadata()`。
3. **Fallback**: `DefaultMetadata` (填补默认值，并在日志中警告)。


* **门控入口**：


* **产物**：
* `artifacts/data_metadata_snapshot.json` (快照)
* `artifacts/explain/eligibility.json` (审计通过/拒绝原因)



### 3.4 Collect/Report：闭环证据链

除了 Metrics 和 Figures，必须生成 **索引文件** 供 Paper 脚本消费：

* **`artifacts/manifest.json`**：
```json
{
  "metrics": "metrics.csv",
  "config": "config_snapshot.yaml",
  "explain_dir": "artifacts/explain/",
  "figures": ["figures/confusion_matrix.png"]
}

```



### 3.5 Agent：TODO 蒸馏

* **模式**：`mode="todo_only"` (仅生成结构化数据，不调用 LLM)。
* **路径**：`distilled/<paper_id>/<run_id>/evidence.json`。

---

## 4) 执行路线 (PR Breakdown)

### PR0：架构落位 (Skeleton & Docs)

* **内容**：建立 `paper/UXFD_paper` 结构，创建 `docs/paper_hub/`，创建 `configs/reference/uxfd_min/` 空目录。
* **验收**：主仓库测试/Demo 运行不受影响；文档链接有效。

### PR1：核心骨架 (GraphModel + Adapters)

* **内容**：
* 实现 `src/model_factory/X_model/uxfd_tspn/graph_model.py`。
* 实现 `BaseOperator` ABC 和 `AdapterOperators` (如 Permute)。
* 注册 `GraphModel` 到 Registry。


* **验收**：编写一个 Unit Test，用 GraphModel 串联 `[Linear -> Permute -> Linear]`，跑通 forward。

### PR2：配置迁移 (Presets & Min Configs)

* **内容**：
* 建立 `configs/presets/uxfd/`。
* 产出 7 篇 Paper 的 **最小镜像配置** (`configs/reference/uxfd_min/*.yaml`)。
* 更新 `docs/paper_hub/<paper_id>.md` 指向这些配置。


* **验收**：7 条命令 `python main.py --config configs/reference/uxfd_min/xxx.yaml` 全部能跑通 Training Loop。

### PR3：Explain Factory (Metadata & Gate)

* **内容**：
* 实现 `metadata_reader` (含 Fallback)。
* 实现 `ExplainReady` 门控。
* 产出 `eligibility.json`。


* **验收**：手动运行 Config，修改 metadata 为空，检查 `eligibility.json` 是否包含正确的错误码 (Reasons)。

### PR4：Report & Manifest

* **内容**：
* 在 Trainer 结束时生成 `artifacts/manifest.json`。
* 提供 `scripts/collect_uxfd_results.py` 读取 manifest 汇总。


* **验收**：运行脚本能生成 7 篇 Paper 的汇总 CSV。

### PR5：Agent (Distillation)

* **内容**：实现 `todo_only` 模式，落盘 `distilled/` 结构化数据。
* **验收**：检查输出目录结构符合规范。

---

## 5) Prompt Pack (v1.1)

用于辅助生成的 Prompt，已更新以匹配上述架构。

### Prompt A: 生成详细 PR 执行文档

```markdown
# Role
资深架构师。

# Task
基于 UXFD v1.1 合并计划，为 **PR1 (GraphModel)** 和 **PR2 (Config Migration)** 生成详细的实施细则。

# Requirements
1. **Model**: 设计 `GraphConfigurableModel` 类，展示它如何读取 YAML 中的 `operator_graph` 列表并用 `nn.Sequential` 或类似机制组装。
2. **Adapters**: 必须包含 `PermuteAdapter` 的代码，演示如何处理 (B, L, C) <-> (B, C, L) 的转换。
3. **Config**: 展示一个 YAML 片段，利用 `configs/presets/` 继承机制，只覆盖 `operator_graph` 部分。
4. **Validation**: 给出一段 Python 脚本，用于在不启动完整 Trainer 的情况下，快速验证模型构建是否成功。

```

### Prompt B: 生成 ExplainFactory 核心逻辑

```markdown
# Role
XAI 工程师。

# Task
实现 UXFD v1.1 计划中的 **ExplainReady** 门控与 **Metadata** 读取器。

# Requirements
1. **Metadata Reader**: 写一个函数，能从 (x, y), (x, y, meta) 或 dict 中提取 meta，如果提取失败，返回默认值并标记 `is_fallback=True`。
2. **ExplainReady**: 实现公式 `(m, x, meta) -> (bool, reasons)`。
3. **Artifacts**: 定义 `eligibility.json` 的 JSON Schema，必须包含 `code` (错误码) 和 `suggestion` (修复建议)。
4. **Integration**: 展示如何在 `trainer.extensions.explain` 开启时调用上述逻辑。

```

### Prompt C: 生成文档模板

```markdown
# Role
技术文档工程师。

# Task
为 `docs/paper_hub/<paper_id>.md` 生成标准模板。

# Requirements
模板必须包含：
1. **Status Badge**: 显示主仓库镜像配置 (`configs/reference/uxfd_min/`) 是否可跑。
2. **Mapping**: 对比表格，展示 "Paper 原配置参数" vs "Vibench Preset 参数"。
3. **Quick Run**: 给出 `python main.py ...` 的一键运行命令。
4. **Data Requirement**: 明确该 Paper 的解释模块需要哪些 Metadata 字段 (如 `fs`, `channel_names`)。
5. **Submodule Note**: 显式说明 "如何对比 Submodule 中的原始代码与配置"。

```

---

## v1.2 建议补强点（直接加到原计划里）

### 0.3 术语与命名口径（避免后面讨论成本）

新增一小节，统一以下名词：

* **paper_id**：稳定索引（只用于 preset 与 docs/paper_hub）
* **preset_id**：配置预设 id（默认 `preset_id == paper_id`）
* **operator_id**：算子注册名（稳定、可版本化）
* **layout**：张量布局（如 `BCL` / `BLC` / `BCHW`），贯穿 operator 的输入输出

并规定：**所有 registry key 都必须是稳定字符串**（不可用类名自动生成）。

---

## 1) 模型算子化：加“布局契约 + 适配器”，把高风险降到可控

你 v1.1 里 Operator 的 ABC 很好，但“千奇百怪输入形状”会在 PR2 爆炸。建议你在 3.1 里再补一个 **LayoutSpec** + **Adapter Operators 必选** 的规范：

### 1.1 LayoutSpec（写成强约束）

每个 operator 必须声明 `input_layout` / `output_layout`，GraphModel 在拼接时自动插入适配器（或报错）。

一个简洁的可拷贝公式（说明“布局变换是可审计的映射”）：

$$
x^{(out)}=\Pi_{\text{layout}*{in}\rightarrow \text{layout}*{out}}\left(x^{(in)}\right)
$$

> 实现上就是 `Permute/Reshape` 等 AdapterOperator，且要落盘到 manifest（可追溯）。

### 1.2 GraphConfigurableModel = 调度器；TSPN_UXFD = preset 模板

把你的“一个 TSPN”表述改成：

* **GraphConfigurableModel/SequentialPipelineModel**：主容器（负责解析 operator_graph、插 adapter、传 meta、收集 hooks）
* **TSPN_UXFD**：一个“常用 preset 的别名/模板”（不是唯一承载所有逻辑的模型类）

这样 PR1 不用追求“完美 TSPN”，更稳。

---

## 2) 配置映射：把“代码映射”进一步降到“数据驱动 preset + override”

你 v1.1 已经引入 `configs/reference/uxfd_min/`，建议再补一个更省维护的规范：

### 2.1 presets 必须支持“继承/覆盖”（不用写硬代码映射）

你可以在文档里规定：

* `configs/base/base_uxfd.yaml`：公共缺省（seed、日志、输出、默认 trainer.extensions 结构）
* `configs/presets/uxfd/<paper_id>.yaml`：只写差异（operator_graph/operator_args + metadata 要求 + explain 默认）
* `configs/reference/uxfd_min/<paper_id>.yaml`：= base + preset 的 resolved 结果（CI 用）

并强制“镜像配置”可由脚本生成/校验（避免手写漂移）：

* `scripts/gen_uxfd_min_configs.py`：读取 base + preset，输出到 `configs/reference/uxfd_min/`

> 这样 PR2 的工作量会从“写 7 个复杂 yaml + 保持一致”变成“写 7 个 preset 差异 + 自动生成镜像”。

---

## 3) metadata 穿透：给一个“最小侵入实现路径”，避免改 trainer/task

你写的 5.2 优先级很对，但落地时容易卡在“trainer 只认识 (x,y)”上。建议你把 3.3/PR3 里补一个 **“不破坏 train_step 的实现策略”**：

### 3.1 统一 Batch 解析函数（只改一处）

新增一个工具函数（放 `src/utils/batch.py` 或 data_factory utils），全仓只用它解包：

* 输入：`batch`（允许 `(x,y)` / `(x,y,meta)` / dict）
* 输出：`x, y, meta`（meta 可为空 dict）

然后：

* trainer 不改接口：仍然拿到 `batch`
* trainer 内部第一行调用 `unpack_batch(batch)`

这样 explain_factory 要 meta 时，永远能从同一路径拿到。

### 3.2 default_metadata 的“降级策略”也要落盘

你已要求 `data_metadata_snapshot.json`，建议再补一条：
如果走 fallback/default，必须在 snapshot 里写：

* `meta_source: "dataset" | "batch" | "default"`
* `degraded: true/false`
* `missing_keys: [...]`

解释门控 reasons 与 snapshot 要能互相对齐（否则排障困难）。

---

## 4) explain 可审计：把 HookSpec 变成“稳定 schema + 统一抓取点”

你 v1.1 的 HookSpec 很好，建议补一条“hook 的抓取位置”：

* **hook 只能由 GraphModel 统一收集**（不要每个 operator 自己写文件）
* GraphModel 负责将 hook 写入一个 in-memory `HookStore`，explain_factory 从 HookStore 读

并固定一个最小 HookStore 结构（便于扩展）：

* `hooks/<hook_name>.npy`（或 pt）
* `hooks/index.json`（hook_name → semantic/axes/layout）

---

## 5) 依赖地狱：加“依赖政策”段落（非常建议写死）

在 0.2 增加两条硬约束：

* **核心依赖不得因某篇 paper 变更**：主仓库 `requirements.txt` 是上限
* explain/agent 的额外依赖只能以 **extras** 形式存在（例如 `requirements-explain.txt`），并且代码中必须 **optional import + 友好降级**

并在 PR3 验收里加：
“在仅安装 core requirements 的环境中，explain.enable=true 必须输出 eligibility.json（即使解释器不可用，也要给原因），不得 ImportError 崩溃”。

---

## 6) 证据链产物：给 manifest / eligibility / snapshot 一个最小字段集（避免后面格式漂移）

你已规定必须产出三类 json，建议把最小 schema 写成表格（计划里短短一段即可）：

* `artifacts/manifest.json` 最小字段：

  * `paper_id`, `run_id`, `config_snapshot`, `metrics`, `figures_dir`
  * `data_metadata_snapshot`, `eligibility`, `explain_summary`（若有）
  * `distilled_dir`（若 agent todo 生成）

* `artifacts/explain/eligibility.json` 最小字段：

  * `ok: bool`
  * `reasons: [{code,message,suggestion,missing_keys?}]`

* `artifacts/data_metadata_snapshot.json` 最小字段：

  * `meta_source`, `degraded`, `missing_keys`
  * `sampling_rate?`, `sensor?`, `domain?`, `operating_condition?`, `transform?`

这样 paper submodule 的脚本不会因为格式变动反复改。

---

## 7) PR 拆分的“微调版”（更抗风险）

你 PR0–PR5 很合理，我建议只做两个小调整：

### PR1（骨架）里明确引入 GraphModel + Layout/Adapter

PR1 改动点建议写成：

* `GraphConfigurableModel`（或 Sequential）
* `OperatorRegistry`
* `LayoutSpec + AdapterOperators`
* `HookStore`（空实现也行）
* `scripts/config_inspect.py`（若不存在）

验收补一条：

* 任意两个 operator layout 不一致时：自动插 adapter 或给出明确报错（并在日志中打印 layout chain）

### PR2（映射）里把“镜像配置”改成“由 preset 自动生成并校验”

PR2 增加：

* `scripts/gen_uxfd_min_configs.py`
* CI 检查：`configs/reference/uxfd_min/` 必须与 base+preset 的生成结果一致（防手改漂移）

---

## 8) Prompt Pack 再“取长补短”的升级（你现在的 Prompt 已很强）

你给的 Prompt 1/2/3 很适合推进，我建议加 3 个强约束句子，让输出更贴合 v1.2：

* **Prompt 1** 增加：必须给出 LayoutSpec/AdapterOperators 方案；GraphModel 为主、TSPN 为 preset
* **Prompt 2** 增加：必须实现 `unpack_batch` 的 meta 穿透策略（不改 task/trainer 的接口）
* **Prompt 3** 增加：在 core requirements 环境下必须可运行（解释不可用也要产出 eligibility.json）

---

# 你可以直接贴回计划的“最小补丁段落”（建议放在 3.1 / 3.3 / 0.2）

**放到 3.1 后面：**

* Operator 必须声明 `input_layout/output_layout`；GraphModel 将在拼接时自动插入 AdapterOperator 或抛出可审计错误。
* GraphModel 统一收集 HookStore；explain_factory 只读 HookStore + data metadata，不允许 operator 自行落盘解释产物。

**放到 0.2 后面：**

* 主仓库 requirements 为硬上限；paper 冲突依赖只能外置环境。
* explain/agent 的额外依赖必须 optional import，且在 core 环境中也要能产出 eligibility/snapshot（降级可审计）。

---



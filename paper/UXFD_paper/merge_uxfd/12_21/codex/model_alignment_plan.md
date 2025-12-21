# 模型整合“尽量不偏离上游 TSPN/Signal_processing”的理由与优化计划

本文面向：你担心当前 `init_plan.md` 的“GraphConfigurableModel/OperatorRegistry/Adapter/Hooks”范式离
`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/` 的实现（`Signal_processing.py`、
`Signal_processing_2D.py`、`TSPN.py`）太远。

结论先说：**我们不改动上游 TSPN 的核心计算范式**（SignalProcessingLayer → FeatureExtractorlayer →
Classifier），只是在 PHM‑Vibench 的配置系统与解释/证据链需求下，给它加“可配置、可审计、可扩展”的外壳。

---

## 1) 上游实现到底是什么范式（关键结构不变）

对齐上游 `TSPN.py` 的实际结构（概念上完全固定）：
- **SignalProcessingLayer**：
  - 输入 `x: (B,L,C)`；先 `InstanceNorm1d`；再 `Linear` 做 channel mixing；
  - 再把通道按 `module_num` 分块，分别送入一组 `signal_processing_modules`（例如 `FFT/HT/WF/I`）；
  - 再拼回；可选 skip connection（也是一个 `Linear`）。
- **FeatureExtractorlayer**：
  - 输入仍是 `x: (B,L,C)`；
  - 先 `InstanceNorm1d`；再 `Linear`；再并行跑多个 `feature_extractor_modules`（Mean/Std/…）；
  - concat 后得到 `(B, feature_dim)`；再 `CustomBatchNorm`。
- **Classifier**：MLP 头输出 logits。

这与主仓库现有 `src/model_factory/X_model/TSPN.py` 的范式一致：它已经是“**配置驱动组网**”
（`args.signal_processing_configs` + `args.feature_extractor_configs` → 构建 modules → forward）。

因此我们要做的是：**把这个已存在的“配置驱动组网”变成 vibench 可维护、可扩展、可解释的统一入口**。

---

## 2) 为什么需要“看起来更工程化”的外壳（但并不改变 TSPN 本体）

PHM‑Vibench 的硬约束（config-first + registry + 可追溯）要求我们补齐三件上游默认缺失的能力：

### 2.1 统一配置口径（否则 7 篇 paper 会变成 7 套脚本）

上游项目里常见“每篇 paper 一套 config.py/parse_network.py/脚本参数”的形态，不适合合并进 vibench 主流程。

我们要做的是：
- 仍然用 `signal_processing_configs` / `feature_extractor_configs` 这套上游最自然的组网方式（**不偏离**）
- 但把它放入 vibench 的 5‑block YAML（`environment/data/model/task/trainer`）里，并保证 `--override` 可控
- 再为“跨 paper 的差异”增加一个更稳定的索引：`paper_id + preset_version`（只做默认 preset，不强迫）

换句话说：**不是用新范式替换 TSPN，而是让 TSPN 能被 vibench 的 YAML 工程化管理**。

### 2.2 布局/形状契约（LayoutSpec/Adapter）是为“2D 模块接入”准备的

上游 `Signal_processing_2D.py` 的核心输出是时频表示，常见形状是：
- `SBCT_NOP.forward()` 输出 `(...)-> (B, T, F, C)`（代码里 `fuse_TFR` 最终 `b t f c`）

而 TSPN 的 1D 主链路是 `(B, L, C)`。

如果我们**不显式声明 layout**，在“融合/2D 接入”阶段一定会出现：
- 某个模块悄悄把 `BLC` 当成 `BCL` 用
- 某个模块输出 `B T F C`，下游还按 `B L C` 解释

因此我们引入：
- `layout`：把 `BLC / BCL / BTFC / BCHW` 这种“隐性假设”显式化
- `AdapterOperators`：只做 `permute/reshape` 的小模块（例如 `BLC <-> BCL`，`BTFC <-> BLC` 的 flatten/pool）

注意：**Adapter 并不改变上游算法**，只是把“张量布局假设”变成可审计的显式步骤。

### 2.3 HookStore 是为 explain_factory 服务（不改 forward 的数学逻辑）

上游的可解释性（例如 parse_network 的可视化）依赖读取 layer 参数/中间量。
在 vibench 中，我们需要把这些“中间量”：
- 统一收集（不能散落在每个 operator 里各自落盘）
- 统一落盘（配合 `eligibility.json`、`manifest.json`）
- 统一汇总（collect/report）

所以 HookStore 的职责是：
- 在一次 forward 中收集：`SignalProcessingLayer.weight_connection` 的 softmax 权重、router 权重、2D 时频图等
- explain_factory 只读 HookStore + data metadata，生成解释产物

HookStore 仍然**不改变模型的 forward 输出**，只是记录“解释需要的证据”。

---

## 3) “如何把上游 1D/2D 模块整合进同一个模型入口”（不偏离版）

这里给出一个“尽量贴近上游”的整合策略：**TSPN‑Compatible 容器**（不是泛化 Graph 语言）。

### 3.1 统一入口类：`TSPN_UXFD`（保持 TSPN 的层级结构）

我们建议最终公开给 vibench registry 的模型类是 `TSPN_UXFD`，它的内部结构仍是：
1) `signal_processing_layers`：复用上游 `SignalProcessingLayer`
2) `feature_extractor_layers`：复用上游 `FeatureExtractorlayer`
3) `clf`：复用/轻微增强上游 `Classifier`

新增部分只允许是“可选插件”：
- 可选 `signal_processing_2d_layers`（来自 `Signal_processing_2D.py`）
- 可选 `fusion`（来自 `Fusion1D2D*.py`，属于 paper 1 的扩展位）
- HookStore（记录解释证据）

### 3.2 2D 模块如何接入（两条最稳路线，任选其一）

**路线 A（最保守）：2D 只做“解释/可视化”分支，不参与分类主链路**
- 主链路仍是 `(B,L,C)` 跑完得到 logits
- 并行分支用 `Signal_processing_2D` 生成 `(B,T,F,C)` 的时频图，写入 HookStore
- explain_factory 用 metadata（采样率/窗长/步长）把图画出来，并做频带重要性等解释

优点：几乎不影响训练稳定性；最接近上游；风险最低。

**路线 B（融合路线）：2D 进入融合层，再参与分类**
- 1D 主链路产生 `z_1d`
- 2D 分支产生 `z_2d`（先把 `BTFC` 通过 pooling/flatten 转成 `BLC` 或 `BC`）
- 用 `Fusion1D2D`（上游已有）融合，再进 head

优点：覆盖 paper 1 的“1D‑2D fusion”；缺点：需要更严格的 layout/adapters 与训练调参。

建议执行顺序：先 A（先闭环 explain），再 B（再做 fusion paper 的主链路）。

---

## 4) 对 `init_plan.md` 的“模型部分”优化建议（不偏离上游的表述方式）

为了减少“看起来很新”的心理落差，建议把模型部分的语言从：
- “GraphConfigurableModel（泛化图容器）”
调整为：
- “TSPN‑Compatible 容器（仍是 SignalProcessingLayer/FeatureExtractorlayer/Classifier，只是配置更可追溯）”

并在文档里明确：
- `OperatorRegistry` 本质就是上游的 `ALL_SP/ALL_FE`（或同等字典）的**稳定命名+注册**
- `operator_graph` 的 stage 化只是为了兼容 2D 与 explain/collect；对纯 1D TSPN 完全等价于
  `signal_processing_configs + feature_extractor_configs`

---

## 5) 具体优化计划（写给“要动手改仓库的人”）

> 目标：让实现“看起来就是上游 TSPN 的代码”，只是多了 vibench 的配置/解释闭环能力。

### Step 1：把模型实现定位为“兼容 TSPN 的增强版”
- 公开模型名：`TSPN_UXFD`
- 内部复用：主仓库 `src/model_factory/X_model/TSPN.py` 的结构（或直接 copy minimal diff）
- 禁止：把训练/任务逻辑写进模型（仍由 task/trainer 管）

### Step 2：配置层优先兼容上游 key（减少迁移成本）
在 `model` block 支持两种写法（同时支持）：
1) 上游式：
   - `signal_processing_configs: {layer1: [...], layer2: [...]}`（与上游一致）
   - `feature_extractor_configs: [...]`
2) stage 化（为了 2D/fusion/explain 汇总）：
   - `operator_graph: {preprocess_1d: [...], preprocess_2d: [...], ...}`

并规定：两者同时出现时，以 `operator_graph` 为准（可审计）。

### Step 3：先做 2D 的“解释分支”（路线 A）
- 从 batch/dataset 取到 `sampling_rate/window_length/stride`
- 让 `Signal_processing_2D` 在 forward 中可选执行，并把 `BTFC` 写入 HookStore
- explain_factory 先支持：
  - 时频图可视化
  - 频带/时间片 occlusion 的 faithfulness（需要 sampling_rate）

### Step 4：再做 2D 的“融合主链路”（路线 B，paper 1）
- 定义一个最小 `FusionAdapter`：
  - `BTFC -> BC`（global pooling）
  - 或 `BTFC -> BLC`（flatten 后线性投影）
- 复用上游 `Fusion1D2D`（尽量不改）作为 fusion operator

### Step 5：解释证据链落盘（HookStore → explain_factory → manifest）
- 由模型/GraphModel 统一收集 hooks（禁止 operator 自己落盘）
- explain_factory 固定落盘：
  - `artifacts/data_metadata_snapshot.json`
  - `artifacts/explain/eligibility.json`
  - `artifacts/explain/summary.json`
- collect/report 固定落盘：
  - `artifacts/manifest.json`

---

## 6) 你可以怎么验证“我们没有偏离上游”

对同一份 `signal_processing_configs + feature_extractor_configs`：
- 用上游 `Transparent_Signal_Processing_Network` 跑 forward
- 用主仓库 `TSPN_UXFD`（禁用 2D 分支/禁用 explain）跑 forward

如果：
- 输入输出 shape 一致
- 数值在允许误差范围内（考虑初始化/seed 差异）

则证明：**整合没有偏离上游范式**。


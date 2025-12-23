# 信号处理 / 特征提取 / 逻辑推理（含融合/路由）算子库补全计划（12/23 待确认）

目标：把 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/` 中与 UXFD 相关的三类“可插拔算子”
（以及 TSPN 里实际依赖的“融合/路由算子”）在 **不偏离上游范式** 的前提下，整理并迁移到 vibench 主仓库的
`src/model_factory/X_model/UXFD/**`，使其可以被 `TSPN_UXFD`（以及后续 paper configs）稳定复用。

未完成 TODO 汇总（从 12_22 迁移的 backlog）：`paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`

约束复述（必须遵守）：
- 不改变 vibench 单入口：`python main.py --config ...`
- 不新增 YAML 第 6 个一级 block；开关放 `trainer.extensions.*`
- 模型范式尽量贴近上游：`Signal_processing.py` / `Signal_processing_2D.py` / `TSPN.py`
- 任何新增/调整目录必须同步更新该目录 `README.md`
- 主仓库 demos/tests 不依赖任何 paper submodule 初始化

---

## 优先级（review 更新：WP0/WP1 是绝对瓶颈）

当前进度属于“骨架已立，血肉未填”。后续工作的顺序必须从“文档规划”切换为“配置驱动的代码移植与验证”：

- **WP0（Submodules）必须先做**：没有 `paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml`，任何通用模块移植都无法在真实 Context 中验证。
- **WP1（Logic Porting / Operators Porting）是最大风险点**：上游 `Fusion1D2D.py`、`FuzzyLogic.py` 等逻辑复杂。
  迁移策略优先采用 **Copy + Adapter（先跑通，再优化）**，避免一次性重构导致无法收敛。

---

## 0) 当前现状（Gap）

vibench 已存在：
- `src/model_factory/X_model/Signal_processing.py`（上游 1D SP 的一部分/变体）
- `src/model_factory/X_model/Feature_extract.py`（上游 `Feature_extract.py` 的一部分/变体）
- `src/model_factory/X_model/TSPN.py` + `src/model_factory/X_model/TSPN_UXFD.py`（稳定别名）
- `src/model_factory/X_model/UXFD/signal_processing_2d/stft_tfr.py`（**非上游**，仅为纯 PyTorch 最小 STFT）

尚未系统化整合（本计划要补齐）：
- 上游 `Signal_processing_2D.py`（2D 时频算子族、输出布局 BTFC 等）
- 上游 `Feature_extract.py` 全量统计特征算子（与 `ALL_FE` 映射对齐）
- 上游 `Logic_inference.py` 逻辑算子（以及“CPU 安全”的实现方式）
- 上游融合/路由算子族（TSPN 常用的 glue）：
  - `Fusion1D2D.py` / `Fusion1D2D_simple.py`
  - `operator_attention.py` / `OperatorAttention_*`
  - `MoE.py` / `MoE_*`
  - `FuzzyLogic*.py`（如果上游把模糊逻辑做成单独模块，也并入逻辑推理库）
- 统一的算子注册/命名（稳定 operator_id）、输入输出布局约束（BLC/BCL/BTFC）

---

## 1) 目标目录结构（vibench 主仓库）

在 `src/model_factory/X_model/UXFD/` 内补全三块：

1) `signal_processing_1d/`
2) `feature_extractor/`
3) `logic_inference/`
4) `fusion_routing/`（1D↔2D 融合、OperatorAttention、MoE 等）

并在 `src/model_factory/X_model/UXFD/tspn/` 内提供“组装层面的 glue”（保持上游层级结构）：
- TSPN 仍是 `SignalProcessingLayer → FeatureExtractorlayer → Classifier`
- 融合/路由与逻辑推理都作为 **可选** 分支/头部（由配置决定是否参与前向）

补充（按你的要求更新）：
- `signal_processing_1d/` 不仅覆盖“1D 信号处理算子”，也要包含 **不同数据集输入形状的归一化/适配**：
  - 支持 1D 数据集（时域序列），也支持 2D 数据集（已给出时频图/谱图）的“进入 TSPN 的统一入口”
  - 目标是：无论数据集怎样给 `x`，进入后续模块前都能被规范化为本文约定的 `BLC` 或 `BTFC`

---

## 2) 统一接口与张量布局（不偏离上游，但把隐含假设显式化）

### 2.1 统一默认布局

- 1D 信号（时域）：默认输入 `x1d: (B, L, C)`（BLC，与 vibench dataset 输出一致）
- 1D SP 内部如需 `BCL`：允许在算子内部临时 `permute`，但 **对外接口仍保持 BLC**
- 2D 时频（时频图/谱图）：统一为 `x2d: (B, T, F, C)`（BTFC）
- 2D 特征提取约束（关键）：`T` 与 `F` 需要分别建模/提取特征，不能在 SP2D 后立即把 `T,F` 全部压成一个向量；
  后续 FE 必须显式提供 “T-branch” 与 “F-branch” 两条路径（见 3.2）

### 2.2 复杂值（FFT/STFT）的处理约定

上游可能出现 `torch.fft` 输出 complex tensor：
- 方案 A（推荐，稳定）：只输出幅值（real），并在 README 明确该算子是 magnitude 版本
- 方案 B（保留信息）：把 complex 拆成 `real/imag` 两个通道（C 翻倍），并显式标注输出通道规则

本计划默认采用方案 A，除非某篇 paper 明确需要 complex 信息。

补充（按你的要求更新）：
- 所有 `torch.fft`（无论在 SP2D 还是 FE 的轴向变换里）统一采用 **方案 A：complex → magnitude（real）**

---

## 3) 具体迁移内容（按 3 类算子）

### 3.1 信号处理算子库（Signal Processing）

来源：
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/Signal_processing.py`
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/Signal_processing_2D.py`

落位：
- `src/model_factory/X_model/UXFD/signal_processing_1d/`：整理 1D SP 算子（按上游命名/实现）
- `src/model_factory/X_model/UXFD/signal_processing_2d/`：移植上游 2D 算子族（替换当前临时 STFT 或并存）

补充：数据集输入形状适配（放在 `signal_processing_1d/`，作为“统一入口”）
- 目标：把不同数据集/不同 reader 输出的 `x` 规范化为：
  - 1D：`BLC`
  - 2D：`BTFC`
- 需要覆盖的常见变体（以 best-effort adapter 形式实现；不做破坏性假设）：
  - 1D：`(B, C, L)` / `(B, L)` / `(B, L, 1)`
  - 2D：`(B, C, T, F)` / `(B, T, F)` / `(B, T, F, 1)`
- 适配原则：
  - adapter 只做 `permute/unsqueeze` 等“形状归一化”，不隐式改变语义（不做强行插值/重采样）
  - 若无法判断语义（例如 `(B, 64, 1024)` 到底是 `C,L` 还是 `T,F`），输出可读错误信息（后续可复用到 eligibility/report）

迁移策略（最小 diff）：
1) 先原样移植算子类（保留数学逻辑），只做“vibench 安全性修正”：
   - device 处理（禁止 import-time `.cuda()`）
   - 随机数/初始化可复现（必要时）
2) 为每个算子写最小 README 说明：
   - 输入输出形状（BLC/BTFC）
   - 需要的 metadata keys（如采样率/窗长/stride）
3) 给每个算子分配稳定 `operator_id`（见 4.1）

必须补齐（P0）：
- 上游 2D 主算子（至少 1 个完整可用路径）
- 1D2D 之间的 shape adapter（BTFC → BLC 或 BC），用于 fusion/分类

### 3.2 特征提取算子库（Feature Extraction）

来源：
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/Feature_extract.py`
- （可选补充）`FeatureExtractor.py` 若存在不同实现

落位：
- `src/model_factory/X_model/UXFD/feature_extractor/`

迁移策略：
1) 以 `FeatureExtractionBase` + `FeatureExtractionModuleDict` 为核心接口（与上游一致）
2) 补齐上游统计特征集合（Mean/Std/Var/Entropy/Max/Min/AbsMean/Kurtosis/RMS/...）
3) 明确输入是 `x: (B, C, L)` 还是 `(B, L, C)`：
   - vibench/TSPN 链路输入是 `BLC`，进入 FE 前统一转成 `BCL`
   - FE 输出维度规则在 README 固定（例如 `(B, C, 1)`，最终 concat 为 `(B, C*F)`）

补充：2D 时频的 “T/F 双分支” 特征变换（按你的要求更新）
- 输入：`x2d: (B, T, F, C)`（BTFC）
- 目标：分别抽取时间轴特征与频率轴特征，并在 classifier 前融合（concat 或可选 fusion）
- 推荐的“两层特征变换”最小实现（保持可解释/可追溯）：
  1) 轴向投影（Axis Projection）：
     - `T-branch`: 沿 `F` 做 reduce（如 mean/max/attention pool）得到 `x_t: (B, T, C)`
     - `F-branch`: 沿 `T` 做 reduce 得到 `x_f: (B, F, C)`
  2) 轴向频域变换（Axis FFT Transform, magnitude only）：
     - 对 `x_t` 在 `T` 维做 `torch.fft.rfft(..., dim=1)`，取 `abs()` 得到 `x_t_fft`（real）
     - 对 `x_f` 在 `F` 维做 `torch.fft.rfft(..., dim=1)`，取 `abs()` 得到 `x_f_fft`（real）
     - 备注：这一步遵循 2.2 的 **方案 A（magnitude）**，避免 complex 在下游扩散
- 下游统计特征（FE）对接方式：
  - 将 `x_t_fft` / `x_f_fft` 都转换为统一的 1D 格式 `BLC` 或 `BCL`（推荐 `BLC`），复用同一套统计 FE
  - 最终 `feature = concat([FE(x_t_fft), FE(x_f_fft)], dim=-1)`，并在 manifest/explain 中记录 branch 信息

### 3.3 逻辑推理算子库（Logic Inference / Neuro-Symbolic）

来源：
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/Logic_inference.py`

落位：
- `src/model_factory/X_model/UXFD/logic_inference/`

关键修正点（必须做，P0）：
- 上游文件目前存在 import-time `ONE = torch.Tensor([1]).cuda()` / `ZERO = ...cuda()`：
  - 这会在 CPU 环境直接崩溃，且违反 vibench “设备可配置”原则
  - 迁移时必须改成 **运行时按输入张量 device/dtype 创建**：
    - `one = x.new_tensor(1.0)` / `zero = x.new_tensor(0.0)`

逻辑算子集合（先对齐上游）：
- implication / equivalence / negation
- weak_conjunction / weak_disjunction
- strong_conjunction / strong_disjunction
- generalized_softmax / generalized_softmin（带温度/alpha）

与模型的对接方式（保持不偏离上游 TSPN）：
- 逻辑推理默认不强制进入 `TSPN_UXFD` 主链路
- 通过配置开启时，逻辑推理接受 FE 输出（或额外可配置的特征输入），输出：
  - 方案 A：作为 logits 的 additive term（与 classifier 输出相加）
  - 方案 B：作为辅助 loss / 约束项（更偏 task/trainer，但先不做，避免越界）

本阶段默认先做方案 A（可解释且侵入性低），并把方案 B 留作后续 TODO。

---

### 3.4 融合/路由算子库（Fusion + Routing）

来源（上游同目录下常见文件名）：
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/Fusion1D2D.py`
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/Fusion1D2D_simple.py`
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/operator_attention.py`
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/OperatorAttention_*.py`
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/MoE.py` / `MoE_*`

落位：
- `src/model_factory/X_model/UXFD/fusion_routing/`

设计理由（为什么放这里、且不偏离上游范式）：
- 在上游 UXFD 的 `TSPN*.py` 变体中，Fusion/Attention/MoE 通常是 “SP/FE 之后、Classifier 之前” 的中间模块；
- 它们是“可插拔算子”的一部分：可被不同 paper 组合/替换，但不会改变数据工厂/任务/训练器的主骨架；
- 因此归属应在 `model`（而不是 task/trainer），并由 `model:` 配置决定是否启用与如何组装。

迁移策略（最小 diff）：
1) 先原样移植核心 forward 逻辑，优先保证与上游 shape 约定一致；
2) 统一“输入输出形状”在 README 中显式声明，并在代码中做最少量 adapter：
   - 1D 输入：`x1d`（BLC 或 BCL，按上游实际）
   - 2D 输入：`x2d`（BTFC 或 BCTF，按上游实际）
3) 对 `OperatorAttention_*` / `MoE_*` 这类“多版本”文件：
   - 先选 1 个上游默认版本作为 P0（例如 enhanced 或 simple）
   - 其余版本先迁移为可选实现（P1），不进入默认路径，避免一次性改动过大

P0 交付：
- `Fusion1D2D`（至少一个可用版本）
- `OperatorAttention`（至少一个可用版本）
- `MoE`（至少一个可用版本）
- 对接点：在 `TSPN_UXFD` 的组装层里加可选配置（默认关闭，保证不影响现有 smoke）

---

## 4) 配置与注册（让 paper configs 可写）

### 4.1 稳定命名（operator_id）

给三类算子统一命名空间（示例）：
- `SP1D/FFT`, `SP1D/HT`, `SP1D/WF`, ...
- `SP2D/SBCT_NOP`, `SP2D/STFT`, ...
- `FE/Mean`, `FE/Std`, `FE/Kurtosis`, ...
- `LOGIC/Implication`, `LOGIC/Negation`, ...

要求：
- `operator_id` 不使用类名自动生成（避免重构导致漂移）
- 在 README 和（后续）explain artifacts 中用 `operator_id` 做追溯键

### 4.2 vibench YAML 约定（保持 5-block）

本阶段只在 `model:` block 内扩展字段（不新增顶层 block）：

```yaml
model:
  type: "X_model"
  name: "TSPN_UXFD"

  signal_processing_configs: {...}   # 保持现有 TSPN 方式
  feature_extractor_configs: [...]   # 保持现有 TSPN 方式

  # 新增（可选）
  signal_processing_2d: {...}        # 2D 算子/参数（仅当某 paper 需要）
  feature_extractor_2d:             # 2D(TF) 双分支特征（可选；默认关闭）
    enable: false
    branches:
      time:
        reduce: "mean_F"            # F 维投影
        fft: true                  # rfft over T, magnitude
      freq:
        reduce: "mean_T"            # T 维投影
        fft: true                  # rfft over F, magnitude
  fusion_routing:                   # 1D↔2D 融合、attention、MoE（可选）
    enable: false
    fusion: {name: "Fusion1D2D"}    # 或 simple 版本
    attention: {enable: false, name: "OperatorAttention"}
    moe: {enable: false, name: "MoE"}
  logic_inference:                  # 逻辑推理开关与算子组合
    enable: false
    operators: ["LOGIC/Implication", "LOGIC/Negation"]
    alpha: 20
```

说明：
- `trainer.extensions.explain/report/collect/agent` 仍在 `trainer` 下（不搬到 task）
- “把逻辑推理放 task 里”不是本阶段目标；逻辑推理是模型算子库的一部分，应落在 `model` 配置

---

## 5) 解释与证据链（HookStore/Explain 需要的最小输出）

本阶段只承诺算子库补齐，不强行上线复杂 explain，但需要预留证据点：
- 对 `TSPN` 的 router/weight_connection：保留可读参数名/访问路径（后续 HookStore 采集）
- 对 2D 时频：明确输出张量的 layout/轴语义（时间/频率），便于 explain_factory 做可视化
- 对逻辑推理：记录逻辑算子输入/输出的张量范围与含义（真值域 [0,1] 还是 logits 域）

关键技术建议（review）：TSPN_UXFD 的 HookStore
- `TSPN_UXFD` 目前只是别名，不应直接修改 `src/model_factory/X_model/TSPN.py`（保持其“上游映射的纯净”）。
- 应在 `src/model_factory/X_model/UXFD/tspn/` 下创建一个 **Wrapper（组合或继承）**：
  1) 捕获中间层输出（如 SP 输出、FE 输出、Fusion/Attention 权重、Router weights）。
  2) 写入 `artifacts/explain/hookstore.json`（或注册到 explain_factory 的全局上下文，二选一；优先“写文件”更可审计）。
  3) 不改变原有 forward 的数学计算结果（仅旁路记录）。

---

## 6) 实施步骤（你确认后我将按这个顺序改代码）

### WP0：Submodules（先跑通 1 篇 pilot）

目标：先选 1 篇 paper 作为 Pilot（建议优先选涉及 2D/Fusion 的）跑通全链路，作为后续 WP1 的验证场。

建议 Pilot（默认）：
- `paper/UXFD_paper/1D-2D_fusion_explainable`（天然覆盖 Fusion + 2D 处理）

WP0 交付物（必须在 Pilot submodule 内）：
- `paper/UXFD_paper/<pilot>/configs/vibench/min.yaml`
  - 仅依赖主仓库的通用模块（`X_model`/`TSPN_UXFD`/extensions 等），避免 submodule 自己写新训练代码
  - 通过 `environment.output_dir` 把输出落在 vibench 的 `results/`（或 `save/`）下，确保 `manifest.json` 与 collect 脚本可复用
- `paper/UXFD_paper/<pilot>/VIBENCH.md`
  - 写清：paper 原始入口 → vibench 入口映射（“用哪个 config 跑”）
- `paper/UXFD_paper/<pilot>/configs/vibench/README.md`（若新建目录必须有 README）

WP0 验证方式（Pilot 必须能跑）：
- `python main.py --config paper/UXFD_paper/<pilot>/configs/vibench/min.yaml --override trainer.num_epochs=1`
- `python -m scripts.collect_uxfd_runs --input <output_root> --out_dir reports/`

注意：
- submodule 内文件修改需要在 submodule 仓库里提交（本地 commit 即可），主仓库记录 gitlink 指向该 commit。

### WP1：Operators Porting（Copy + Adapter，先跑通）

Step 1（准备目录，P0）：
- 新建目录（均含 README）：
  - `src/model_factory/X_model/UXFD/feature_extractor/`
  - `src/model_factory/X_model/UXFD/logic_inference/`
  - `src/model_factory/X_model/UXFD/fusion_routing/`

Step 2（优先迁移“Pilot 必需”的算子，按 Copy + Adapter）：
- 先以“能跑通 Pilot”为准，优先级建议：
  1) `Fusion1D2D`（1D↔2D glue）
  2) `Signal_processing_2D`（至少 1 条 BTFC 可用路径）
  3) `feature_extractor_2d`（T/F 双分支 + FFT(magnitude) + 复用统计 FE）
  4) `FuzzyLogic`（若 Pilot 需要；否则进入下一轮）
- 对每个上游文件：先 Copy 过来，再用最小 adapter 适配 vibench 的输入输出布局；不要先做“完美重构”。

Step 3（迁移 FE 全量算子，P1）：
- 移植 `Feature_extract.py` 全量统计特征算子
- 统一 `BLC→BCL` 的进出接口规则，保证所有 paper configs 可复用

Step 4（迁移 LogicInference，P1 但高风险必须早做）：
- 移植 `Logic_inference.py` 并修复 import-time cuda 常量（CPU 安全）
- 默认关闭，通过 `model.logic_inference.enable` 打开

### WP2：HookStore Wrapper（Explainability 的关键前置）

Step 5（新增 Wrapper，不改 TSPN.py）：
- 在 `src/model_factory/X_model/UXFD/tspn/` 新增 wrapper（组合/继承）：
  - 捕获 `features / attention_weights / router_weights / fusion_weights` 等
  - best-effort 写 `artifacts/explain/hookstore.json`（不影响训练）
- 在 `src/model_factory/model_registry.csv` 增加一个新模型条目（例如 `TSPN_UXFD_HOOKED`），让 paper config 可选用

### WP3：测试与最小验证（新增代码必须有测试）

Step 6（新增单测：collect 脚本稳定性）：
- 为 `scripts/collect_uxfd_runs.py` 增加一个最小单元测试：
  - 在临时目录构造 `run/artifacts/manifest.json`
  - 调用 `collect_manifests()` 或 `main()` 生成 CSV
  - 断言 CSV 列与关键字段（`run_dir/metrics_path/metric/*`）存在且值正确

Step 7（回归验证）：
- `python main.py --config paper/UXFD_paper/<pilot>/configs/vibench/min.yaml --override trainer.num_epochs=1`
- `python -m pytest test/`

---

## 7) Definition of Done（本阶段）

当满足以下条件，视为“算子库补全（第一版）”完成：
- WP0：至少 1 个 pilot paper submodule 的 `min.yaml` 能端到端跑通并产出 `artifacts/manifest.json`
- `signal_processing_1d/feature_extractor/logic_inference/signal_processing_2d/fusion_routing` 五个目录齐全且有 README
- 逻辑推理库在 CPU 环境可 import（无 `.cuda()` import-time 依赖）
- `TSPN_UXFD` 能在默认关闭逻辑推理/2D 时频时保持现有行为不变
- `feature_extractor_2d` 提供 BTFC 的 `T/F` 双分支特征提取最小闭环（含 `torch.fft` magnitude）
- `scripts/collect_uxfd_runs.py` 有最小单元测试覆盖（防止证据链工具回归）

---

## 8) 已确认的默认选择（按你的回复更新）

- `torch.fft` complex 处理：统一 **方案 A（magnitude）**
- 逻辑推理对接：默认 **logits additive term（方案 A）**
- 配置归属：`logic_inference` / `fusion_routing` / `feature_extractor_2d` 都放在 `model:` 下（默认关闭）

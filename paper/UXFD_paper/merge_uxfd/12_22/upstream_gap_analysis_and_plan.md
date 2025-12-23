# `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis` 未整合项分析 + vibench 集成计划（12/22）

目标：回答“Unified_X_fault_diagnosis 里还有哪些没有整合到 vibench 中”，并给出下一步可执行的集成计划。

对照基线：
- vibench 计划 SSOT：`paper/UXFD_paper/merge_uxfd/12_18temp/codex/final_plan.md`
- 当前状态复盘：`paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`

## 1) 结论概览（哪些没整合）

Unified_X_fault_diagnosis 目前未整合到 vibench 的内容主要分 4 类：

1) **UXFD 模型组件**（`Unified_X_fault_diagnosis/model/*.py`）
   - 2D 时频：`Signal_processing_2D.py`
   - 1D2D 融合：`Fusion1D2D.py`, `Fusion1D2D_simple.py`
   - 模糊逻辑：`FuzzyLogic*.py`
   - 算子注意力：`operator_attention.py`, `OperatorAttention_*.py`, `TSPN_OperatorAttention.py`
   - MoE / DEN / 神经符号：`MoE*.py`, `DEN.py`, `Logic_inference.py`, `NNSPN.py`, `TFON.py`
   - TSPN 变体：`TSPN_sparse.py`, `TSPN_KAN.py`, `TSPN_LLM_Enhanced.py`, `TSPN_explainable.py`
   - 支撑工具：`parse_network.py`, `utils.py`, `explainable_base.py`, `llm_explainable_base.py`

2) **对比 baseline 模型集**（`Unified_X_fault_diagnosis/model_collection/*.py`）
   - `Resnet.py`, `Sincnet.py`, `WKN.py`, `EELM.py`, `F_EQL.py`, `Physics_informed_PDN.py`, `TFN.py`, `MCN.py`
   - `CI_GNN.py`（通常依赖 `torch_geometric`，需 optional 或先不注册）
   - `base_explainable.py`（可解释基类）
   - `GradCAM_XFD.py`（已部分以 explainer 形式移植到 vibench 的 `src/explain_factory/explainers/gradcam_xfd.py`）

3) **“uxfd” 框架包**（`Unified_X_fault_diagnosis/uxfd/**`）
   - CLI / paper pipelines / schema/validate / report collector 等
   - vibench 主仓库不会直接复用其入口（vibench 单入口固定），但其中的 **schema/协议思想** 可以选择性吸收

4) **Explainability 大包**（`Unified_X_fault_diagnosis/explainability/**`）与 **trainer**（`Unified_X_fault_diagnosis/trainer/**`）
   - explainability 包含 LLM、知识图谱、评估、对话等；trainer 是上游训练器
   - vibench 侧只建议“挑可复用的 explain 方法/协议”，训练逻辑仍由 vibench 的 Lightning 管

## 2) 已整合/部分整合项（避免重复劳动）

已在 vibench 主仓库出现（但不一定完全等价于上游版本）：
- `Unified_X_fault_diagnosis/model/Signal_processing.py`：vibench 有 `src/model_factory/X_model/Signal_processing.py`
- `Unified_X_fault_diagnosis/model/TSPN.py`：vibench 有 `src/model_factory/X_model/TSPN.py`（并提供稳定别名 `TSPN_UXFD`）
- `Unified_X_fault_diagnosis/model/Feature_extract.py`：vibench 有 `src/model_factory/X_model/Feature_extract.py`
- `Unified_X_fault_diagnosis/model_collection/MWA_CNN.py`：vibench 有 `src/model_factory/X_model/MWA_CNN.py`
- `Unified_X_fault_diagnosis/model_collection/GradCAM_XFD.py`：vibench 已移植为 explainer：`src/explain_factory/explainers/gradcam_xfd.py`

已完成的通用闭环能力（与上游无关，但为整合提供基础）：
- manifest/证据链：`src/trainer_factory/extensions/manifest.py`
- manifest→CSV：`scripts/collect_uxfd_runs.py`

## 3) “缺口清单”与建议落位（vibench 的 target paths）

> 原则：通用模块进主仓库 `src/model_factory/X_model/UXFD/**`；paper 个性化 configs 全部进各自 submodule。

### 3.1 UXFD 模型组件（上游 model/）

P0（先做，支撑 7 篇 paper 的公共复用）：
- `Signal_processing_2D.py` → `src/model_factory/X_model/UXFD/signal_processing_2d/`
- `Fusion1D2D*.py` → `src/model_factory/X_model/UXFD/fusion/`
- `FuzzyLogic*.py` → `src/model_factory/X_model/UXFD/fuzzy/`
- `operator_attention.py` + `OperatorAttention_*.py` → `src/model_factory/X_model/UXFD/operator_attention/`

P1（在 P0 稳定后再合入）：
- `TSPN_OperatorAttention.py`（作为 TSPN_UXFD 的一个可选变体/组件）
- `MoE*.py`, `DEN.py`（作为 moe_xfd paper 的通用模块部分；避免把“实验脚本”混入主仓库）
- `Logic_inference.py`（作为 nesy/fuzzy 的推理模块组件；并给出可解释证据输出接口）

P2（可选/研究扩展，先不阻塞）：
- `TFON.py`, `NNSPN.py`, `kan.py`, `TSPN_KAN.py`, `TSPN_sparse.py`, `TSPN_LLM_Enhanced.py`, `TSPN_explainable.py`

不建议直接移植（用 vibench 的配置系统替代）：
- `config.py`, `parse_network.py`：上游组网/脚本逻辑应迁移为 vibench YAML + registry；其中可视化/解释逻辑可拆到 `explain_factory`
- `utils.py`：按需拆分进对应模块，不做“整包复制”

### 3.2 Baselines（上游 model_collection/）

P0（torch-only、容易作为对比表输入）：
- `Resnet.py` → `src/model_factory/X_model/baselines/ResNet_MC.py` + entry `src/model_factory/X_model/BASE_ResNet_MC.py`
- `Sincnet.py` → `.../baselines/SincNet.py` + entry `BASE_SincNet.py`
- `WKN.py` → `.../baselines/WKN.py` + entry `BASE_WKN.py`
- `EELM.py` → `.../baselines/EELM.py` + entry `BASE_EELM.py`
- `F_EQL.py` → `.../baselines/FEQL.py` + entry `BASE_FEQL.py`

P1（可能较重，先跑通最小 forward 再注册）：
- `TFN.py` / `MCN.py` / `Physics_informed_PDN.py`

P2（依赖额外库，默认不注册到 registry，避免破坏主仓库可运行性）：
- `CI_GNN.py`（`torch_geometric` 等）

### 3.3 uxfd 与 explainability 包（上游框架层）

原则：
- 不把上游的 CLI/pipeline 入口迁入 vibench（vibench 单入口固定）
- 只选择性吸收“可复用协议/评估/解释方法”，并保证 `requirements.txt` 不新增硬依赖

建议吸收的最小子集（P1）：
- `uxfd/io/schema_v1.py` / `uxfd/explain/protocol_v1.py`：作为 vibench explain artifacts 的 schema 参考（可转写为 vibench 内部 schema）
- `uxfd/report/collector.py`：作为 collect 逻辑参考（vibench 已有 `scripts/collect_uxfd_runs.py`，可逐步对齐字段）

暂不吸收（P2/可选）：
- `explainability/**` 的 LLM/知识图谱/对话模块（需要先定义 vibench 的 agent/explain 边界与开关）

## 4) 12/22 后续集成计划（按依赖顺序）

### 阶段 A（WP0，阻塞项）：至少落地 1 个 paper submodule
1) 进入 `paper/UXFD_paper/<paper_repo>/`（先选 1 篇 paper）
2) 增加 `paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml`
3) 增加 `paper/UXFD_paper/<paper_id>/VIBENCH.md`
4) 验证：`python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml --override trainer.num_epochs=1`

### 阶段 B（P0 模型组件）：把 2D/fusion/fuzzy/op-att 真正移植到 `UXFD/**`
1) 从上游拷贝并最小改造：
   - tensor layout 统一为 `(B,L,C)` 输入；2D 输出 `(B,T,F,C)`
   - 不允许模块自行落盘（证据链由 explain/report 控制）
2) 每个新增目录补齐 `README.md`
3) 增加最小 smoke 配置（放到某个 paper submodule）验证 forward + 1 epoch

### 阶段 C（TSPN_UXFD 增强壳）：HookStore/registry/layout adapters（不改上游计算范式）
1) 在 `src/model_factory/X_model/UXFD/tspn/` 增加 HookStore 与中间量采集
2) 建立稳定 operator_id（对齐上游 `ALL_SP/ALL_FE`）
3) explain_factory 增加 “router_weights/timefreq/fuzzy_rules” 三类最小解释器（先写 summary.json，再写可视化）

### 阶段 D（Baselines 扩展）：按 torch-only 优先移植并注册
1) 逐个 baseline 建 wrapper + entry module（`BASE_*`）
2) 更新 `src/model_factory/model_registry.csv`
3) 每个 baseline 给出最小 paper submodule config（用于跑表）

### 阶段 E（选择性吸收 uxfd 协议）：与 manifest/explain artifacts 对齐
1) 明确 `artifacts/explain/summary.json` 字段 schema（可参考上游 `protocol_v1.py`）
2) `scripts/collect_uxfd_runs.py` 扩展输出 explain 相关 CSV（可选第二张表）

## 5) Definition of Done（本计划的验收点）

- 至少 1 个 paper submodule 的 `min.yaml` 可跑通（不污染主仓库 configs）
- `src/model_factory/X_model/UXFD/` 中至少包含可用的：
  - 2D 时频算子（上游版或等价实现）
  - 1D2D fusion（上游版）
  - fuzzy（上游版，至少一个）
  - operator attention（上游版，至少一个）
- `TSPN_UXFD` 能写出 HookStore 关键证据（供 explain_factory 消费）
- 解释闭环：
  - `artifacts/manifest.json` + `scripts/collect_uxfd_runs.py` 生成 CSV
  - explain.enable=true 时至少生成 `artifacts/explain/eligibility.json`（已具备）与 `summary.json`（待实现）

# UXFD Suite 整体架构与代码质量分析

日期：2026-03-11

## 1. 总体判断

UXFD Suite 当前已经形成了“1 个主仓库 + 7 个子项目”的统一入口形态，但整体仍处于 **协议先行、代码后补** 的阶段。

从架构上看：

- 主仓库负责统一训练入口、配置加载、运行产物和少量 UXFD 公共模块，[`paper/UXFD_paper/README.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/README.md#L17) 到 [`paper/UXFD_paper/README.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/README.md#L23) 已把这个契约写清楚。
- 7 个子项目都提供了 `configs/vibench/min.yaml` 和 `VIBENCH.md`，说明“统一入口”已经具备最小闭环。
- 但子项目之间的依赖关系主要仍由文档维护，而不是由代码、schema、CI 或 capability registry 强制执行。
- `NSN -> TSPN_UXFD -> UXFD/*` 这条主线已经能承接 fuzzy / logic / operator-attention / sp2d 等能力，[`src/model_factory/X_model/NSN.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/model_factory/X_model/NSN.py#L3) 到 [`src/model_factory/X_model/NSN.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/model_factory/X_model/NSN.py#L14)、[`src/model_factory/X_model/TSPN_UXFD.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/model_factory/X_model/TSPN_UXFD.py#L47) 到 [`src/model_factory/X_model/TSPN_UXFD.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/model_factory/X_model/TSPN_UXFD.py#L152) 证明了这一点。
- 真正的短板不在“有没有入口”，而在 **接口一致性、统一评估协议、可解释性接口收敛，以及占位实现过多**。

一句话概括：**现在的 UXFD Suite 更像“文档协调的系列工程”，还不是“协议驱动的产品级 suite”。**

## 2. 7 个子项目依赖关系与接口一致性

### 2.1 实际依赖关系

当前可归纳为：

1. `Explainable_FD_Toolkit` 是基础设施层，定义了理想中的解释协议，如 `ExplainabilityMethod` 和 `ModelPlugin`，[`paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py#L17) 到 [`paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py#L228)。
2. `1D-2D_fusion_explainable`、`Paper_fuzzy_XFD`、`TII_operator_attention`、`Neuralsymbolic_theory`、`MOE_explainable` 当前主要通过主仓库 `NSN/TSPN_UXFD` 的开关映射接入。
3. `LLM_Explainable_FD_Toolkit` 目前并未真正消费 Toolkit 的统一结构化解释接口，而是先落成一个 `LLM-free distilled artifact` 占位实现，[`paper/UXFD_paper/LLM_Explainable_FD_Toolkit/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/LLM_Explainable_FD_Toolkit/VIBENCH.md#L20) 到 [`paper/UXFD_paper/LLM_Explainable_FD_Toolkit/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/LLM_Explainable_FD_Toolkit/VIBENCH.md#L28)、[`src/trainer_factory/extensions/agent.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/trainer_factory/extensions/agent.py#L21) 到 [`src/trainer_factory/extensions/agent.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/trainer_factory/extensions/agent.py#L89)。

### 2.2 接口一致性的核心问题

最明显的问题是：**文档中的“paper 接口”与代码中的“真实配置接口”并不完全一致。**

例子：

- `1D-2D_fusion_explainable` 的 `VIBENCH.md` 声称当前入口使用 `model.name=TSPN_UXFD` 且直接写 `model.uxfd.enable_sp2d`，[`paper/UXFD_paper/1D-2D_fusion_explainable/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/1D-2D_fusion_explainable/VIBENCH.md#L20) 到 [`paper/UXFD_paper/1D-2D_fusion_explainable/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/1D-2D_fusion_explainable/VIBENCH.md#L38)，但真实 `min.yaml` 用的是 `model.name: NSN` 和 `model.signal_processing_2d.*` 的扁平接口，[`paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml#L48) 到 [`paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml#L71)。
- `Neuralsymbolic_theory` 的文档说启用了 `model.uxfd.logic.enable: true`，[`paper/UXFD_paper/Neuralsymbolic_theory/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Neuralsymbolic_theory/VIBENCH.md#L20) 到 [`paper/UXFD_paper/Neuralsymbolic_theory/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Neuralsymbolic_theory/VIBENCH.md#L25)，但真实配置写的是 `decision_configs.type: logic`，[`paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml#L44) 到 [`paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml#L58)。
- `Paper_fuzzy_XFD` 同样存在文档写 `model.uxfd.fuzzy.*`、实际配置走 `decision_configs.fuzzy` 的情况，[`paper/UXFD_paper/Paper_fuzzy_XFD/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Paper_fuzzy_XFD/VIBENCH.md#L20) 到 [`paper/UXFD_paper/Paper_fuzzy_XFD/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Paper_fuzzy_XFD/VIBENCH.md#L25)、[`paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml#L44) 到 [`paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml#L57)。

这不是单纯的文档小问题，而是会直接影响：

- 子项目作者如何写配置；
- 后续自动生成器如何做 schema 校验；
- 外部用户是否能把 `VIBENCH.md` 当成可信 API 文档。

## 3. 统一评估协议的现状

当前已经具备一些协议化基础：

- `manifest.json` 的 run-level contract 已经存在，[`src/trainer_factory/extensions/manifest.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/trainer_factory/extensions/manifest.py#L32) 到 [`src/trainer_factory/extensions/manifest.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/trainer_factory/extensions/manifest.py#L121)。
- `predictions.npz`、`eligibility.json`、`data_metadata_snapshot.json` 已经被纳入默认产物链，[`src/Pipeline_01_default.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/Pipeline_01_default.py#L137) 到 [`src/Pipeline_01_default.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/Pipeline_01_default.py#L205)、[`src/task_factory/Default_task.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/task_factory/Default_task.py#L175) 到 [`src/task_factory/Default_task.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/task_factory/Default_task.py#L260)。
- `collect_uxfd_runs.py` 已能把 manifest 聚合成 CSV。

但它距离“统一评估协议”还差几步：

- 现在更像 **产物协议**，不是 **评估协议**。
- 它能保证“你产出了什么文件”，不能保证“你用了什么数据切分、什么指标定义、什么 explainability score、什么 latency/resource 口径”。
- `ExplainReady` 目前只校验 metadata key 缺失，不校验方法兼容性、模型可解释能力或数据语义完整性，[`src/explain_factory/eligibility.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/explain_factory/eligibility.py#L35) 到 [`src/explain_factory/eligibility.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/explain_factory/eligibility.py#L64)。
- metrics 体系仍以分类/回归 torchmetrics 为中心，不包含解释质量指标、对话质量指标或跨 paper 可比指标，[`src/task_factory/Components/metrics.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/task_factory/Components/metrics.py#L18) 到 [`src/task_factory/Components/metrics.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/task_factory/Components/metrics.py#L68)。

## 4. 可解释性接口的通用性

这里存在一个明显的“双轨制”：

1. 主仓库 `src/explain_factory/` 当前主要实现的是 metadata snapshot、eligibility 和一个最小 Grad-CAM 包装，[`src/explain_factory/README.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/explain_factory/README.md#L5) 到 [`src/explain_factory/README.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/explain_factory/README.md#L12)、[`src/explain_factory/explainers/gradcam_xfd.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/explain_factory/explainers/gradcam_xfd.py#L14) 到 [`src/explain_factory/explainers/gradcam_xfd.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/explain_factory/explainers/gradcam_xfd.py#L101)。
2. `Explainable_FD_Toolkit` 子项目中已经设计了更完整、更通用的 `ExplainabilityMethod / ModelPlugin / UnifiedExplainer` 体系，[`paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py#L17) 到 [`paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/explainability/core/interfaces.py#L228)。

问题在于，这两套接口目前没有真正收敛。

更严重的是，Toolkit 里的 `model_adapters.py` 还在尝试导入旧路径 `model.TSPN_explainable`、`model.NNSPN`、`model.TKAN`，导入失败后回退到 demo 模型，[`paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/model_adapters.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/model_adapters.py#L18) 到 [`paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/model_adapters.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/Explainable_FD_Toolkit/toolkit_integration/model_adapters.py#L38)。这说明 Toolkit 目前还没有可靠地接到主仓库的真实模型实现。

所以结论很直接：

- **可解释性接口设计是先进的；**
- **主仓库落地是保守的；**
- **二者之间缺一个正式 adapter/compat layer。**

## 5. 代码复用、模块化与技术债务

优点：

- `NSN` 对 `TSPN_UXFD` 的扁平映射是合理的兼容层设计。
- `TSPN_UXFD` 采用插槽式装配，模块边界清晰。
- manifest / predictions / agent artifact 都已经往 callback 化靠拢。

主要问题：

- `Pipeline_01_default.py` 和 `Pipeline_05_default_w_explain.py` 大量重复，artifact 写出逻辑几乎平行维护，[`src/Pipeline_01_default.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/Pipeline_01_default.py#L137) 到 [`src/Pipeline_01_default.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/Pipeline_01_default.py#L205)、[`src/Pipeline_05_default_w_explain.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/Pipeline_05_default_w_explain.py#L146) 到 [`src/Pipeline_05_default_w_explain.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/Pipeline_05_default_w_explain.py#L240)。
- 配置 schema 过于宽松，`extra="allow"` 和 `trainer.extensions: Dict[str, Any]` 意味着很多接口漂移根本不会被发现，[`src/config_schema/models.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/config_schema/models.py#L8) 到 [`src/config_schema/models.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/src/config_schema/models.py#L100)。
- `validate_configs.py` 只验证 `configs/demo` 和 registry active config，不覆盖 `paper/UXFD_paper/*/configs/vibench/min.yaml`，[`scripts/validate_configs.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/scripts/validate_configs.py#L24) 到 [`scripts/validate_configs.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/scripts/validate_configs.py#L39)、[`scripts/validate_configs.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/scripts/validate_configs.py#L68) 到 [`scripts/validate_configs.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/scripts/validate_configs.py#L94)。
- `MOE_explainable` 的 submodule 内已经有独立的真正 MoE 模型 `NNSPNMoE`，[`paper/UXFD_paper/MOE_explainable/code/moe_model.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/MOE_explainable/code/moe_model.py#L17) 到 [`paper/UXFD_paper/MOE_explainable/code/moe_model.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/MOE_explainable/code/moe_model.py#L145)，但 `VIBENCH.md` 和 `min.yaml` 只是把它映射为 `operator_attention` 插槽，[`paper/UXFD_paper/MOE_explainable/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/MOE_explainable/VIBENCH.md#L20) 到 [`paper/UXFD_paper/MOE_explainable/VIBENCH.md`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/MOE_explainable/VIBENCH.md#L23)、[`paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml#L44) 到 [`paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml#L59)。这类“paper 能力被 placeholder 映射”会积累理解债。

## 6. 按优先级排序的优化建议

## P0-1 建立可执行的 Suite Contract，消除子项目接口漂移

问题描述：

当前 suite 合同主要写在 `README.md`、`VIBENCH.md` 和 `UXFD_Family_Tree.md` 中，但没有一个机器可读的 capability/contract 文件来约束 7 个子项目的真实接口。结果是 `VIBENCH.md` 与 `min.yaml` 已经出现偏差。

影响范围：

- 7 个子项目全部受影响。
- 后续 OpenClaw / research_OS / CI 自动化会基于错误接口理解做分析。
- 新人或外部协作者会把文档当真接口，导致复现实验失败或理解偏差。

具体改进方案：

1. 在每个子项目根目录新增 `suite_contract.yaml`，最少包含：
   - `paper_id`
   - `config_surface_version`
   - `entry_config`
   - `entry_pipeline`
   - `model_contract`
   - `required_artifacts`
   - `capabilities`
2. 用脚本自动校验 `suite_contract.yaml`、`VIBENCH.md` 和 `configs/vibench/min.yaml` 的一致性。
3. 明确只保留一个配置表面：
   - 要么统一写 `model.uxfd.*`
   - 要么统一写 `NSN` 扁平接口
   - 不要文档一套、配置一套。
4. 在 `research_OS` 中维护的系列映射表只引用 `suite_contract.yaml` 自动生成结果，不再手工维护关键字段。

## P0-2 把“统一评估协议”从产物清单升级为版本化评估规范

问题描述：

当前只有 run artifact contract，没有真正的 suite-level evaluation protocol。manifest 里缺少协议版本、数据切分口径、解释质量指标、资源消耗和 paper capability 说明。

影响范围：

- 所有跨 paper 横向比较都不稳定。
- Explainable_FD、LLM、NSN、MOE 之间无法形成真正可比较的 benchmark。
- 后续论文表格很容易出现“同名指标但非同口径”的问题。

具体改进方案：

1. 定义 `uxfd_eval_protocol/v1`，至少包含：
   - task metrics
   - explainability metrics
   - optional llm/dialog metrics
   - runtime/resource metrics
   - data split policy
   - random seed policy
2. 给 `manifest.json` 增加：
   - `protocol_version`
   - `dataset_id`
   - `split_id`
   - `git_sha_main`
   - `git_sha_submodule`
   - `capability_flags`
   - `runtime_ms`
   - `peak_mem_mb`
3. 扩展 `collect_uxfd_runs.py`，让它输出 run-level CSV 和 protocol-aggregated CSV 两层结果。
4. 把 `paper/UXFD_paper/*/configs/vibench/min.yaml` 纳入 `validate_configs` 和 CI，不允许只验证 `configs/demo`。

## P0-3 收敛主仓库 explain 接口与 Toolkit explain 接口

问题描述：

主仓库只有轻量 explain artifact 机制，而 Toolkit 已经定义了完整协议；两者没有桥接，导致 `Explainable_FD_Toolkit` 的“统一接口”并未真正成为 suite 公共能力。

影响范围：

- `Explainable_FD_Toolkit`
- `LLM_Explainable_FD_Toolkit`
- 所有需要 explainability benchmark 的方法论文

具体改进方案：

1. 在主仓库新增统一 adapter 层，例如 `src/explain_factory/adapters/`。
2. 让 `NSN/TSPN_UXFD` 暴露最小 explain contract：
   - `get_signal_path`
   - `get_operator_graph`
   - `get_attention_maps`
   - `get_explainability_info`
3. 把 Toolkit 的 `ModelPlugin` 精简成主仓库可维护的最小子集，再由 adapter 兼容 Toolkit 完整协议。
4. 删除或重写 Toolkit 中对旧 `model.*` 路径的导入逻辑，直接对接 `src.model_factory`。

## P1-1 用真实 paper 能力替换 placeholder 映射

问题描述：

若干子项目当前只是“借主仓库插槽占位”，但并未真正接入 submodule 内的核心方法实现。`MOE_explainable` 最典型，paper 代码里已有独立 MoE 架构，但 suite 入口只是 operator-attention 占位。

影响范围：

- `MOE_explainable`
- `LLM_Explainable_FD_Toolkit`
- 后续可能包括 `1D-2D_fusion_explainable`

具体改进方案：

1. 为每个子项目建立 `paper adapter backlog`。
2. 先优先接真实差异最大的三个：
   - MOE
   - LLM
   - 1D-2D fusion
3. 每接一个 paper，都补三件事：
   - capability contract
   - end-to-end smoke test
   - artifact delta definition

## P1-2 抽取共享 pipeline orchestration，去掉 Pipeline_01 / Pipeline_05 重复逻辑

问题描述：

默认 pipeline 与 explain pipeline 大量重复，后续很容易出现一个修了、另一个漏掉的分叉维护问题。

影响范围：

- 全部跑实验的入口
- 所有 artifact 产物逻辑

具体改进方案：

1. 抽出统一的 `run_lifecycle.py` 或 `pipeline_runtime.py`。
2. 把这些行为变成公共 hook：
   - config snapshot
   - metadata snapshot
   - explain eligibility
   - manifest rewrite
3. 保留一个默认 DG pipeline，再通过 `trainer.extensions.*` 驱动行为差异。

## P1-3 强化 schema，使接口错误在配置层就失败

问题描述：

当前 schema 太宽松，很多错拼字段、漂移字段和 paper 自定义字段不会被及时发现。

影响范围：

- 全部配置
- 所有 paper 子项目

具体改进方案：

1. 将 `trainer.extensions` 从 `Dict[str, Any]` 逐步替换为 typed model。
2. 为 `NSN` 明确声明：
   - `signal_processing_2d`
   - `decision_configs`
   - `uxfd`
3. 为 `paper_id`、`preset_version`、`required_artifacts` 加基础校验。
4. 引入 `strict mode`：
   - demo 可 `extra=allow`
   - paper release config 必须 `extra=forbid`

## P2-1 建立 suite 级测试矩阵，而不是只测试核心装配与 manifest

问题描述：

现有测试主要覆盖 `TSPN_UXFD` 装配和 manifest contract，[`test/test_tspn_uxfd_assembly.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/test/test_tspn_uxfd_assembly.py#L63) 到 [`test/test_tspn_uxfd_assembly.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/test/test_tspn_uxfd_assembly.py#L107)、[`test/test_run_artifacts_contract.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/test/test_run_artifacts_contract.py#L13) 到 [`test/test_run_artifacts_contract.py`](/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench%20copy%202/test/test_run_artifacts_contract.py#L149)。对子项目契约、explain adapter、LLM artifact、protocol 版本兼容几乎没有覆盖。

影响范围：

- 所有 suite 级回归风险

具体改进方案：

1. 增加 `test/test_uxfd_suite_contracts.py`：
   - 检查 7 个子项目都存在 `VIBENCH.md` / `min.yaml` / `suite_contract.yaml`
   - 检查 contract 与 YAML 一致
2. 增加 `test/test_explain_protocol.py`
3. 增加 `test/test_paper_min_configs.py`
4. CI 按 `paper_id` 输出 matrix 报告，而不是只跑核心单元测试

## P2-2 做仓库边界清理，降低“子项目像源码镜像”的维护成本

问题描述：

当前主仓库、paper submodule、research_OS 三处都在承载系列级信息，容易形成多源真相。

影响范围：

- 文档维护
- OpenClaw / research_OS 追踪
- 新人 onboarding

具体改进方案：

1. 定义单一事实来源：
   - 代码能力：主仓库
   - paper-specific mapping：submodule
   - 项目管理视图：research_OS
2. 所有系列级索引从代码侧自动生成。
3. 对 submodule 中明显过时的路径说明、旧 import、demo fallback 做一次系统清理。

## 7. 建议的落地顺序

建议按下面顺序推进：

1. 先做 `P0-1 + P0-2`，把 suite contract 和 evaluation protocol 固定下来。
2. 再做 `P0-3 + P1-2 + P1-3`，把主仓库的 explain / pipeline / schema 收敛成稳定底座。
3. 最后做 `P1-1 + P2-*`，逐个把 paper placeholder 替换成真实能力接入。

## 8. 本次检查的验证情况

- 已执行：`python3 -m scripts.validate_configs`
- 结果：`[OK] 14/14 configs passed schema validation.`
- 未执行成功：`pytest`
- 原因：当前环境缺少 `pytest` 模块，无法在本地直接运行仓库测试

## 9. 结论

UXFD Suite 已经有统一入口、统一产物链和可扩展的核心模型装配，这是很好的底子。现在最需要的不是继续往里塞新 paper，而是先把 **suite contract、evaluation protocol、explainability adapter** 三件事定死。只要这三项收口，后面的 7 个子项目就能从“并排放在一起”真正升级为“可协同、可比较、可持续维护的一套系统”。

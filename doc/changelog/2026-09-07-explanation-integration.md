# 2026-09-07：选择性吸收模型解释接口

本记录随 PR #223 合并生效。仅吸收既有 XOAN 与 TSPN-UXFD 的模型内解释轨迹，不整体恢复历史研究分支的运行框架。

## 用户入口

`phmfactory.explanation` 提供两个明确的模型适配器，以及一次用户指定的 LLM 回调。接口不导入服务商 SDK、不读取 API key、不重试、不替换服务商，不在解析失败后自动修复响应。

完整示例见 [LLM explanation integration](../../docs/LLM_EXPLANATION_INTEGRATION.md)。

## 本次吸收与修正

- XOAN 解释与用户选择的 `relaxed` / `discrete` 推断分支一致，不再固定解释 relaxed logits。
- 读取模型实际输出的 `normalized_sparsemax_selection_entropy`；预测熵由当前选择的 logits 计算。
- 对外仅保留算子表达式、执行边与显式干预信息，不复制历史轨迹中的摘要校验字段。
- 模糊分支保留同次前向的规则贡献；只导出部分规则时，不声明完整决策重构能力。
- 拒绝重复类别名，避免多个 logit 维度落入同一贡献目标。拒绝把非文本 LLM 字段转换成字符串后继续接受。
- 导出当前样本实际生效的规则 mask 与 consequent permutation，区分受干预轨迹和原始轨迹。缺少该信息的外部 trace 明确披露未知，不伪造未干预状态。

## 验证范围

针对性测试纳入现有 Core quality gates，覆盖解析和引用检查、真实 XOAN 两种推断路线、真实 TSPN-UXFD 完整／部分贡献、重复类别名，以及共享／逐样本 consequent permutation。没有新增常驻工作流。

合并以最终候选的实际检查结果为准；测试定义的存在不代替执行结果。

## 科学边界

结构化输出和引用检查只验证标识符引用关系，不证明自然语言语义忠实性或物理机理正确性。模型算子和模糊规则不自动等同于物理机制；机制关系必须显式给定。未重跑 THU，未修改数据划分、训练目标、原模型实现、MFPT 状态或发布检查标准。

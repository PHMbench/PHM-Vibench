# 2026-09-08：统一 AI 入口与模块指导

在 PR #235 已纠正四个高风险入口的基础上，完成剩余 15 份模块 CLAUDE.md 的中立化。

## 入口与规范

- 新增根级 AGENTS.md，内容仅为任务边界、只读启动检查、接口导航和验证／报告规则。
- 根级 CLAUDE.md 仅包含 `@AGENTS.md`，不维护第二份指令。
- CORE.md 保留长期科学与组件约束，移除固定任务队列和会过期的当前状态。
- 明确修订原有根级文件禁令：仅允许上述两个准确名称。现有检查继续拒绝个人 Agent 工作区、局部覆盖和重复模块指令；未停用 CI。
- 现有文档检查验证 CLAUDE 导入目标和单行内容。现有边界测试增加对应正反例，不建立新的扫描／策略框架。

## 模块知识去向

| 原指导范围 | 保留位置与处理 |
| --- | --- |
| configs/base | README 保留五段式组合、可移植路径与完整配置验证边界 |
| configs/demo | 现有 README 已覆盖类别、模板与变更规则，不复制正文 |
| configs/experiments、reference | README 区分研究配置、显式本机覆盖与历史模板 |
| Data／Model／Task Factory | README 保留构造与 batch/metadata 接口；纠正旧命令及 catalogue 等同支持的表述 |
| CNN／RNN／MLP／NO／Transformer | 家族 README 保留确切实现名和参数边界，不保留无配置／结果出处的性能表或通用代码配方 |
| ISFM／ISFM_Prompt | README 保留组件分工和逐样本 metadata；不宣称未经测量的参数节省 |
| X_model | README 保留可解释实现与当前研究源码入口；引用闭合不等于物理机理忠实性 |

源码和历史文档仍可通过 Git 历史读取，没有把旧正文复制到新的 Agent 目录。
本次未修改模型、训练、数据、split、objective、metric、checkpoint 或研究 PR #231。
没有新建 hash、manifest、Goal Registry、个人工具配置或重跑 THU。

文件与 CI 检查不证明实际客户端已经加载并遵循指令。Codex／Claude 的真实客户端加载
验收仍需在使用者环境中执行；本轮不声称执行了该验收。

# B−1 Scientific Scope Freeze Prompt

保存自上一轮 B−1 提示词，按 Markdown 整理。执行前以当前用户指令及仓库事实为准，不把参考提交当永久状态。

你正在重构 PHMbench/PHM-Vibench 的模型 catalogue 集成项目。本阶段是 B−1 — Scientific Scope & Integration Governance Freeze，禁止实现新模型，不向 Draft PR #266 追加提交。

## 1. Current state

先只读核验工作区、origin/dev、#266 的 head、提交数和文件列表。网络读取可以使用既有连接器。不要自动 reset、stash、clean 或切换用户分支。

读取 AGENTS.md、CORE.md、Model/Task README、当前 model_registry.csv 和 task_registry.csv。用户允许保存到独立文档分支，不表示允许合入 dev。

## 2. Freeze Benchmark coverage

建立 BENCHMARK_COVERAGE.md，记录 scientific question、required capabilities、simple/strong/distinct-inductive-bias/PHM-specific/optional-pretrained role 和 missing coverage。模型不因出现在 catalogue 中就自动入队。

## 3. Freeze inclusion gate

G1 scientific role；G2 non-duplicate information；G3 auditable source；G4 license；G5 task compatibility；G6 fidelity verifiability；G7 maintenance/resource budget。任何条件未满足不得批准实现。

## 4. Normalize candidate identity

语义身份为 name + variant + task + source_ref。同名不同任务或上游不得混并。输入表未提供精确 variant/ref 时，保留原行及未审状态，不虚构。

## 5. Orthogonal state

分别维护 lifecycle、disposition、reason_code，不继续使用混合枚举。结构已定义不等于全部候选已裁决。

## 6. Catalogue and registry

Catalogue 记录全部候选的研究处置。先核对当前解析器究竟消费什么；只冻结职责，不改运行时 registry 格式或动态导入路径。

## 7. Evidence levels

E0 source/legal；E1 framework conformance；E2 algorithm fidelity；E3 PHM utility。INTEGRATED 需要 E0+E1+E2；Benchmark-ready 需要对应配置的 E3。不得扩大既有软件回归的含义。

## 8. Acceptance profiles

分别定义 native trainable、optional pretrained、statistical/non-gradient、external service。每类只要求有意义的测试。

## 9. Resource budget

读取已经执行的 CI runtime/resource 记录。明确 measured、proposed、unmeasured。可先冻结测试分层与测量方法，不能把未知内存、安装大小或导入成本填成通过。

## 10. Output

保存提示词、原始表格、BENCHMARK_COVERAGE.md、MODEL_CATALOGUE.csv、MODEL_INTEGRATION_POLICY.md、必要资源摘要。只做文件生成及目录自身检查，不写新运行时代码、新 Task、Pipeline、Trainer、Manager、Registry v2 或 hash ledger。

## 11. Acceptance and stop

规则可在不给定固定名单时判定候选应如何处置。说明尚未核验的来源、许可或预算。

完成后停止：不执行 B00；不修改、关闭、评论或合并 #266；不合并 dev；不重训 THU；不发布包或 tag。

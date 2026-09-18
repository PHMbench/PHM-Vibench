# B−1：模型候选处置与验收约定

只用于这一轮候选研究，不建立新的运行时策略层。当前全局规则仍是 [CORE](../../CORE.md) 和 [AGENTS](../../AGENTS.md)。[保存的 Plan v2](prompts/PLAN_V2.md)是来源方案；下列“执行澄清”与方案提案分开记录。

## 1. Catalogue 与现有索引

已核验 Model Factory 根据 `model.type/name` 动态导入并构造 `Model(args_model, metadata)`。当前 `model_registry.csv` 是实现目录和测试说明，不是该动态解析器。因此 B−1 不改其列、名称、内容，不给运行时添加 candidate_id 外键，也不新建 runtime registry。

MODEL_CATALOGUE.csv 只记录候选研究处置，不由生产运行读取。实现通过后，后续模型 PR 可在旧索引按旧格式记录模块，并在 catalogue 记录准确联系。来源/许可证的事实只在源记录和对应实现许可中维护，不再创造第二套 executable support authority。

## 2. 原始来源行与最终语义身份

原 XLSX 有 187 个来源行 M001–M187，并不已经证明为 187 个独立实现变体。完整原文件在本次交付包原样保存，六个 sheet 同时导出 CSV；原来的年份、评分、Tier、Wave、Task 规划和 Existing 名单均是历史输入，不自动成为当前事实。

B−1 对每个来源行建立 `source:Mnnn` 暂定 candidate_id，原行位置保存在 source_catalogue_row。这不是最终语义身份。

最终审查后才使用 `(canonical_name, variant, task, source_ref)`；改变 source_ref 时保留来源关系，不挪用旧版证据。来源行有多个实际变体时可以展开；多个来源行指向同变体时可以归并。必须保留每条原行→最终处置的联系，不能因分裂/去重篡改 187 的来源分母。

未知 variant、revision、license 不填猜测值；本次全部 187 行为 DISCOVERED，未作 E0 审核、未批准 QUEUED、未标 VERIFIED。空 disposition 表示尚未决定，不是 NOT_ADOPTED。

## 3. 三个状态字段

```text
lifecycle = DISCOVERED | AUDITED | QUEUED | IMPLEMENTING | VERIFIED | RETIRED
disposition = EXISTING | CORE_NATIVE | OPTIONAL_ADAPTER | RESEARCH_ONLY | EXTERNAL_ONLY | DUPLICATE | NOT_ADOPTED
reason_code = NONE | LOW_SCIENTIFIC_VALUE | DUPLICATE_COVERAGE | LICENSE | SOURCE_UNAVAILABLE | TASK_UNAVAILABLE | MAINTENANCE_COST | RESOURCE_EXCEEDED | FIDELITY_UNVERIFIABLE | EVIDENCE_INSUFFICIENT
```

进入 QUEUED 需要 coverage 中 G1–G7 均通过。处置与进展独立：EXISTING 不等于 VERIFIED，RESEARCH_ONLY 可以继续有科学任务，NOT_ADOPTED/EVIDENCE_INSUFFICIENT 是特定 release 的关闭决定，不是证明模型不存在。

VERIFIED 必须有实际模型版本、任务变体、配置和 E0–E2 证据。不能从 CI 绿色自动批量改状态。本阶段只冻结这些转移规则，不进行大批转移。

## 4. 证据范围

| 层 | 证明什么 | 不能替代什么 |
| --- | --- | --- |
| E0 来源和许可 | 论文/代码/版本对应与使用边界 | 无法证明能运行或算法正确 |
| E1 框架一致 | 构造、输入输出、状态、实际公共路径 | Dummy 不证明 held-out 独立性或 PHM 优势 |
| E2 算法保真 | 特定算法与所声明变体的输出/目标/推断对应 | 前向一致不等于训练目标、预训练权重或所有分支一致 |
| E3 科学比较 | exact PHM protocol、split、预算、评价与可复算结果 | 小切片不能推广到所有数据、工况或原论文效果 |

INTEGRATED 需要所声明范围的 E0+E1+E2；Benchmark-ready 还需要对应配置的 E3。此处是研究记录术语，不改仓库现有 support enum 或 release checker。

每项证据保存直接的配置、来源、命令、测试名/原始结果位置，不建 receipt/manifest/ledger。

E2 先选官方实现的同输入同权重对齐；确实无法执行时，说明原因并采用独立中间量/等式/极限性质。必须覆盖核心算子和声明的 loss/采样/归一化，而不是和移植实现复制出的“参考”自比较。单独 shape 不合格。

```yaml
fidelity_oracle:
reference_source:
reference_commit:
comparison_target:
weight_mapping:
input_fixture:
tolerance:
known_deviation:
```

已复现的上游 bug 不应静默传播，也不能静默修复后继续叫原样移植；保留反例、改动、变体名称与影响。

## 5. 分型验收

| Profile | 最低可执行验收 | 条件项 |
| --- | --- | --- |
| A 原生可训练 | E0；真实 forward/loss/backward/update；输入/配置不变；状态恢复；算法 oracle；实际 CLI fit/test 与声明指标 | eval 随机性按原算法，不要求随机算法无 RNG；E3 才跑所需真实数据 |
| B 可选预训练 | E0 代码和权重分开；lazy import；无依赖错误；默认不下载；真实本地权重离线加载及推断；Task 输出转换；模型版本/资源定位 | 声称微调时才测试梯度；随机 toy checkpoint 只能证明加载格式，不证明预训练权重集成 |
| C 非梯度/统计 | 真实 fit/predict；训练边界；状态保存；随机种子语义；上游/公式对齐；适用公共推断路径 | 不添加无意义 optimizer；若现有 Lightning 路径不兼容，先记 Task/执行合同缺口 |
| D 外部服务 | 当前 release 只做来源、terms、privacy、复现边界记录 | 不加入 core；不调用收费 API；不把未持有凭据改成空 wrapper |

不适用的测试写 N/A 并解释；适用而未运行是 NOT_RUN，不能用 skip/xfail 冒充完成。

## 6. 权重来源的执行澄清

保存的 Plan v2 讨论了外部权重的 SHA256 字段。B−1 没有下载权重，没有计算任何 digest，也不改全局禁令或新增校验器。

后续若明确集成某一权重，最小来源是 model_id、immutable revision、filename、代码/权重许可和来源 URL；已有上游校验值可以作为来源附注，不作为 PHM 科学正确性证据。是否消费它由该独立 adapter PR 明确，不在本阶段预设新系统。

## 7. 批次与审阅

默认一模型一 PR。仅同一实现主体、同一参考来源、参数化变体、无新依赖且同一 fidelity oracle 能共同验证时合并 2–3 项。共享 Task 必须先有合同及引用基线，不为了拆 PR 提前创建无人使用的抽象。

修改者提交实现和原始结果；fresh-context reviewer 只读源码、参考、diff、tests 和原始 artifacts，不依赖作者总结。如果独立审阅工具/人员未执行，写 NOT_RUN，不冒充独立审批。遵守现有仓库保护规则。

默认依赖/组件 tests、公共 smoke、安装验证复用现有环境与入口；不为每个模型复制 workflow。旧 CI 绿灯可作历史基线，不能替代新 head 验证。

## 8. B−1 完成与后续控制

本次只保存来源材料并冻结 coverage、状态/身份规则、证据 profiles、已测资源基线和未测处理。当前没有候选被科学批准；内存/导入/依赖体积的数值预算未校准，G7 对未来新入队模型仍需补测。

B00 提示词仅保存。B−1 文档未经后续批准，不执行 B00、不拆/关 #266、不创建模型 PR、不合并 dev、不修改其他 PR、main、版本号或发布配置。

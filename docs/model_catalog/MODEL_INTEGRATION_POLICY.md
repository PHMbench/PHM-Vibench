# B−1：模型候选处置与验收约定

本页记录 B−1 冻结的耐久决策，不建立新的运行时策略层。全局项目约束仍以
[CORE](../../CORE.md) 和 [AGENTS](../../AGENTS.md) 为准。

## 1. Catalogue 与现有索引

已核验 Model Factory 根据 `model.type/name` 动态导入并构造
`Model(args_model, metadata)`。当前 `model_registry.csv` 是实现目录和测试说明，
不是该动态解析器。因此 B−1 不改其列、名称、内容，不给运行时添加 candidate_id
外键，也不新建 runtime registry。

`MODEL_CATALOGUE.csv` 只记录候选研究处置，不由生产运行读取。实现通过后，后续
模型 PR 可按现有索引格式记录模块，并在 catalogue 记录准确联系。来源/许可证事实
应由候选来源记录与对应实现许可支撑，不能把候选表变成第二套 executable support
authority。

## 2. 原始来源行与最终语义身份

原始工作簿有 187 个来源行 M001–M187，并不已经证明为 187 个独立实现变体。
所有六个工作表的单元格值保存在
[WORKBOOK_SOURCE_EXPORT.md](sources/WORKBOOK_SOURCE_EXPORT.md)；年份、评分、Tier、
Wave、Task 规划和 Existing 注释都是历史输入，不自动成为当前事实。

B−1 对每个来源行建立 `source:Mnnn` 暂定 candidate_id，原行位置保存在
`source_catalogue_row`。这不是最终语义身份。

最终审查后才使用：

```text
(canonical_name, variant, task, source_ref)
```

改变 source_ref 时不能挪用旧版证据。一个来源行可以展开为多个真实变体，多个来源行
也可以归并为同一语义候选，但必须保留每条原行到最终处置的联系，不能因分裂或去重改变
187 行来源分母。

未知 variant、revision、license 不填猜测值。B−1 冻结时全部 187 行仍为
`DISCOVERED`；空 disposition 表示尚未决定，不等于 NOT_ADOPTED。

## 3. 三个正交状态字段

```text
lifecycle =
  DISCOVERED | AUDITED | QUEUED | IMPLEMENTING | VERIFIED | RETIRED

disposition =
  EXISTING | CORE_NATIVE | OPTIONAL_ADAPTER |
  RESEARCH_ONLY | EXTERNAL_ONLY | DUPLICATE | NOT_ADOPTED

reason_code =
  NONE | LOW_SCIENTIFIC_VALUE | DUPLICATE_COVERAGE |
  LICENSE | SOURCE_UNAVAILABLE | TASK_UNAVAILABLE |
  MAINTENANCE_COST | RESOURCE_EXCEEDED |
  FIDELITY_UNVERIFIABLE | EVIDENCE_INSUFFICIENT
```

进入 QUEUED 需要 coverage 中 G1–G7 均通过。处置与进展独立：EXISTING 不等于
VERIFIED；RESEARCH_ONLY 可以继续有研究价值；NOT_ADOPTED/EVIDENCE_INSUFFICIENT
是特定审查时点的明确处置，不证明模型不存在。

VERIFIED 必须绑定实际模型版本、任务变体、配置和 E0–E2 证据。CI 绿色不能自动批量
晋级候选。

## 4. 证据层级

| 层 | 证明什么 | 不能替代什么 |
| --- | --- | --- |
| E0 来源和许可 | 论文/代码/版本对应与使用边界 | 无法证明能运行或算法正确 |
| E1 框架一致 | 构造、输入输出、状态、实际公共路径 | Dummy 不证明 held-out 独立性或 PHM 优势 |
| E2 算法保真 | 所声明算法变体的输出、目标、归一化、采样或关键算子对应 | forward shape 不证明训练目标或预训练权重一致 |
| E3 科学比较 | exact PHM protocol、split、预算、评价与可复算结果 | 小切片不能推广到全部数据或原论文效果 |

研究记录中：

```text
INTEGRATED      = E0 + E1 + E2
BENCHMARK_READY = E0 + E1 + E2 + E3
```

这些术语不修改仓库现有 support enum 或 release checker。

每项证据优先保存直接的配置、来源、命令、测试名和原始结果位置，不新建
receipt/manifest/ledger。

E2 首选官方实现的同输入同权重数值对齐；确实无法执行时，说明原因并使用独立中间量、
等式、极限性质或算子参考。必须覆盖所声明的核心机制，而不是复制移植实现作为“参考”。
单独 shape smoke 不足。

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

已复现的上游 bug 不应静默传播，也不能静默修复后仍称“原样移植”；必须记录反例、
改动、变体名称与影响。

## 5. 分型验收

| Profile | 最低可执行验收 | 条件项 |
| --- | --- | --- |
| A 原生可训练 | E0；真实 forward/loss/backward/update；输入/配置不变；状态恢复；算法 oracle；实际 CLI fit/test 与声明指标 | eval 随机性按原算法；E3 才跑对应真实数据 |
| B 可选预训练 | E0 代码和权重分开；lazy import；无依赖错误；默认不下载；真实本地权重离线加载及推断；Task 输出转换；版本/资源定位 | 声称微调才测试梯度；toy checkpoint 只证明格式 |
| C 非梯度/统计 | 真实 fit/predict；训练边界；状态保存；随机性；边界；参考等式或上游对齐 | 不制造无意义 optimizer |
| D 外部服务 | 来源、terms、privacy、复现边界 | 不加入 core；不调用收费 API；不做空 wrapper |

不适用的测试写 N/A 并解释；适用而未运行是 NOT_RUN，不能用 skip/xfail 冒充完成。

## 6. 外部权重来源

B−1 没有下载外部权重、计算新的 digest 或增加校验器。后续若明确采用某一权重，最小
来源信息应至少能回答“实际验证的是哪一权重版本”，例如：

```text
model_id
immutable revision
filename
code / weight license
source URL
```

若上游本身发布校验值，可以作为来源附注；它不替代 PHM 科学正确性验证，也不授权建立
通用 artifact ledger、hash chain、receipt 或 attestation 系统。

## 7. PR 与独立审阅

默认一模型一 PR。仅同一实现主体、同一参考来源、参数化变体、无新依赖且同一
fidelity oracle 能共同验证时合并 2–3 项。

共享 Task 必须先有真实合同和消费者；不能为了拆 PR 提前创建无人使用的抽象。

修改者提交实现和原始结果；fresh-context reviewer 只读源码、参考、diff、tests 和
原始 artifacts，不依赖作者总结。没有实际执行独立审阅时必须标 NOT_RUN，不能把作者
自查或 CI 绿色称为独立批准。

## 8. B−1 之后的控制原则

B−1 只冻结 coverage、候选身份/状态、证据 profiles、资源测量方法和 catalogue /
runtime 职责。它不批准任何具体候选，也不自动排出后续模型实现队列。

资源页已经取得历史 CI 时间与一个简单基线的局部 import/RSS 观测，但完整 package
import、训练峰值、正常安装依赖增量和真实 PHM/GPU 预算仍须在相关候选实际入队时按
适用范围补测。未知维度不能自动 PASS G7。

任何后续阶段都需要新的当前状态核验与明确授权，而不是从本页推导固定 PR 编号或
“下一任务”。

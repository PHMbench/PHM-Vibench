# B−1：科学范围与覆盖矩阵

日期：2026-09-18。以下是科学范围与候选选择规则，不是实验结论，也不是所有模型的
接入批准。

依据：[CORE](../../CORE.md)、[Model Factory](../../src/model_factory/README.md)、
[Task Factory](../../src/task_factory/README.md) 与 187 模型 / 18 Task 的原始候选表。
B−1 所依据的 dev 快照并不包含当时实验分支上的新增预测实现，因此这些实现没有被当成
已支持能力。

## 1. 首要科学问题

| ID | 科学问题与目标 | 本轮地位 | 必要比较角色 | 有效证据与适用边界 |
| --- | --- | --- | --- | --- |
| S1 | 在相同训练/测试总体、预算与模型选择规则下，振动故障分类结果有多少来自架构归纳偏置，而非数据或预算差异？ | 首批核心范围；复用已有分类 Task | 简单统计/线性控制；常规可训练时序基线；只有预先提出机制假设时才增加差异化偏置代表 | 独立记录/机器分组；工况变化单独定义；acc/macro-F1 对明确标签集合计算 |
| S2 | 仅给定历史振动/工业传感器窗口时，周期、频域或跨通道建模是否改善明确预测区间上的误差—成本关系？ | 条件核心范围；先冻结 point-forecasting 科学合同，再批准模型 | 持续值/简单线性控制；一个匹配预算的现代基线；一个确实补足 S2 假设的代表 | 原始时间先分组再构窗；history-only；normalization 不含未来后缀；MSE/MAE 明确 horizon/channel 支持 |
| S3 | 固定标签与支持集预算下，表示或适配方式是否改变跨工况/少样本诊断？ | 保留已有 FS/GFS/DG 能力；不是本次批量新建算法的理由 | S1 可复用 encoder；简单 probe/原型对照；只比较真实不同干预 | support/query 不泄漏；域与标签可用性明确；协议未审完不宣称覆盖 |
| X1 | 异常检测：正常训练下如何检出异常及校准阈值？ | 后续候选，须有具体 Benchmark 需求才开 Task PR | 可解释分数控制 + 一个必要异常方法 | 正常训练、验证校准、测试标签严格分离；point-adjustment 不默认使用 |
| X2 | 缺失观测：只用 observed values/mask 能否恢复指定支持集？ | 后续候选 | 简单插补控制 + 一个结构不同的方法 | mask 分布、随机重复、归一化和 metric support 明确 |
| X3 | 外部预训练：相同信息/资源条件下，预训练是否带来迁移价值？ | 可选研究轨道，不为凑数接基础模型 | 未预训练、冻结 probe、明确迁移设置 | 真实权重、训练数据重合、上下文和输出语义可追溯 |
| X4 | 图、多模态、RUL、生成：额外结构或输入能否解释性能差异？ | 具体论文提出后单独立项 | 与干预匹配的简单控制 | 实际拓扑、退化标签、多模态同步或生成评价缺失时不强行扩展接口 |

187 是原始来源行数，不是 core 模型 KPI；旧规划中的“30–45 / 12–20 / 6–8”
同样不是当前完成条件。

## 2. 覆盖不是模型计数

对 S1/S2 的必要角色，后续采用记录应连接：

```text
科学角色
→ exact task/config
→ 当前实现
→ E0/E1/E2
→ 对应 E3 / 资源记录
```

一个模型可以覆盖多个角色；同一 backbone 的别名或仅换宽度不能伪造独立科学信息。
已有模型可以满足角色，不要求为了“新”而再接一个。弱于简单基线的合法负结果也保留。

只有适用角色都有实际证据时才可以宣称相应科学覆盖完成。

## 3. Scientific Inclusion Gate

先回答：

> 如果不采用这个候选，S1/S2 或已批准论文的哪条结论会实质变弱？

| 条件 | 必需材料 | 未满足时 |
| --- | --- | --- |
| G1 科学角色 | 指定 scope ID、比较作用、目标结论 | 不入队；LOW_SCIENTIFIC_VALUE 或保持待审 |
| G2 非重复信息 | 与现有实现的具体差别及控制实验 | 同义/别名记 DUPLICATE；必要证伪 comparator 可说明保留 |
| G3 来源 | 论文/实现对应、固定源码版本、真实文件 | SOURCE_UNAVAILABLE / EVIDENCE_INSUFFICIENT |
| G4 许可 | 代码和权重分别确认使用/再分发边界 | LICENSE |
| G5 Task | 已冻结输入、禁用信息、目标和输出 | TASK_UNAVAILABLE |
| G6 保真 | 比较对象、参考代码、测试权重、容差、偏离 | FIDELITY_UNVERIFIABLE |
| G7 维护预算 | 适用环境的运行、依赖、CI 和维护依据 | 未测不判 PASS；超预算则 optional/research/not-adopted |

G1–G7 同时通过才进入 QUEUED。未逐候选审查时保持 DISCOVERED，不能为了“187/187”
人为填结论。

## 4. 两个首要 Task 合同

### S1：固定窗口监督分类

输入由既有 Dataset 明确提供信号、标签和 sample/file/domain identity；具体布局以
实际模型合同为准，不强迫重写全部历史模型为 BLC。新 BLC adapter 需要明确转置边界。

禁止：
- 标签作为模型的非协议输入；
- target-domain/test 数据参与 scaler、HPO 或模型选择；
- 隐式改变类别集合、域、通道或数据总体。

模型输出对应明确标签集合的 logits，或显式 features；Task 拥有 loss、metric、
optimizer，Trainer 拥有 device、fit/test 与 selected checkpoint。

训练随机策略显式。eval 是否具有随机性按原算法判断，不能机械要求所有模型 bitwise
deterministic。跨 seed 汇总只能合并相同 estimator。

### S2：历史前缀到固定未来后缀的确定性点预测

最小合同：

```text
model input  = history [B,L,C]
task target  = future suffix [B,H,C]
model output = point forecast [B,H,C]
```

首批不把未来协变量、非规则采样或概率预测伪装成这个合同；需要这些输入的模型先进入
Task 缺口。

禁止：
- target suffix 进入模型输入、输入 normalization、频率选择或 context embedding；
- test 数据用于 early stopping、阈值、HPO 或模型选择；
- 为适配现有 Task 删除上游算法必需输入后仍声称完整保真。

Task 在冻结的 horizon/channel 支持上计算 MSE/MAE，并明确原始单位与 normalization。
不能先逐 batch 求平均再错误平均不同 batch。

如果上游方法训练目标是全长重建，而移植只优化预测后缀，即使 forward 类似也属于不同
变体；E2 必须明确这个差异。

## 5. B−1 的耐久结论

B−1 的结果是以上科学选择规则与证据边界。它不裁决全部 187 个候选、不选择固定模型
名单、不修改运行时，也不指定以后必须执行的 PR 编号或顺序。

任何后续模型或 Task 变更都应重新核验当前 dev，并按 G1–G7 与相应 E0–E3 证据决定。

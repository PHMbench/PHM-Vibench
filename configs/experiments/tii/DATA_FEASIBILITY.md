# Goal：TII支持条件化论文的数据依据与分阶段可执行性

## 核心主张及必需缺口

本文是 `AI4Engineering-L/P-vibench-HSEv2` 的TII支持条件化联合学习研究，不是 `experiments/p01/` 的TSPN残差研究。SC/UU在配置/历史产物中仍分别写作 `support` / `ordinary`；不重命名原数据或方法字段。

现有资料没有支持工业SC/UU主比较。本任务沿用原Goal07：先解决PU原始观测解释，再用已有资格和记录清单检查各阶段是否可执行；不以软件通过替代数据资格。当前唯一执行批次是这些CPU前置检查，不启动新的长训练或更多模型。

问题/假设：PU Id47567的时间、单位与元数据差异，能否由原始材料解释而不改变采集模型？资料修正后，源拟合、独立目标迁移和pooling分别具备哪些条件？结果可以是resolved、incompatible或unresolved。目标量是记录级测量摘要、证据覆盖率与静态可行性，不是NLL、准确率或模型优势。

已有完成项直接复用：`reports/tii_local_j1/`的资格/折/原记录表、`RAW_ACQUISITION_CHECKS.csv`、原始与缓存比较、M0及恢复记录。不要重复18-H5全库扫描、旧20步构造训练、失败episode或旧指标计算。

## Inputs and protocol

- 数据只读根：`/home/user/data/PHMbenchdata/PHM-Vibench`，现有README、metadata.xlsx、PU H5及原MAT/采集资料。实际路径按原生Name/Id/File与`RECORD_INVENTORY.csv`解析，不猜路径或物理组。
- 起点：PU Id47567、`K001/N09_M07_F10_K001_1.mat`。旧检查为256823样本、跨度平均约64205.445 Hz，metadata为64000 Hz，Unit字段为空；平均速率不证明均匀采样，原时间与振动通道的关联未闭合。
- 依赖：先核对当前子仓和论文gitlink，复用 `src/task_factory/Components/tii_target_head.py::make_episode`、元数据接口和既有读取器。`select_l2`只验证分数网格完整，不证明排除源拟合的来源合法。
- 固定：原波形、真实record/group、局部标签、共同/增量支持和完整可用/不可用规则；每类5条原记录、PCG64(1729)、剔除全部adaptation组、每类至少2个query组。不重抽，不换shot，不按模型分数选source/target。
- 统计单位：原始物理组不能拆成窗口或按类别拆开重复计数。本任务只做固定集合描述；不bootstrap旧构造结果，不把通过抽样的PU直接记为整体合格。
- 资源：沿用 `LQ_signal`，只读CPU检查；0次fit、0次HPO、0次GPU训练。必要的新时间字段检查仅覆盖已有PU inventory，复用已查项。其他来源只复用现有资料以核对阶段逻辑，不新增跨数据集搜索或下载。

## Execution

工作目录为PHMFactory根目录。只读核对：

```bash
git status --short
git rev-parse HEAD
conda run -n LQ_signal --no-capture-output python -m phmfactory --help
```

1. **PU原始依据。**读取既有报告，核对Id47567振动channel与time-field的实际关联、n_signal/n_time、首末时间、重复/倒序、dt范围及分布摘要。明确区分时间关联错误、存储/导出规则、metadata错误和真实非均匀采样；证据不足写unresolved，不能选择最有利解释。
2. **补齐原来未检查的PU范围。**按同一inventory记录原始时间/单位字段；已检查且可追溯的条目复用。单位只能由原采集资料/换算记录确定，不能根据幅值补g或m/s²。名义64kHz和30kHz低通不能自动提供本条记录的时间映射、校准和过渡带依据。不得原地改用户H5/Excel，若有确定错误仅形成带原值/新值/依据的建议。
3. **记录PU裁决。**只解除已用证据解决的子缺口。若需要插值、截短、删异常记录或更换支持模型才能通过，记为incompatible/需协议决策；不实施这些修复。无法取得更多依据则停止数据读出，保留未决字段。
4. **区分阶段可执行性。**在原 `FOLD_ELIGIBILITY.csv` 对应的后续副本/原生产物中，单独记录下表要求，不把所有任务绑到一个全局J1布尔值。沿用已有表，添加必要列即可，不新建registry。

| 阶段 | 静态条件 | 允许的后续工作 |
|---|---|---|
| C5源拟合 | 至少2个合格独立来源；source_train/val组分离；源几何/RMS可定义；实际共享实现通过 | 原有一次SC/DLinear fit95；不要求必须先有held-out target |
| C6迁移 | 总计至少3个合格数据集；排除t后至少2个源；target固定episode可行；非空合法inner-alpha任务集 | 才具备执行SC/UU的前提；额外训练预算仍需确认 |
| C7 pooling | C6及四格全部可执行；single source预先固定、可观测完整union；四格同full-pool geometry与adaptation规则 | 才能估交互；强diversity归因还须record/exposure匹配 |

5. **在任何encoder fit之前检查inner任务。**对每个预定outer target，按同一source几何枚举全部pseudo-target候选。检查排除该pseudo-target后是否仍有至少一个source观测完整fixed union，再在其允许的source-side record inventory上调用原生固定episode规则。逐候选保留可用性、episode状态/原因及group数。只有通过这些条件的非空集合才能进入完整alpha网格；不能先训练，再因失败换pseudo-target。目标query标签可由custodian用于固定划分，不可用于拟合或选择。
6. **停止并同步。**若仍不足3个合格dataset，只报告C6 blocked，不把两源fit或其他TSPN/PU协议当作迁移结果。本次交付只更新前置证据及阶段状态，不自动启动Goal08或整个主比较。

现有可调用函数是 `make_episode(records, shots=5, seed=1729)`；输入为每原record一行的 `recording_id/group/label`。本轮没有运行原MAT，也没有交付一个新的静态可行性CLI。必要的只读诊断在子仓复用现有接口，以最小改动输出新增字段；不能声称 `run.sh pu` 或 `--transfer-preflight` 已存在。实际实现、命令及退出状态写入本轮commands，不另建训练入口。

源拟合资格真的成立后，另次按既有 [FIT95.md](FIT95.md) 执行下列**已有入口**；相对路径以子仓为准。真实YAML未提供时不可执行，不将示例当已验证配置：

```bash
bash scripts/run_tii_fit95.sh --check-only configs/experiments/tii/one_model.local.yaml
# 仅通过源拟合资格并取得本地执行授权后：
bash scripts/run_tii_fit95.sh configs/experiments/tii/one_model.local.yaml results/tii_fit95_001
```

## Artifacts and validation

沿用 `reports/tii_pu_observation_followup/`：`record_observations.csv`、`acquisition_evidence.md`、`decision.md`、`commands.md`与真实失败日志。记录复用/新检查/未执行、原record与字段关联、单位依据和异常，不填虚构PASS。

阶段可行性继续使用现有资格/折表命名，在新报告目录保留版本，不覆盖历史failure/ineligible。必要字段包括：qualified source数、outer-target排除后的source数、source-fit资格、target episode、每个inner pseudo-target的remaining-full-union/episode结果、可用inner数量、single-source/four-cell状态、原因及代码版本。数量下限不是覆盖充分或论文可投保证。

future统计约定回填论文Supplementary A，不在本批执行：每target每seed效果与范围；条件paired physical-group区间。仅当group为单一类别时class-stratified采样并用原始n_c/n保留group均衡estimand；多类别物理组必须整体重采样，缺类replicate的相应指标记未定义、报告数量，不重抽、不拆组。旧0.005 nat不再作为等效或实用优势阈值。该约定尚需正式evaluator按新真实协议验收；旧结果不回溯改写。

图只读真实摘要，不调用模型。本文待回填位置是科学状态、现有资格说明与Supplementary A；本批不修改工业Results数字或声称方法有效。

## Failure and sync

时间解释受阻时可核对独立的单位/采集资料；资料不存在就提交准确阻塞并停止，不重开全文审查。判定当前信息不足与确认原测量不适用要分开；负向资格结果同样是有效任务交付。

先子仓正常PR合入dev，保留真实代码、配置、摘要和失败日志；禁止强推/改master/删除他人分支。论文仓只同步准确可取回gitlink、原有STATUS/实验表引用，不复制执行正文。大文件使用已有存储，原H5与受限采集文件不上传。当前文件仅为文档与实验前置规范，不代表新诊断已运行或资格已通过。

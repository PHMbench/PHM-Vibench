# Goal 07 — HSE/TII 的 PU 观测依据：本批唯一执行任务

服务论文：`AI4Engineering-L/P-vibench-HSEv2` 的 TII 多源时序表征研究。此文件续接并替代论文侧 `paper/goals/local_joint/07_PU_OBSERVATION_PROVENANCE.md` 的执行正文；不是新一轮同义审查，也不属于相邻的 TSPN D1/G07/G08 协议。

**核心主张及必需缺口：**C5/C6 的真实多源学习及独立目标比较需要可解释的采样、物理尺度和分组。既有 J1 为 No-Go；不解决这些输入含义，工业迁移效应没有有效评价对象。PU 是既有固定 episode 唯一通过组数条件的候选，不是按模型成绩挑选的目标。

**问题与测量：**原始振动通道关联的时间字段、单位和本次使用频带的采集资料，能否解释 `Id47567` 的元数据冲突？配对比较同一记录的原始依据与已存检查，报告时间关联、样本/时刻数量、时间间隔摘要及有依据的尺度。结果可以是 resolved、incompatible 或 unresolved；不预设合格，不计算 tokenizer NLL/F1。

**完成项与复用：**使用 `reports/tii_local_j1/` 的资格表、完整记录 inventory、失败项和原始检查。18份H5/19任务/49,855索引项的资格扫描、构造源20轮、checkpoint恢复、既有理论与旧分数不重做。报告起源是 child `20528d286f0c15e7de6fe93ee2fb7008d644a750`，不是当前代码版本的替代名称。本次规划读取的实现 dev 为 `1902d839c3ae8a6d7f72ec591983f698eb12d942`；报告树仍无本Goal的完成产物。作者本机可能有未同步工作，先查实际产物，已有相同有效检查直接复用。

## Inputs and protocol

- 数据：只读 `/home/user/data/PHMbenchdata/PHM-Vibench/{README.md,metadata.xlsx,RM_027_PU.h5}`。路径由用户提供；本轮远端未访问。依据 `RECORD_INVENTORY.csv` 与原生 metadata 定位原始 MAT，不拼接猜测路径。
- 起点：PU `dataset_id20 / Id47567`，原记录 `K001/N09_M07_F10_K001_1.mat`。已有检查记录256,823点、跨度约4.000003421789442 s、跨度平均速率约64,205.445 Hz、metadata 64,000 Hz、vibration Unit为空。这些是待解释的旧事实，跨度平均值不证明均匀采样。
- 输入证据：`DATASET_QUALIFICATION.csv`、`FOLD_ELIGIBILITY.csv`、`RECORD_INVENTORY.csv`、`RAW_CACHE_COMPARISON.csv`、`RAW_ACQUISITION_CHECKS.csv`、`DATA_QUALIFICATION_NOTES.md`。相关失败和未执行状态原样保留。
- 依赖：论文实际 gitlink 与子仓实际 HEAD 分别记录；Goal 所在提交不自动授权升级生产依赖。正式读取路径仍是 `src/data_factory/data_utils.py::read_metadata_table` 及 `src/data_factory/reader/RM_027_PU.py::read`。后者组合电流与振动数值，但没有证明时间字段与振动通道的关系，因此必须查看原 MAT 关联字段。
- checkpoint/API/GPU：均不需要；使用现有 `LQ_signal` 环境。允许只读 CPU 诊断，0次训练、0次模型推理、0次 HPO。
- 配置：本地诊断仅记录数据根、已有inventory、PU身份、sentinel、实际测量资料位置与新输出目录。原 `LOCAL_PLAN.yaml` 不是本诊断的运行YAML，不修改 Method、单位、采样、波段、标签、分组或目标适配预算。
- 比较：同一原记录的原采集说明、MAT字段、metadata与已有缓存检查。不是两种模型，也不是人为强套因果实验。
- 统计单位与重复：对既有PU记录清单作一次确定性的新增字段检查；已核验且协议相同的行复用。0个优化seed、0个新episode，不要求3次重复、不bootstrap窗口。按物理bearing/工况描述异常及记录覆盖率；缺时钟精度依据时报告原差异，不按结果选择容差。

## Execution

工作目录是 PHMFactory 根目录。先检查而不自动切分支、覆盖未跟踪文件或改父仓指针：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
sed -n '1,80p' reports/tii_local_j1/J1_GATE.md
sed -n '1,160p' reports/tii_local_j1/DATA_QUALIFICATION_NOTES.md
conda run -n LQ_signal --no-capture-output python -c 'import sys; import phmfactory; print(sys.executable); print(phmfactory.__file__)'
```

上述路径和读取接口经源码核对；本轮只验证命令语法，未在作者电脑执行。不存在已验证的 `run.sh pu`。`scripts/tii_qualify_data.py` 是已完成的全库检查，不能重新调用它来代替本次有界诊断。

1. **先复用。**检查工作区与 `reports/tii_pu_observation_followup/` 是否已有本任务产物；核对实际配置、范围和终态。兼容且完整则直接同步，不重测；不完整只补确切缺项。
2. **解析 sentinel。**由已有inventory定位原MAT和实际振动字段，检查通道—时间索引、时间语义和单位字段。已证实相同的整段H5数值不再逐点比较。区分字段配错、时间戳精度/导出规则、元数据错误及真实非均匀采样；证据不能区分时标 unresolved，不试到“接近64kHz”。
3. **仅补 PU 新字段。**沿既有inventory核查尚未验证的通道关联和时间摘要，保留全部计划记录的 reused/checked/failed/not_run 状态。不得重扫18库、重新切窗或生成等间隔时间；没有可靠时间关联时不计算假定正确的速率。只需要读取产生新测量所必需的原始字段。
4. **独立补齐采集依据。**读取本地原始设备/采集说明，核实所选振动通道的单位及拟使用频带所需的响应依据。典型传感器目录值、名义低通截止和其他实验的规格不能替代本批资料。无需追求与既定频带无关的全硬件百科；必要字段确实无法取得时列出确切缺项，停止该分支，仍可完成独立的时间关联检查。
5. **实现范围。**复用原始reader/metadata接口，在实现仓当前诊断owner下补一段必要的只读字段检查。直接检查原MAT结构用于语义诊断，不复制或注册第二个训练loader。本轮不提供尚不存在的批处理命令：本地Agent完成后将实际执行命令、环境、版本、输入范围、输出位置、退出码写入 `commands.md`；帮助输出不等于完成测量。
6. **裁决并停止。**输出哪些矛盾得到原始材料解释，哪些确认不相容，哪些仍缺依据。只提出有原文位置的更正建议，不原地改H5/metadata。不重抽seed1729的5-record/class适配集，不启动Goal08、J2、N/H、J3或其他数据集。

## Artifacts and validation

沿用既定 `reports/tii_pu_observation_followup/`，不另建registry或平行结果目录：

| 产物 | 必要内容 |
|---|---|
| `record_observations.csv` | metadata_id、原文件引用、bearing、振动字段、关联时间字段、n_signal/n_time、首末时刻、重复/倒序数、dt min/median/max、可合法计算的跨度速率、metadata_fs、单位原文/依据、复用或新检查状态、失败原因 |
| `acquisition_evidence.md` | 原始资料位置、实际关联及解释边界；仅引用许可允许的片段，不公开受限文档或原波形 |
| `decision.md` | resolved/incompatible/unresolved、记录覆盖、已解除与未解除的具体条件；不是工业效果或自动J1 Go |
| `commands.md`及必要日志/配置 | 实际工作目录、解释器、两仓版本、命令、输入范围、终态和失败。没有测量的时间/内存不填0 |

验收检查原信号、identity、协议未被修改；每条计划记录有状态；时间轴确与振动通道关联；错误或缺项没有被丢弃。不存在可靠关联的行保留缺值及原因。若新增诊断代码，用最小结构夹具验证字段关联错误能被发现，不用重训练或全库扫描验证文档。

本批不要求预测、checkpoint、置信区间或新图。确需解释时间结构时仅从已保存CSV绘图，代码留实现仓，不调用模型。证据B以后只将有效资料回填论文已有的测量资格/限制表与贡献映射；本批不修改 Results、Abstract 或 Method。

## Failure and sync

- 缺原MAT、环境或采集资料：记录具体缺项与其影响，继续本任务内不依赖该项的必要检查；没有独立可执行项则停止。环境修复可在不改变数据/方法协议的前提下进行并记录；不更换数据、求解方法或硬件掩盖失败。
- **Resolved**仅解除已有依据支持的PU子问题；**incompatible**是有效的观测模型反例；**unresolved**是证据不足而非算法负结果。一个PU不构成多源训练，也不授权下游GPU。需要改变支持定义、单位换算、分组或episode时，将理由交回设计阶段，旧协议结果不混用。
- TSPN/MLP16的PU D1/G07属于不同模型与协议，其已揭盲测试和拟合不得用于本论文选模型或证明S/U有效；本任务只读采集语义，不重新评估这些模型。
- 先在实现仓当前dev的独立工作分支提交必要诊断、真实摘要及失败记录，经针对性检查和当前PR检查后正常合入dev。大/受限资料沿用现有授权存储，dev保留可访问引用，不强塞Git。
- 然后论文仓只同步实验表中的证据位置与必要依赖说明；是否更新生产gitlink须核对受影响实现和接受状态，不自动把其他协议的最新dev并入实验基线。不改master、不强推、不删其他分支、不重复合并已完成PR。
- 记录实际两仓提交；交回真实产物后停止，等待证据B。没有新的可改变裁决的信息，不再改名生成Goal或重开全文评审。

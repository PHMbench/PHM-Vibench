# 2026-10-17 有界仿真：输入组织与读出尺度

本协议在实现及查看新仿真结果前固定。用户本轮明确要求优化 motivation 并仿真验证；这是新增构造机制检查，不释放暂停的工业研究、Goal08、工业 J2/J3 或 HPO。复用原生 S/U、DLinear、共享源 RMS、局部 heads、source-only 选模与现用目标 head，不引入新模型或数据工厂。

**问题锚点：在相同观测并集、监督和训练资源下，common/increment 输入组织能否改变工业迁移风险，其收益与失效条件是什么？** 本仿真仅检查该问题中的有限模型假设及竞争解释；工业问题仍须真实主比较回答。

## 固定生成机制

- 两个相反任务：`common_informative` 中完整来源/目标标签来自共同分量；`increment_informative` 中完整来源/目标标签来自增量分量。低带宽来源始终以可观察的共同分量作局部监督，标签取反，来源间没有标签语义统一要求。
- 每个独立记录一秒、一个窗口、一个独立生成组；共同正弦频率为3或5Hz，增量为12或16Hz。两个潜变量按四种组合均衡，无相关标签捷径；每记录独立均匀相位、幅度 U[.8,1.2]、Gaussian噪声标准差.1。完整来源64Hz，低带宽来源16Hz且不生成增量，独立目标64Hz。来源分别生成，不共享事件或降采样同一记录。
- 每源64个train、32个val记录，目标128记录；角色用不交叠的记录/组ID。全部以解析构造的 m/s² 单位定义，证据种类 `tensor_fixture`，不是已核定工业采集。原生整窗口投影：common=[1,8)、increment=[8,24)Hz，64Hz公共网格、K4/P16。不会修改任何真实数据。
- 固定三个联合重复 `(data_seed, optimizer_seed)=(101,0),(202,1),(303,2)`；两任务共享同一生成规则，不筛选有利种子。数据与优化随机性在这三个独立重复中共同变化，不能据此分别归因。

完整来源与目标使用相同生成分布、不同独立记录；本设计没有人为加入目标分布偏移。它检查的是该生成器下冻结表示对独立新记录的任务可用性，不构成OOD或工业跨库迁移验证。两任务共享同一批生成信号而改变完整来源/目标标签依据，不能将任务情景当作额外独立重复。

## 学习、选择和适配

- 两任务×三重复×S/U，共12个CPU encoder fits；每fit固定300更新，每源32窗/update。原生 SourceRounds、tii_joint loss/AdamW（lr=.001、wd=.0001，原betas/eps，无scheduler），两辅助系数为0。
- 原生 M_01_ISFM / SupportConditionedTokenizer / B_04_Dlinear / H_01_Linear_cla，输出D8、每支隐藏宽度16；匹配参数、初始化与批次。隐藏宽度不小于P16，不人为制造首层碰撞；记录初始化和选定状态U首层堆叠的秩及最小奇异值。两个分支参数始终跨来源共享。
- RMS只在source_train上、按原生等源/组/窗规则拟合。每50更新记录完整source_val NLL（含step0）；只按该值、1e-8平局取早选择一个checkpoint。完整源train/val诊断与轨迹保留，不看目标选模，不延长预算，不添加HPO。
- 目标在encoder冻结后由现用 `make_episode`（seed1729、5原记录/class）选support；全部support组排除于query，失败不重抽。原生mean pooling、`fit_head`，alpha=.01为此仿真事前固定值，不声称已经完成工业source-only inner选择。

## 尺度正对照

对每个冻结表示r，先拟合原head；在2r上复用W/2、b检查logits/NLL/决策不变，再以同alpha额外拟合一次head，共24个head fits（12原始+12缩放）。不扫描尺度或alpha，不把缩放head当新的encoder。记录两head的求解迭代、gradient_inf和真实query logits/NLL/accuracy。该对照保持表示信息和无限制仿射函数族不变，测量正则化适配的坐标敏感性。

## 输出、判据与停止

保存每记录角色/标签清单、实际配置/命令/环境、每fit selected checkpoint与RMS、冻结features、原始及缩放head与全部query logits、源训练曲线、逐重复差值、均值与sample SD、实际CPU时间/内存。结果正、负、未决全部保留。软件测试另用小规模构造输入，不能计作额外独立科学重复。

预定 estimand 为各任务/重复的group等权目标NLL差 S−U；所有记录仅一窗，组均值等于记录均值。另报告accuracy、source train/val、相同r同alpha缩放重拟合的NLL变化及决策变化。只有三个重复，dispersion不当作精确总体区间，不预置显著性或“必须获胜”阈值。

若S仅部分情景有利，则只保留对应构造条件的有限学习结果；若S更差/无稳定优势，则收缩“有效性”表述，不能调整生成器救正结果。尺度对照变化支持读出敏感性存在，不解释全部S/U，也不证明信息量改变。U首层满秩时不能用该层碰撞解释差值。没有single/joint四格，不估pooling特异性或正迁移。没有N/H/FISHER强参照，不认证方法竞争力；本仿真不关闭工业C5/C6/C7或T2。

一次固定pass后停止。禁止按query追调参数、删重复、换任务、追加宽度/训练预算或改冻结工业协议。运行全程 `CUDA_VISIBLE_DEVICES=''`、`conda run -n LQ_signal --no-capture-output`，0 GPU、0工业数据读取。

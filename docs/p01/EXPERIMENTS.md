# 实验、基线与创新点对应

## 创新点—证据

K1：阶次与固定 Hz 路径使用各自正确的参数规律。比较固定绑定、正确绑定、错误共振绑定；输入、参数数量和训练预算相同。

K2：同一路径的一致性是否超出相同数据增强。比较无增强、同增强仅监督、同增强加一致性；不能把增强收益计为损失收益。

K3：路径 margin 保持是否超出一般回放或输出蒸馏。比较 ER、DER++ 目标、单边总 margin、对称路径、单边路径、错误路径对应。所有过去、当前、下一工况性能和成本同时报告。

## 已编码条件

| 名称 | 实际内容 | 本次测试状态 |
|---|---|---|
| S1_additive | 相同路径，可加头，周期带固定为源参考转速对应 Hz | 合成训练通过 |
| S2_physical | 周期/包络读出/STFT 阶次绑定，载波保持 Hz | 合成训练通过 |
| S3_augmented | S2 + 圆周时间起点增强 + 配对监督 | 合成训练通过 |
| S4_covariance | S3 + 同路径贡献一致性 | 合成训练通过 |
| N_wrong_physics | 错误地把载波共振也按转速缩放 | 合成训练通过；当前 toy 未明显区分它 |
| N_global_alignment | 强迫不同路径相同的负对照 | 合成训练通过；不作为提出方法 |
| A_mlp | 相同物理路径，非线性 MLP 读出 | 合成训练通过；不产生可加贡献 |
| A_raw/periodic/envelope/stft_only | 单路径消融 | 合成训练通过；不是等参数基线 |
| raw_cnn / stft_cnn | 小型原始波形/二维 CNN，均获得相同采样率与转速 | 合成训练通过；不是当前最优 backbone |
| tspn_metadata | 原 TSPN + 明确的侧信息线性头 | 已编码，完整上游环境尚未运行 |
| vrex / groupdro | 域均值风险方差 / 指数域权重目标，相同 S1 backbone | 合成训练通过；实验级适配 |
| FT / ER | 相同离线 S4 后的顺序微调/单位回放 | 合成序列通过 |
| DERPP | ER + 历史首次见到时 logits MSE，不用上一阶段 teacher 冒充历史 | 合成序列通过；单位级分域适配，不是原论文在线全协议 |
| TOTAL_MARGIN | 单边总分类 margin 保持 | 合成序列通过 |
| PATH_SYMMETRIC | 对称路径 margin 保持 | 合成序列通过 |
| OURS | 单边路径 margin 保持，包含偏置 | 合成序列通过 |
| N_WRONG_PATH | teacher 路径打乱 | 合成序列通过；负对照 |

S1/S2/S3/S4/两个负对照的参数量完全相同；CNN、原 TSPN、MLP 和单路径消融不是等参数结构，不使用“全基线严格参数匹配”的说法。报告总参数与实际秒数。

## 文献强基线：不把未接入方法冒充已完成 SOTA

1. **V-REx**：Krueger et al., ICML 2021, *Out-of-Distribution Generalization via Risk Extrapolation*. https://proceedings.mlr.press/v139/krueger21a.html 。本实现使用经验域风险方差，超参数需源验证；不能继承其论文表格数值。
2. **GroupDRO**：Sagawa et al., ICLR 2020, *Distributionally Robust Neural Networks for Group Shifts*. 采用最坏组目标的指数域加权对照；正则化、数据量与调参预算必须说明。
3. **DER/DER++**：Buzzega et al., NeurIPS 2020, *Dark Experience for General Continual Learning*. https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html 。已实现监督回放与历史-logit 目标；buffer 是域批次结束后更新的单位 reservoir，与原在线流协议不同，必须标注。
4. **LPR**：Yoo et al., ICML 2024, *Layerwise Proximal Replay*. https://proceedings.mlr.press/v235/yoo24a.html 。直接对比路径保持与层表示保持；**尚未实现**。它修改优化几何，不得用 feature MSE 改名冒充。
5. **SYNC**：He et al., ICML 2025, *Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains*. https://proceedings.mlr.press/v267/he25j.html 。可作 evolving-DG 强近邻；**尚未实现**。其时序 VAE/因果目标和可用数据协议不同，单独对齐后比较，不简单套固定 expert。

以上是经过定位的高水平竞争工作，不是“截至今天全部任务排名第一”的名单。LPR/SYNC 尚未接入，当前包不足以完成投稿级 SOTA 总表。

## 有限执行顺序

E0：原生环境与数据字段 → 一个真实数据集 raw/processed/joint 小训练 → 概率重算 → 不含测试数据的调参。

E1：一个机制数据集，S1/S2/S3/S4/错误绑定/错误对齐；跨工况风险矩阵，检验是否物理规则而非任意正则有效。

E2：同一个 S4 初始 checkpoint 规则，FT/ER/DERPP/总 margin/对称路径/OURS/错误路径；永久 test，正向/反向/预声明跳变顺序。

E3：冻结超参数后扩展三组真实数据，每组保留合法标签空间、至少一个未见工况、三 seed。并非三个同源窗口子集。

E4：LPR 与一个适配的强 DG/EDG 方法完成忠实接入后，再形成 SOTA 对比。未运行行保持缺失，不复制原论文不同任务数值。

E5：路径删除、匹配贡献替换，验证计算依赖。完整波形级物理替换、CWT、角域处理、真实变速配对需另行实现与验证，不在当前结果中混称已完成。

## 数据和统计

normal 类使用 0 是本项目约定；真正科学要求是各集合标签语义一致。不在本驱动中自动重新映射未知标签。按试件/独立运行划分。每类 test 支持不足时，报告 represented_classes，不把缺类数据称完整任务评价。

同一物理单元多工况是重复测量。正式统计以物理单元配对，seed 仅为训练重复。Brier、交叉熵和 macro-F1 的结论分别写；当前包不自动产生显著性或总体置信区间。

必须分开 memory 样本字节、历史 targets、teacher 参数、变换计算。保留所有失败与不显著/无差异结果。

## 当前结果边界

已执行 24 个针对性测试、10 个主离线条件、5 个额外消融、7 个持续条件，以及 120 行贡献级删除/替换检查。都是合成数据或纯数值测试。结构绑定在当前 toy 有作用，但同增强与一致性项近乎无差别、错误物理对照也未被可靠排除，故不能宣布 K1–K3 已成立。

极小合成训练中 raw-statistics 单路径未拟合三类阶次任务；这一路径特征本就很弱，不能把其结果当 raw 神经模型的 Stage 0。完整物理模型和周期路径能拟合该构造，但真实 raw CNN、原 TSPN 和整平台 Pipeline 尚未完成验证。

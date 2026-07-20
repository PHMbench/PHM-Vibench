# Physics-Constrained MoE for Explainable Fault Diagnosis

## 标题
Physics-Constrained Mixture-of-Experts for Explainable Fault Diagnosis: A Truth-First Draft

## 摘要
本稿仅同步当前 accepted autoresearch artifacts。当前证据覆盖 CWRU, XJTU，多数据集 bridge 的 mean test accuracy 为 68.75%。三 seed 稳定性报告 mean accuracy=84.72%、std=4.81%、95% CI=5.44 percentage points、CV=5.68%。routing analysis 派生 route entropy mean=0.6522，expert usage distribution=[0.763653039932251, 0.19125157594680786, 0.045095477253198624]。3/5/8 expert ablation 当前仅是 CWRU 上的受限探针，不应外推为完整跨数据集结论。

## 关键词
fault diagnosis, mixture-of-experts, routing interpretability, PHM, truth-first autoresearch

## 1. 引言
- 目标：把 Physics-Constrained MoE 从黑盒门控改成可审计路径级推理系统。
- 本轮稿件只陈述 accepted artifacts 支撑的事实，不补写未验证结论。

## 2. 相关工作
- MoE 在故障诊断中的可解释性缺口。
- 路由可审计与物理先验结合的必要性。

## 3. 方法
- 物理同构专家池：低通、谐波、包络等专家按物理机制分工。
- 当前 routing path signature 示例：LowPassExpert。
- 本稿只引用 accepted routing analysis 中可复现的专家激活与路径统计。

## 4. 实验与结果
- 数据范围：CWRU, XJTU
- Dataset bridge mean test accuracy: 68.75%
- Stability: mean=84.72%, std=4.81%, 95% CI=5.44 pp, CV=5.68%
- Routing entropy mean: 0.6522
- Expert usage distribution: [0.763653039932251, 0.19125157594680786, 0.045095477253198624]
- Expert ablation: 在当前受限预算下，5 experts 在 CWRU 上给出 mean_test_acc=0.3750

## 5. 讨论
- 当前 dataset bridge 与 ablation 都是 bounded probe；结论应按预算范围解释。
- XJTU bridge 达成覆盖，但 3/5/8 ablation 目前只有 CWRU，不能冒充全面泛化结论。

## 6. 结论
- 当前 paper-local accepted artifacts 仅支持内部证据审查 checkpoint。
- 外部投稿 readiness 仍以父仓库 UXFD submission gate 为准，需要扩展更强预算和更完整 ablation/generalization。

## 参考文献
当前由正式稿阶段再补充。

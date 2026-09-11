# 2026-09-11：声明指标与逐 seed 结果发布闭合

分类执行链在每次 test 前，从实际测试 dataset 已选定的 file IDs 取得 Name 集合，并由 Task 使用既有规范化指标名生成 expected keys。实际 metric logging 与 expected keys 共享同一个普通命名函数。

每次 test 后先检查有限标量，再检查配置声明的指标是否完整。缺少指标时，错误列出 seed、声明指标、测试 Name、缺项及实际返回键；当前 seed CSV、汇总 CSV 和 summary 不发布成功。此前完成的 seed 诊断可以保留，不做事务式产物回滚。

- 只声明 acc 的实验仍只要求 acc；不强制增加 f1。
- 允许额外 test loss，不改变 F1/AUROC 等数学定义。
- 同 Name 的不同 Dataset_id 继续按现有协议汇总；不新增身份管理层。
- 测试总体不从完整训练 metadata 或测试后结果反推，也不提前读取样本。
- training-only 不请求测试 loader，不要求测试指标。

HSE 对比预训练保留自己的全局 `test_acc`，不套用普通分类 Task 的 Name 后缀。该任务仅在分类目标启用时计算 acc；要求 evaluated 但未启用分类、或声明它没有计算的 f1 等指标，现在在 Task 构造时失败，不再训练后才发现缺项。纯对比学习仍可显式使用 `test_after_fit=false`。现有预训练模板中“分类权重为零但要求测试 acc”的组合需要用户明确选择训练还是评价；本次不自动改模板、目标权重或声明指标。

新增回归位于现有 `test/test_run_summary.py` 和 `test/test_hse_objective_contract.py`。覆盖同漏 f1、后 seed 缺项、遗漏整个测试 Name、完整正例、既有大小写规范化、额外 loss、Name pooling、acc-only、training-only、非法测试身份，以及 HSE 的实际日志键和矛盾请求前置失败。`accuracy` 不是现有合法 metric ID，测试不新增该别名；结果断言区分输入 metadata CSV 与成功结果 CSV。

低层 summary 保持纯数值汇总；原缺项反例提升至真实分类生命周期，并检查失败产物，而不是把配置解析塞入汇总函数。本次不修改数据划分、模型、训练目标、checkpoint 选择、实验 YAML、依赖或 benchmark 状态。THU 不下载、不重训。相关 CI 结果以对应 PR 的实际最终检查为准。

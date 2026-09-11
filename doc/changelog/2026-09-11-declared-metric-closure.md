# 2026-09-11：声明指标与逐 seed 结果发布闭合

分类执行链在每次 test 前，从实际测试 dataset 已选定的 file IDs 取得 Name 集合，并由 Task 使用既有规范化指标名生成 expected keys。实际 metric logging 与 expected keys 共享同一个普通命名函数。

每次 test 后先检查有限标量，再检查配置声明的指标是否完整。缺少指标时，错误列出 seed、声明指标、测试 Name、缺项及实际返回键；当前 seed CSV、汇总 CSV 和 summary 不发布成功。此前完成的 seed 诊断可以保留，不做事务式产物回滚。

- 只声明 acc 的实验仍只要求 acc；不强制增加 f1。
- 允许额外 test loss，不改变 F1/AUROC 等数学定义。
- 同 Name 的不同 Dataset_id 继续按现有协议汇总；不新增身份管理层。
- 测试总体不从完整训练 metadata 或测试后结果反推，也不提前读取样本。
- training-only 不请求测试 loader，不要求测试指标。

新增回归位于现有 `test/test_run_summary.py`，覆盖同漏 f1、后 seed 缺项、遗漏整个测试 Name、完整正例、别名、额外 loss、Name pooling、acc-only、training-only 及非法测试身份。低层 summary 保持纯数值汇总；原缺项反例提升至真实分类生命周期，并检查失败产物，而不是把配置解析塞入汇总函数。

本次不修改数据划分、模型、训练目标、checkpoint 选择、实验 YAML、依赖或 benchmark 状态。THU 不下载、不重训。相关 CI 结果以对应 PR 的实际最终检查为准。

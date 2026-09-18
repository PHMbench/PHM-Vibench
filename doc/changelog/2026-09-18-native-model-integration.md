# 2026-09-18：原生时间序列分类模型接入

新增 `Transformer.iTransformer`、`CNN.TimesNet`、`CNN.TSLANet` 三个原生分类实现，复用现有 Model Factory、`DG.classification` Task、Trainer 和结果发布路径。没有为每个模型增加 Task、Pipeline 或外部训练 Runner；不增加运行依赖或下载权重。

三个完整实验位于 `configs/experiments/model_integration/`。输入明确为 float32 `[B,L,C]`，输出 logits 或 `(logits, features)`。分类目标、数据划分、checkpoint 选择和指标定义不变。保留上游 MIT 许可及源码位置，并说明原生移植与上游多任务／预训练入口的区别。

本批包含：

- iTransformer 的变量 token、注意力及分类头，不借用预测分支的归一化。
- TimesNet 的频率选择和二维周期卷积；排除 DC，避免常量输入出现零频除法。内部周期补零是该模型原有步骤，不是输入修复。
- TSLANet 的自适应谱模块、交互卷积及显式消融开关；移除脚本级全局参数依赖，不导入未执行的掩码预训练。

本地实际完成 63 项组件测试，覆盖梯度更新、严格参数恢复、评估随机状态、参考公式与非法输入。另有三项真实公共 CLI Dummy 测试，分别执行一个 epoch 并检查 acc/f1、checkpoint 和直接结果路径；这三项的结论以本 PR 最终 CI 为准，不用组件测试代替。

原 187 项表格仍是候选目录。本批不表示剩余模型已接入或测试，不把目录条目、空适配器、跳过用例或缺少权重的模型计为完成。全量合并条件尚未满足，不以本批通过冒充全部完成。不修改 main、其他开放 PR、THU 数据、benchmark 晋级状态或版本发布。

使用与来源见 [原生分类模型说明](../../src/model_factory/NATIVE_MODELS.md)。

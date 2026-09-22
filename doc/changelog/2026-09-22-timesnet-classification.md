# 2026-09-22：TimesNet 分类模型接入

按维护者本轮“先执行模型的合并优化”的优先级，从当前 dev 重建一个单模型实现。不修改无关的 #259，也不重开或整体重放 #266；不继续增加候选规划前置。

新增 `CNN.TimesNet` 和完整 CPU Dummy 配置，复用现有 `DG.classification`、Trainer、checkpoint 与结果路径。原生输入是固定长度、完整观测的 float32 `[B,L,C]`，输出 logits 或 `(logits, features)`。没有新 Task、模型管理器、共享跨模型 helper、运行依赖或权重下载。

E2 使用固定 THUML Time-Series-Library 版本的真实源定义，数值逻辑不改、只重定位测试 imports。测试比较完整分类 logits、CE、输入梯度和全部生效参数梯度，覆盖训练及评估模式。上游未调用的日历参数不进入原生模型；固定位置 buffer 明确保留配置长度。

明确修正：上游将 DC 幅值置零仍不能排除零谱并列时选中 DC。原生版仅将该候选幅值设为负无穷，不修改输入。单独保留上游反例；不把该边界修正称为逐位保真。保留上游批次依赖的频率选择。

本地源码切片的 29 项测试已通过。现有 public-package workflow 增加安装后验证：正常 wheel 环境、离开 checkout、固定上游对齐、preflight、真实 TimesNet fit、selected checkpoint 恢复、acc/f1 和直接结果路径；同时记录实际导入时间、完整 CLI 墙钟和进程 RSS。最终执行结果以本 PR 的当前 head 产物为准，不复用 #266 旧绿灯。

本轮不运行 THU，不声称 E3/benchmark-ready、不修改 main、不发布 tag 或包。使用和来源见 [CNN 文档](../../src/model_factory/CNN/README.md#timesnet-classification)。

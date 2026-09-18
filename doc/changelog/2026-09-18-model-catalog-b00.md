# 2026-09-18：解构模型目录大型 PR（B00）

在 B−1 科学范围进入 dev 后，对模型集成 Draft PR #266 的 48 个 changed paths 做只读
解构，没有重放任何实现。

结果为：

- 1 项与模型目录无关的独立 regularization correctness change；
- 2 个共享 helper，其中 forecasting helper 反向依赖 classification helper；
- 2 项 S2 point-forecasting Task/Data-adapter 合同变更；
- 15 个模型实现；
- 28 个随实现产生的配置、测试、文档、registry 或 CI 变更。

发现 PAttn 直接 import iTransformer 模型文件内部层，因此不能作为独立模型 diff 原样
重放。另有 TSMixer 实现不在保存的 187-row 源 catalogue 中，不能因为已经写过代码而
绕过 Scientific Inclusion Gate。

14 个能映射到原 catalogue 的模型 diff 只记为可审计 source material；没有模型被
B00 晋级为 QUEUED/VERIFIED。forecasting 模型还必须等待独立的 S2 Task contract
裁决。聚合 workflow、bulk registry、branch-level model docs/changelog 不作为未来
bounded PR 的直接 authority。

B00 本身只增加依赖图、48 行 inventory、候选审计表和本记录，不修改 Model、Task、
Data、Trainer、Pipeline、测试、配置、workflow、registry、依赖或版本。

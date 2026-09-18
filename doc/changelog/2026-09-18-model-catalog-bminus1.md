# 2026-09-18：冻结模型目录科学范围（B−1）

本轮把原先“尽可能集成 187 个模型”的数量目标改成“187 个来源候选全部可追溯处置，
同时只实现科学覆盖所需且可维护的模型”。

## 实际保留的耐久内容

- 保留原始 187 个模型来源行与 18 个 Task 规划行；
- 将原工作簿六个 sheet 的单元格值导出到仓库内
  `docs/model_catalog/sources/WORKBOOK_SOURCE_EXPORT.md`，使 fresh clone 能审计
  compact catalogue 省略的字段；
- 冻结 S1 监督诊断、条件 S2 history-only forecasting 以及后续研究问题的 coverage
  边界；
- 冻结 G1–G7 Scientific Inclusion Gate；
- 将 lifecycle / disposition / reason_code 分开；
- 定义 E0 source/legal、E1 framework、E2 algorithm fidelity、E3 PHM scientific
  evidence；
- 为 native trainable、pretrained adapter、non-gradient 和 external service 定义
  不同验收 profile；
- 候选表不进入 production runtime，现有 Model Factory 和 discovery catalogue
  保持原状。

所有 187 个候选在 B−1 结束时仍为 `DISCOVERED`；没有因表格分数、源码存在或 CI
绿色而晋级。

## 资源证据边界

复用了实验分支既有 JUnit 产物作为历史时间基线：321 项测试、0 failure/error/skip，
其中 15 个公共 Dummy。没有把这些旧执行写成新的 E2/E3。

另使用 dev 快照中未修改的 GlobalAverageLinear 源码切片进行三次独立本地进程测量：
PyTorch 导入中位数 0.859135 s，模块增量导入中位数 0.201664 ms，100 次无梯度前向
中位数 2.083534 ms，进程 VmHWM 最高 250.25 MiB。该 RSS 包含解释器和框架，不是模型
净内存或训练峰值。

完整 package import、训练峰值、正常 wheel 依赖增量和真实 PHM/GPU 预算没有在 B−1
中测量，不能据此自动通过 G7。

## 独立审阅修订

B−1 候选 PR 的独立 Codex 审阅提出 2 个 P1 和 1 个 P2：

1. 不应把对话 prompt 存档提交为第二套 AI 指令正文；
2. 版本文档不应保留实时 PR gate 和“下一步”命令；
3. 原始 workbook 内容需要仓库内可审计入口。

三项均已修正：prompt 正文从版本库删除，耐久决策保留在 policy/coverage/changelog；
acceptance 页面仅记录稳定结论；六个 workbook sheet 的单元格值提交为 source export。
对应 review threads 已回复并关闭。

第二次自动 `@codex review` 请求在本轮执行窗口内没有生成新的 Codex review 对象，
因此没有把它描述成第二次独立批准。最终合并仍以当前 head 的实际 CI、已解决的独立
review findings 与无未解决线程为依据。

## 未改变

本轮没有修改模型、Task、Data、Trainer、Pipeline、配置、workflow、依赖、benchmark
状态或版本号；没有重新训练 THU、下载外部权重或发布软件包。对话中的完整 prompt 和
原始 XLSX 仍保留在维护者的外部交付包，但不作为版本库内的执行 authority。

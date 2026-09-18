# Model catalogue — B−1 科学范围冻结

本轮继续 B−1 验收，补做现有简单基线的独立进程资源测量；尚不开始模型接入或 #266 拆分。

## 阅读顺序

1. [科学覆盖矩阵](BENCHMARK_COVERAGE.md)：科学问题、角色和 Inclusion Gate。
2. [候选处置与验收](MODEL_INTEGRATION_POLICY.md)：身份、正交状态、E0–E3、profiles 和独立审查。
3. [资源预算](RESOURCE_BUDGET.md)：已有 CI、本轮本地切片补测、暂定时间警戒值和未测维度。
4. [候选表](MODEL_CATALOGUE.csv)：保留 187 个原始来源行，当前全部 DISCOVERED。
5. [来源材料](sources/README.md)：原 187 模型及 18 Task 表格的保存与导出范围。

## 保存的后续提示词

[Plan v2](prompts/PLAN_V2.md)；[B−1](prompts/Bminus1.md)；[B00](prompts/B00.md)。这些是本项目的显式执行材料，不是新的全局 Agent 指令或自动执行队列。B00 只保存，不执行。

## 本轮已核验

- dev：`b4c3209c7cf7c59d0a83fab7189716f291dcc96a`。
- #266：Draft、未合并，head `c31c888ec85c18c3db2f2cd2af89c5ec3e840867`，21 commits、48 files。
- 原始 XLSX：187 个模型来源行、18 个 Task 规划行、6 张表。
- 既有 native CI：重新读取 321 项 JUnit，0 fail/error/skip，其中 15 项公共 Dummy；没有重跑这些测试。
- 新测量：三个独立进程读取 dev 中未修改的 GlobalAverageLinear 源码；仅 import 和无梯度前向，没有 fit。见 [原始观测](RESOURCE_LOCAL_BASELINE.csv)。

## 执行澄清

当前 model_registry.csv 实为模块目录，Model Factory 动态解析 type/name；B−1 不按旧方案将它重建为另一 runtime registry。候选表不参与生产运行。

原表没有逐条精确 variant/ref/license。为避免虚构，source:Mnnn 是暂定导入身份；最终语义身份按 name/variant/task/source_ref 审查后确定。没有把 187 行标成已审或已裁决。

CI 时间基线与局部推断进程 RSS/导入测量已取得。完整 PHMFactory package import、训练峰值、独立安装依赖增量仍未测，因此完整 G7 数值校准未完成。局部测量不等于所有候选通过，作者复核也不等于 fresh-context 独立审查。

## 本轮不做

不向 #266 追加提交、评论、关闭或合并；不修改 src、configs、workflow 或 runtime registry；不做 B00 依赖图、不创建模型实现 PR；不修改 dev/main，不下载权重/THU，不创建 tag 或发布软件包。当前推进仅发生在现有 #267 文档分支。

文档保存在独立 Draft 分支供审阅。后续先审阅这些范围决定及未测项；不能自动执行下一阶段。

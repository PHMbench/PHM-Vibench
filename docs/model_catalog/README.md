# Model catalogue — B−1 科学范围冻结

本轮只保存材料并完成 B−1 文档工作，尚不开始任何模型接入或 #266 拆分。

## 阅读顺序

1. [科学覆盖矩阵](BENCHMARK_COVERAGE.md)：科学问题、角色和 Inclusion Gate。
2. [候选处置与验收](MODEL_INTEGRATION_POLICY.md)：身份、正交状态、E0–E3、profiles 和独立审查。
3. [资源预算](RESOURCE_BUDGET.md)：已有执行数据、暂定时间警戒值和未测维度。
4. [候选表](MODEL_CATALOGUE.csv)：保留 187 个原始来源行，当前全部 DISCOVERED。
5. [来源材料](sources/README.md)：原 187 模型及 18 Task 表格的保存与导出范围。

## 保存的后续提示词

[Plan v2](prompts/PLAN_V2.md)；[B−1](prompts/Bminus1.md)；[B00](prompts/B00.md)。这些是本项目的显式执行材料，不是新的全局 Agent 指令或自动执行队列。B00 只保存，不执行。

## 本轮已核验

- dev：`b4c3209c7cf7c59d0a83fab7189716f291dcc96a`。
- #266：Draft、未合并，head `c31c888ec85c18c3db2f2cd2af89c5ec3e840867`，21 commits、48 files。
- 原始 XLSX：187 个模型来源行、18 个 Task 规划行、6 张表。
- 既有 native CI：读取 321 项测试 JUnit，0 fail/error/skip，其中 15 项公共 Dummy，未重跑。

## 执行澄清

当前 model_registry.csv 实为模块目录，Model Factory 动态解析 type/name；B−1 不按旧方案将它重建为另一 runtime registry。候选表不参与生产运行。

原表没有逐条精确 variant/ref/license。为避免虚构，source:Mnnn 是暂定导入身份；最终语义身份按 name/variant/task/source_ref 审查后确定。没有把 187 行标成已审或已裁决。

时间基线已取得，暂定警戒值已给出；峰值内存、冷 import、依赖增量尚无测量，因此完整 G7 数值校准未完成。冻结规则不等于 187 项合格，也不等于预算全面验证。

## 本轮不做

不向 #266 追加提交、评论、关闭或合并；不修改 src、configs、workflow 或 runtime registry；不做 B00 依赖图、不创建模型实现 PR；不修改 dev/main，不下载权重/THU，不创建 tag 或发布软件包。

文档保存在独立 Draft 分支供审阅。后续先审阅这些范围决定及未测项；不能自动执行下一阶段。

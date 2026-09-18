# 2026-09-18：保存模型目录并执行 B−1

本轮保存 Plan v2、B−1/B00 提示词整理版，保留原 187 模型来源行、18 Task 规划行。完整原 XLSX、六张表 CSV 和旧计划已保存在交付包；仓库文档分支保存工作 catalogue 与 Task 原表，不以新表替换原始档案。

B−1 冻结科学覆盖范围、Scientific Inclusion Gate、正交状态和候选身份规则、E0–E3、四类验收 profiles、独立审阅要求及资源记录方法。当前 registry 保持原状，候选表不进入 runtime。

复用 #266 当前 head 的既有 CI JUnit：321 项、零 fail/error/skip、15 个公共 Dummy，用于计时基线，没有重新训练。峰值内存、冷 import 和依赖体积无记录，不编数值；完整 G7 资源校准仍待相应测量。

本轮全部候选保持 DISCOVERED，没有取得集成或 Benchmark-ready 声明。B00 只保存，不解构/关闭 #266。

独立文档分支基于 dev@b4c3209c；不向 #266 添加第 22 个提交，不改 dev/main，不修改模型、Task、数据、测试、workflow、配置或 benchmark 状态，不下载权重/THU，不发布包或 tag。

入口：[B−1 文档](../../docs/model_catalog/README.md)。

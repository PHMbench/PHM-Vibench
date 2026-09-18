# 2026-09-18：保存模型目录并执行 B−1

本轮保存 Plan v2、B−1/B00 提示词整理版，保留原 187 模型来源行、18 Task 规划行。完整原 XLSX、六张表 CSV 和旧计划已保存在交付包；仓库文档分支保存工作 catalogue 与 Task 原表，不以新表替换原始档案。

B−1 冻结科学覆盖范围、Scientific Inclusion Gate、正交状态和候选身份规则、E0–E3、四类验收 profiles、独立审阅要求及资源记录方法。当前 registry 保持原状，候选表不进入 runtime。

复用 #266 当前 head 的既有 CI JUnit：321 项、零 fail/error/skip、15 个公共 Dummy，用于计时基线，没有重新训练。峰值内存、冷 import 和依赖体积无记录，不编数值；完整 G7 资源校准仍待相应测量。

本轮全部候选保持 DISCOVERED，没有取得集成或 Benchmark-ready 声明。B00 只保存，不解构/关闭 #266。

独立文档分支基于 dev@b4c3209c；不向 #266 添加第 22 个提交，不改 dev/main，不修改模型、Task、数据、测试、workflow、配置或 benchmark 状态，不下载权重/THU，不发布包或 tag。

入口：[B−1 文档](../../docs/model_catalog/README.md)。

## 同日继续验收

复核了原始 catalogue 的 187 个来源 ID、名称和来源位置，以及已有 321 项 JUnit；原表和模型状态没有改写。

本轮新增三次独立 Python 进程的实际测量，使用 dev@b4c3209c 中未修改的 GlobalAverageLinear 源码切片：PyTorch 导入中位数 0.859135 s，单模块增量导入中位数 0.201664 ms，100 次推断中位数 2.083534 ms，进程 VmHWM 最大值 250.250000 MiB。配置为 CPU 单线程、float32 [4,128,2]、两个类别。原始逐次值写入 RESOURCE_LOCAL_BASELINE.csv。

这不是完整安装、Trainer 生命周期或 PHM 训练验证；没有新增/修改模型。不同于既有 CI 环境，本轮是 Python 3.13.5 / torch 2.10.0+cpu，禁止混合推导通用阈值。完整 package import、训练峰值和依赖安装增量仍未测，独立审查未执行。

只更新 #267 的资源说明、观测表和入口。#266 继续保持原 head 与 21 commits；B00、dev 合并和主线推广未执行。
